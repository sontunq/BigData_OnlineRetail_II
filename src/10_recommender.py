"""Product recommendation - popularity baselines vs item-based CF vs ALS (implicit).

Data: EFFECTIVE product sales with a customer id (Hive). Interaction = customer bought the
stock code; strength = number of distinct invoices containing it.

Temporal evaluation (no look-ahead):
  TRAIN  invoices before CUTOFF (2011-09-01)
  TEST   invoices from CUTOFF to the end of data (2011-12-09; includes the Christmas peak)
  ALS hyper-parameters are tuned on an earlier split INSIDE the training period
  (fit < 2011-06-01, validate 2011-06-01..2011-08-31), then refitted on all of TRAIN.
Evaluated customers: bought in TRAIN and in TEST. Two targets:
  all  relevant = all distinct items bought in TEST (repeat purchases count)
  new  relevant = items bought in TEST that the customer never bought in TRAIN; already
       bought items are removed from the recommendation lists
Metrics @K: hit rate, precision, recall, MAP (AP normalised by min(K, |relevant|)).
Cold start: customers whose first purchase is in TEST; their first invoice is the only
history, relevant = items in their later TEST invoices. ALS cannot score them without
retraining (no user factors), so only popularity and item-CF are compared.

Models
  pop_all     most purchased items (distinct customers) over all of TRAIN
  pop_recent  same, last 90 days of TRAIN
  item_cf     item-item cosine similarity on the binary customer x item matrix of TRAIN;
              score(u, j) = sum of sim(i, j) over items i bought by u. Computed with SciPy
              sparse products on the driver (matrix is ~5k x 4k); Spark prepares the data.
  als         Spark MLlib ALS, implicitPrefs, rating = log1p(invoices containing the item)
"""
import json
import time

import numpy as np
import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from scipy import sparse

from common import HIVE_DB, SEED, TABLES_DIR, get_spark, save_table, to_pandas
from plotstyle import SERIES, TEXT_2, plt, save

DB = HIVE_DB
CUTOFF = "2011-09-01"
VAL_FIT_END, VAL_END = "2011-06-01", "2011-09-01"
KS = [5, 10, 20]
K_MAIN = 10
# First run: best was rank 50 / alpha 10, both at the edge of a (20, 50) x (10, 40) grid, so the
# grid was widened to rank 100 and alpha 1.
ALS_GRID = [dict(rank=r, regParam=reg, alpha=a) for r in (20, 50, 100) for reg in (0.01, 0.1)
            for a in (1.0, 10.0, 40.0)]
N_BOOT = 1000
MODELS = ["pop_all", "pop_recent", "item_cf", "als"]
MODEL_LABEL = {"pop_all": "Popularity (all train)", "pop_recent": "Popularity (last 90 days)",
               "item_cf": "Item-based CF", "als": "ALS (implicit)"}


# ---------------------------------------------------------------- data access
def interactions(spark, start, end):
    """customer x item with invoice counts for effective sales in [start, end)."""
    return spark.sql(f"""
        SELECT customer_id, stock_code, COUNT(DISTINCT invoice) AS n_inv,
               MIN(invoice_ts) AS first_ts
        FROM {DB}.fact_sales
        WHERE txn_type = 'sale' AND is_product AND NOT is_reversed AND has_customer
          AND invoice_ts >= TIMESTAMP '{start}' AND invoice_ts < TIMESTAMP '{end}'
        GROUP BY customer_id, stock_code""")


def truth_sets(df_pd):
    return df_pd.groupby("customer_id")["stock_code"].apply(set).to_dict()


# ---------------------------------------------------------------- metrics
def per_user_ap(recs, truth, users, k=K_MAIN):
    out = {}
    for u in users:
        rel = truth.get(u, set())
        if not rel:
            continue
        flags = np.array([it in rel for it in recs.get(u, [])[:k]], dtype=float)
        if flags.sum():
            prec_at = np.cumsum(flags) / (np.arange(len(flags)) + 1)
            out[u] = float((prec_at * flags).sum() / min(k, len(rel)))
        else:
            out[u] = 0.0
    return out


def paired_bootstrap(ap_a, ap_b, n=N_BOOT, seed=SEED):
    users = sorted(set(ap_a) & set(ap_b))
    diff = np.array([ap_a[u] - ap_b[u] for u in users])
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    return {"mean_diff": float(diff.mean()), "ci95_low": float(np.percentile(boots, 2.5)),
            "ci95_high": float(np.percentile(boots, 97.5)),
            "share_boot_diff_gt_0": float((boots > 0).mean()), "users": len(users)}


def evaluate(recs, truth, users, ks=KS):
    out = {}
    for k in ks:
        hits, precs, recs_, aps = [], [], [], []
        for u in users:
            rel = truth.get(u, set())
            if not rel:
                continue
            lst = recs.get(u, [])[:k]
            flags = np.array([it in rel for it in lst], dtype=float)
            n_hit = flags.sum()
            hits.append(float(n_hit > 0))
            precs.append(n_hit / k)
            recs_.append(n_hit / len(rel))
            if n_hit:
                prec_at = np.cumsum(flags) / (np.arange(len(flags)) + 1)
                aps.append(float((prec_at * flags).sum() / min(k, len(rel))))
            else:
                aps.append(0.0)
        out.update({f"hit_rate@{k}": np.mean(hits), f"precision@{k}": np.mean(precs),
                    f"recall@{k}": np.mean(recs_), f"map@{k}": np.mean(aps)})
    out["users"] = len(hits)
    return out


# ---------------------------------------------------------------- models
def popularity(inter_pd):
    """Items ranked by number of distinct customers in the given interaction table
    (ties broken by stock code for determinism)."""
    c = inter_pd.groupby("stock_code")["customer_id"].nunique().reset_index()
    return c.sort_values(["customer_id", "stock_code"], ascending=[False, True])["stock_code"].tolist()


def pop_recs(ranking, users, seen=None, n=max(KS)):
    out = {}
    for u in users:
        s = seen.get(u, set()) if seen else set()
        out[u] = [i for i in ranking if i not in s][:n] if s else ranking[:n]
    return out


class ItemCF:
    def __init__(self, train_pd):
        self.items = sorted(train_pd["stock_code"].unique())
        self.iidx = {it: j for j, it in enumerate(self.items)}
        users = sorted(train_pd["customer_id"].unique())
        uidx = {u: i for i, u in enumerate(users)}
        X = sparse.csr_matrix((np.ones(len(train_pd)),
                               (train_pd["customer_id"].map(uidx), train_pd["stock_code"].map(self.iidx))),
                              shape=(len(users), len(self.items)))
        X.data[:] = 1.0
        co = (X.T @ X).tocsr().astype(np.float32)
        norm = np.sqrt(co.diagonal())
        norm[norm == 0] = 1
        inv = sparse.diags(1 / norm)
        sim = (inv @ co @ inv).tocsr()
        sim.setdiag(0)
        sim.eliminate_zeros()
        self.sim = sim

    def recommend(self, histories, exclude_seen, n=max(KS)):
        out = {}
        for u, hist in histories.items():
            idx = [self.iidx[i] for i in hist if i in self.iidx]
            if not idx:
                continue
            scores = np.asarray(self.sim[idx].sum(axis=0)).ravel()
            if exclude_seen:
                scores[idx] = -np.inf
            top = np.argpartition(-scores, n)[:n]
            top = top[np.argsort(-scores[top])]
            out[u] = [self.items[j] for j in top if scores[j] > 0]
        return out


def fit_als(spark_inter, params, seed=SEED):
    als = ALS(userCol="customer_id", itemCol="item_idx", ratingCol="rating", implicitPrefs=True,
              nonnegative=True, coldStartStrategy="drop", maxIter=10, seed=seed, **params)
    return als.fit(spark_inter)


def als_recs(model, spark, users, idx_to_item, seen=None, n=max(KS)):
    """Top-n items per user scored over the FULL catalogue with the learned factors
    (score = user_factor . item_factor), excluding `seen` items.

    The first version used recommendForUserSubset(N_CAND=200) and filtered seen items
    afterwards; heavy buyers who had already bought most of those 200 items got fewer than
    n recommendations, which penalised ALS in the "new items" evaluation. Scoring the full
    catalogue removes that truncation. Factors are collected to the driver (small matrices).
    """
    uf = to_pandas(model.userFactors)
    itf = to_pandas(model.itemFactors)
    urow = {int(i): k for k, i in enumerate(uf["id"])}
    U = np.array(uf["features"].tolist(), dtype=np.float32)
    V = np.array(itf["features"].tolist(), dtype=np.float32)
    item_codes = [idx_to_item[int(i)] for i in itf["id"]]
    col = {c: j for j, c in enumerate(item_codes)}
    out = {}
    users = [u for u in users if int(u) in urow]
    for start in range(0, len(users), 512):
        batch = users[start:start + 512]
        S = U[[urow[int(u)] for u in batch]] @ V.T
        for r, u in enumerate(batch):
            s = S[r]
            if seen:
                idx = [col[c] for c in seen.get(u, ()) if c in col]
                s[idx] = -np.inf
            top = np.argpartition(-s, n)[:n]
            top = top[np.argsort(-s[top])]
            out[u] = [item_codes[j] for j in top if np.isfinite(s[j])]
    return out


def als_input(spark, inter_sdf, indexer_model):
    return (indexer_model.transform(inter_sdf)
            .withColumn("item_idx", F.col("item_idx").cast("int"))
            .withColumn("rating", F.log1p(F.col("n_inv").cast("double"))))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    spark = get_spark("10_recommender")
    first = spark.sql(f"SELECT MIN(invoice_ts) a, MAX(invoice_ts) b FROM {DB}.fact_sales").first()
    end = str((pd.Timestamp(first["b"]) + pd.Timedelta(days=1)).date())
    start = "2009-12-01"

    train_s = interactions(spark, start, CUTOFF).cache()
    test_s = interactions(spark, CUTOFF, end).cache()
    train = to_pandas(train_s)
    test = to_pandas(test_s)
    train_hist = truth_sets(train)
    test_all = truth_sets(test)
    warm_users = sorted(set(train_hist) & set(test_all))
    test_new = {u: test_all[u] - train_hist[u] for u in warm_users}
    test_new = {u: s for u, s in test_new.items() if s}
    train_items = set(train["stock_code"])
    test_items = set(test["stock_code"])

    split = {
        "cutoff": CUTOFF, "train_customers": len(train_hist), "train_items": len(train_items),
        "train_pairs": len(train), "test_customers": len(test_all), "test_pairs": len(test),
        "warm_eval_customers": len(warm_users), "warm_customers_with_new_items": len(test_new),
        "test_customers_new_in_test": len(set(test_all) - set(train_hist)),
        "test_items_not_in_train_pct_of_test_items":
            round(100 * len(test_items - train_items) / len(test_items), 2),
        "matrix_density_pct": round(100 * len(train) / (len(train_hist) * len(train_items)), 3),
    }
    print(split, flush=True)

    # ---------------- ALS tuning on an earlier split inside TRAIN ----------------
    indexer = StringIndexer(inputCol="stock_code", outputCol="item_idx").fit(train_s)
    idx_to_item = dict(enumerate(indexer.labels))
    v_fit_s = interactions(spark, start, VAL_FIT_END)
    v_val = to_pandas(interactions(spark, VAL_FIT_END, VAL_END))
    v_hist = truth_sets(to_pandas(v_fit_s))
    v_truth = truth_sets(v_val)
    v_users = sorted(set(v_hist) & set(v_truth))
    v_fit_in = als_input(spark, v_fit_s, indexer).dropna().cache()
    tuning = []
    for p in ALS_GRID:
        m = fit_als(v_fit_in, p)
        r = als_recs(m, spark, v_users, idx_to_item)
        res = evaluate(r, v_truth, v_users, ks=[K_MAIN])
        tuning.append({**p, **{k: v for k, v in res.items() if k != "users"}, "users": res["users"]})
        print("ALS tuning", tuning[-1], flush=True)
    tuning = pd.DataFrame(tuning).sort_values(f"map@{K_MAIN}", ascending=False)
    save_table(tuning, "rec_als_tuning")
    best = tuning.iloc[0][["rank", "regParam", "alpha"]].to_dict()
    best = {"rank": int(best["rank"]), "regParam": float(best["regParam"]), "alpha": float(best["alpha"])}

    # ---------------- fit all models on TRAIN ----------------
    pop_all = popularity(train)
    since = str((pd.Timestamp(CUTOFF) - pd.Timedelta(days=90)).date())
    pop_recent = popularity(to_pandas(interactions(spark, since, CUTOFF)))
    cf = ItemCF(train)
    als = fit_als(als_input(spark, train_s, indexer).cache(), best)

    rows, recs_by = [], {}
    for mode, truth, users, excl in [("all", test_all, warm_users, False),
                                     ("new", test_new, sorted(test_new), True)]:
        seen = train_hist if excl else None
        recs = {
            "pop_all": pop_recs(pop_all, users, seen),
            "pop_recent": pop_recs(pop_recent, users, seen),
            "item_cf": cf.recommend({u: train_hist[u] for u in users}, exclude_seen=excl),
            "als": als_recs(als, spark, users, idx_to_item, seen),
        }
        recs_by[mode] = recs
        for name in MODELS:
            res = evaluate(recs[name], truth, users)
            topk = [it for u in users for it in recs[name].get(u, [])[:K_MAIN]]
            res.update(model=name, mode=mode,
                       catalog_coverage_pct=100 * len(set(topk)) / len(train_items),
                       users_with_recs_pct=100 * sum(1 for u in users if recs[name].get(u)) / len(users))
            rows.append(res)
    metrics = pd.DataFrame(rows)
    cols = ["mode", "model", "users"] + [c for c in metrics.columns if "@" in c] + \
        ["catalog_coverage_pct", "users_with_recs_pct"]
    metrics = metrics[cols]
    save_table(metrics, "rec_metrics")

    # paired bootstrap of MAP@10 differences (per-customer AP, same customers)
    boot = []
    for mode, truth in [("all", test_all), ("new", test_new)]:
        users = warm_users if mode == "all" else sorted(test_new)
        aps = {m: per_user_ap(recs_by[mode][m], truth, users) for m in MODELS}
        for a, b in [("als", "item_cf"), ("item_cf", "pop_all"), ("als", "pop_all"),
                     ("item_cf", "pop_recent"), ("als", "pop_recent")]:
            boot.append({"mode": mode, "model_a": a, "model_b": b,
                         **paired_bootstrap(aps[a], aps[b])})
    boot = pd.DataFrame(boot)
    save_table(boot, "rec_bootstrap_map10")

    # popularity bias: mean train popularity rank of recommended items (lower = more popular)
    pop_rank = {it: r + 1 for r, it in enumerate(pop_all)}
    bias = []
    for name in MODELS:
        ranks = [pop_rank.get(it) for u in warm_users for it in recs_by["all"][name].get(u, [])[:K_MAIN]]
        ranks = [r for r in ranks if r]
        bias.append({"model": name, "median_popularity_rank_of_recs": float(np.median(ranks)),
                     "share_recs_in_top100_popular_pct": 100 * float(np.mean(np.array(ranks) <= 100))})
    save_table(pd.DataFrame(bias), "rec_popularity_bias")

    # ---------------- cold start: new customers in TEST ----------------
    first_inv = to_pandas(spark.sql(f"""
        WITH t AS (SELECT customer_id, invoice, invoice_ts, stock_code FROM {DB}.fact_sales
                   WHERE txn_type='sale' AND is_product AND NOT is_reversed AND has_customer
                     AND invoice_ts >= TIMESTAMP '{CUTOFF}'),
             f AS (SELECT customer_id, MIN(invoice_ts) AS t0 FROM {DB}.fact_sales
                   WHERE txn_type='sale' AND is_product AND NOT is_reversed AND has_customer
                   GROUP BY customer_id)
        SELECT t.customer_id, t.stock_code, (t.invoice_ts = f.t0) AS is_first
        FROM t JOIN f ON t.customer_id = f.customer_id
        WHERE f.t0 >= TIMESTAMP '{CUTOFF}'"""))
    hist_cs = first_inv[first_inv["is_first"]].groupby("customer_id")["stock_code"].apply(set).to_dict()
    later = first_inv[~first_inv["is_first"]].groupby("customer_id")["stock_code"].apply(set).to_dict()
    cs_users = sorted(u for u in later if u in hist_cs)
    cs_truth = {u: later[u] for u in cs_users}
    cs = []
    for name, recs in [("pop_all", pop_recs(pop_all, cs_users)),
                       ("pop_recent", pop_recs(pop_recent, cs_users)),
                       ("item_cf", cf.recommend({u: hist_cs[u] for u in cs_users}, exclude_seen=False))]:
        res = evaluate(recs, cs_truth, cs_users)
        res["model"] = name
        cs.append(res)
    cs = pd.DataFrame(cs)
    save_table(cs, "rec_cold_start_metrics")

    # ---------------- example recommendations ----------------
    desc = to_pandas(spark.sql(f"SELECT stock_code, description FROM {DB}.dim_product")) \
        .set_index("stock_code")["description"].to_dict()
    ex_users = warm_users[:: max(1, len(warm_users) // 3)][:3]
    ex = []
    for u in ex_users:
        recent = sorted(train_hist[u])[:5]
        for name in ["item_cf", "als", "pop_all"]:
            for rank, it in enumerate(recs_by["all"][name].get(u, [])[:5], 1):
                ex.append({"customer_id": u, "model": name, "rank": rank, "stock_code": it,
                           "description": desc.get(it), "bought_in_test": it in test_all.get(u, set())})
    save_table(pd.DataFrame(ex), "rec_examples")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=False)
    mets = [f"hit_rate@{K_MAIN}", f"precision@{K_MAIN}", f"recall@{K_MAIN}", f"map@{K_MAIN}"]
    for ax, mode, title in [(axes[0], "all", "All items bought in test period"),
                            (axes[1], "new", "Only items new to the customer")]:
        d = metrics[metrics["mode"] == mode].set_index("model").loc[MODELS]
        x = np.arange(len(mets))
        w = 0.2
        for i, name in enumerate(MODELS):
            vals = d.loc[name, mets].values.astype(float)
            ax.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=SERIES[i], label=MODEL_LABEL[name])
            for xi, v in zip(x + (i - 1.5) * w, vals):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=TEXT_2)
        ax.set_xticks(x, [m.replace(f"@{K_MAIN}", "") .replace("_", " ") + f"@{K_MAIN}" for m in mets])
        ax.set_title(f"{title} (n={int(d['users'].iloc[0]):,} customers)", fontsize=10)
        ax.set_ylim(0, 1.0)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Recommender evaluation: train < {CUTOFF}, test {CUTOFF}..2011-12-09",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    save(fig, "rec_metrics")

    summary = {**split, "als_best_params": best, "als_grid_size": len(ALS_GRID),
               "cold_start_customers": len(cs_users), "runtime_seconds": round(time.time() - t0, 1)}
    (TABLES_DIR / "rec_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 250)
    print(json.dumps(summary, indent=2))
    print(tuning.round(4).to_string(index=False))
    print(metrics.round(4).to_string(index=False))
    print(pd.DataFrame(bias).round(2).to_string(index=False))
    print(cs.round(4).to_string(index=False))
    print(boot.round(4).to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
