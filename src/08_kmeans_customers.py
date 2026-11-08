"""K-Means customer segmentation with Spark MLlib (reads Hive, writes
retail.customer_clusters).

Population: customers in retail.customer_rfm with net monetary > 0.
Features (log1p where skewed, then standardised to mean 0 / sd 1):
  log_recency        log1p(days since last purchase)
  log_frequency      log1p(distinct invoices)
  log_avg_invoice    log1p(net monetary / invoices)     -> ticket size
  log_n_products     log1p(distinct stock codes bought) -> assortment breadth
  tenure_days        days from first purchase to the reference date
  peak_season_share  share of effective sales in September-December
Monetary itself is NOT a feature: monetary = frequency x avg invoice value, so it would be
redundant (Spearman F-M = 0.86 in 07). It is used to profile the clusters.

Selection rule (fixed before running): scan k = 2..10 for
silhouette and WSSSE; fit k = 3..8 with 5 seeds and compute the mean pairwise ARI between
seeds; among k >= 3 whose silhouette is within 0.01 of the best k >= 3, pick the k with the
highest mean ARI (ties -> smaller k). The final model is the lowest-WSSSE fit among the 5 seeds.
"""
from itertools import combinations
import json

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from sklearn.metrics import adjusted_rand_score

from common import HIVE_DB, SEED, TABLES_DIR, get_spark, save_table, to_pandas
from plotstyle import BLUE_DARK, SERIES, TEXT_2, plt, save

DB = HIVE_DB
FEATURES = ["log_recency", "log_frequency", "log_avg_invoice", "log_n_products", "tenure_days",
            "peak_season_share"]
K_SCAN = range(2, 11)
K_STAB = range(3, 9)
SEEDS = [SEED, 1, 7, 123, 2024]
SIL_TOL = 0.01

FEATURE_SQL = f"""
WITH eff AS (
  SELECT customer_id, stock_code, MONTH(invoice_ts) AS m, CAST(revenue AS DOUBLE) AS rev
  FROM {DB}.fact_sales
  WHERE txn_type = 'sale' AND is_product AND NOT is_reversed AND has_customer
),
agg AS (
  SELECT customer_id, COUNT(DISTINCT stock_code) AS n_products,
         SUM(CASE WHEN m >= 9 THEN rev ELSE 0 END) / SUM(rev) AS peak_season_share
  FROM eff GROUP BY customer_id
)
SELECT r.customer_id, r.country, r.segment AS rfm_segment, r.recency_days, r.frequency,
       r.monetary, r.monetary / r.frequency AS avg_invoice, a.n_products,
       DATEDIFF(r.ref_date, TO_DATE(r.first_purchase_ts)) AS tenure_days,
       a.peak_season_share,
       LN(1 + r.recency_days) AS log_recency, LN(1 + r.frequency) AS log_frequency,
       LN(1 + r.monetary / r.frequency) AS log_avg_invoice, LN(1 + a.n_products) AS log_n_products
FROM {DB}.customer_rfm r JOIN agg a ON r.customer_id = a.customer_id
WHERE r.monetary > 0
"""


def fit(data, k, seed):
    return KMeans(k=k, seed=seed, maxIter=100, tol=1e-6, featuresCol="features",
                  predictionCol="cluster").fit(data)


def main():
    spark = get_spark("08_kmeans_customers")
    cust = spark.sql(FEATURE_SQL)
    for c in FEATURES + ["avg_invoice", "monetary"]:
        cust = cust.withColumn(c, cust[c].cast("double"))
    cust = cust.cache()
    n_cust = cust.count()
    n_excluded = spark.sql(f"SELECT COUNT(*) FROM {DB}.customer_rfm WHERE monetary <= 0").first()[0]

    prep = Pipeline(stages=[VectorAssembler(inputCols=FEATURES, outputCol="raw"),
                            StandardScaler(inputCol="raw", outputCol="features",
                                           withMean=True, withStd=True)]).fit(cust)
    data = prep.transform(cust).cache()
    ev = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                             metricName="silhouette", distanceMeasure="squaredEuclidean")

    # ---------- k scan (seed 42) ----------
    scan = []
    for k in K_SCAN:
        m = fit(data, k, SEED)
        scan.append({"k": k, "wssse": m.summary.trainingCost, "silhouette": ev.evaluate(m.transform(data))})
        print(f"k={k} silhouette={scan[-1]['silhouette']:.4f} wssse={scan[-1]['wssse']:,.0f}", flush=True)
    scan = pd.DataFrame(scan)
    scan["wssse_drop_pct"] = -100 * scan["wssse"].pct_change()

    # ---------- stability ----------
    runs, stab, models = [], [], {}
    for k in K_STAB:
        labels = {}
        for seed in SEEDS:
            m = fit(data, k, seed)
            pred = m.transform(data)
            labels[seed] = to_pandas(pred.select("customer_id", "cluster")) \
                .set_index("customer_id")["cluster"].sort_index()
            models[(k, seed)] = m
            runs.append({"k": k, "seed": seed, "wssse": m.summary.trainingCost,
                         "silhouette": ev.evaluate(pred),
                         "cluster_sizes": str(sorted(m.summary.clusterSizes, reverse=True))})
        aris = [adjusted_rand_score(labels[a].values, labels[b].values)
                for a, b in combinations(SEEDS, 2)]
        stab.append({"k": k, "mean_pairwise_ari": float(np.mean(aris)),
                     "min_pairwise_ari": float(np.min(aris))})
        print(f"stability k={k}: mean ARI={np.mean(aris):.3f} min={np.min(aris):.3f}", flush=True)
    save_table(pd.DataFrame(runs), "kmeans_stability_runs")
    scan = scan.merge(pd.DataFrame(stab), on="k", how="left")
    best_sil = scan.loc[scan["k"] >= 3, "silhouette"].max()
    scan["candidate"] = scan["k"].isin(list(K_STAB)) & (scan["silhouette"] >= best_sil - SIL_TOL)
    cand = scan[scan["candidate"]].sort_values(["mean_pairwise_ari", "k"], ascending=[False, True])
    k_best = int(cand["k"].iloc[0])
    scan["chosen"] = scan["k"] == k_best
    save_table(scan, "kmeans_k_selection")

    best_seed = min(SEEDS, key=lambda s: models[(k_best, s)].summary.trainingCost)
    final = models[(k_best, best_seed)]
    out = to_pandas(final.transform(data).select(
        "customer_id", "country", "rfm_segment", "recency_days", "frequency", "monetary",
        "avg_invoice", "n_products", "tenure_days", "peak_season_share", *FEATURES[:4], "cluster"))
    # Relabel clusters by descending total monetary so C0 is the most valuable cluster
    order = out.groupby("cluster")["monetary"].sum().sort_values(ascending=False).index
    out["cluster"] = out["cluster"].map({c: i for i, c in enumerate(order)})
    save_table(out, "kmeans_customer_assignments")

    # ---------- profiles ----------
    tot_m = out["monetary"].sum()
    prof = out.groupby("cluster").agg(
        customers=("customer_id", "size"), revenue=("monetary", "sum"),
        recency_median=("recency_days", "median"), frequency_median=("frequency", "median"),
        monetary_median=("monetary", "median"), avg_invoice_median=("avg_invoice", "median"),
        n_products_median=("n_products", "median"), tenure_median=("tenure_days", "median"),
        peak_season_share_mean=("peak_season_share", "mean"),
        uk_pct=("country", lambda s: 100 * (s == "United Kingdom").mean())).reset_index()
    prof["customers_pct"] = 100 * prof["customers"] / len(out)
    prof["revenue_pct"] = 100 * prof["revenue"] / tot_m
    z = out[FEATURES].sub(out[FEATURES].mean()).div(out[FEATURES].std())
    z["cluster"] = out["cluster"]
    zprof = z.groupby("cluster")[FEATURES].mean()
    prof["highest_feature"] = prof["cluster"].map(zprof.idxmax(axis=1))
    prof["lowest_feature"] = prof["cluster"].map(zprof.idxmin(axis=1))
    save_table(prof, "kmeans_cluster_profile")
    save_table(zprof.reset_index(), "kmeans_cluster_zprofile")
    xt = pd.crosstab(out["cluster"], out["rfm_segment"])
    save_table(xt.reset_index(), "kmeans_cluster_vs_rfm")

    # write to Hive for later steps / dashboard
    spark.sql(f"DROP TABLE IF EXISTS {DB}.customer_clusters")
    spark.sql(f"CREATE TABLE {DB}.customer_clusters (customer_id INT, cluster INT) STORED AS PARQUET")
    mapping = {int(c): i for i, c in enumerate(order)}
    from pyspark.sql import functions as F
    expr = F.create_map(*[x for kv in mapping.items() for x in (F.lit(kv[0]), F.lit(kv[1]))])
    (final.transform(data).select("customer_id", expr[F.col("cluster")].cast("int").alias("cluster"))
     .write.insertInto(f"{DB}.customer_clusters", overwrite=True))

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(scan["k"], scan["wssse"], color=SERIES[0], marker="o", ms=5)
    axes[0].set_title("Elbow: within-cluster SSE", fontsize=10)
    axes[1].plot(scan["k"], scan["silhouette"], color=SERIES[0], marker="o", ms=5)
    axes[1].set_title("Silhouette (higher is better)", fontsize=10)
    st = scan.dropna(subset=["mean_pairwise_ari"])
    axes[2].plot(st["k"], st["mean_pairwise_ari"], color=SERIES[0], marker="o", ms=5)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Stability: mean ARI across 5 seeds", fontsize=10)
    for ax in axes:
        ax.axvline(k_best, color=TEXT_2, lw=1, ls="--")
        ax.set_xticks(list(K_SCAN))
        ax.set_xlabel("k")
    fig.suptitle(f"Choosing k for customer clustering (chosen k = {k_best})", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    save(fig, "kmeans_k_selection")

    div = LinearSegmentedColormap.from_list("div", [BLUE_DARK, "#6da7ec", "#f0efec", "#ee8f8e", "#b83232"])
    lim = float(np.ceil(np.abs(zprof.values).max() * 2) / 2)
    fig, ax = plt.subplots(figsize=(10, 0.6 * k_best + 2))
    im = ax.imshow(zprof.values, cmap=div, vmin=-lim, vmax=lim, aspect="auto")
    labels = ["recency\n(log days)", "frequency\n(log)", "avg invoice\n(log £)",
              "products\n(log)", "tenure\n(days)", "peak-season\nshare"]
    ax.set_xticks(range(len(FEATURES)), labels, fontsize=8)
    ax.set_yticks(range(k_best), [f"C{r.cluster}: {r.customers_pct:.0f}% cust, {r.revenue_pct:.0f}% rev"
                                  for r in prof.itertuples()], fontsize=9)
    for i in range(zprof.shape[0]):
        for j in range(zprof.shape[1]):
            v = zprof.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > lim * 0.6 else "#0b0b0b")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Mean z-score")
    ax.set_title("Customer cluster profiles (mean standardised feature)")
    save(fig, "kmeans_cluster_profile")

    xtp = xt.div(xt.sum(axis=1), axis=0) * 100
    seg_order = [s for s in ["Champions", "Loyal customers", "Potential loyalists", "New customers",
                             "Promising", "Need attention", "About to sleep", "At risk",
                             "Can't lose", "Hibernating"] if s in xtp.columns]
    xtp = xtp[seg_order]
    fig, ax = plt.subplots(figsize=(11, 0.55 * k_best + 2))
    im = ax.imshow(xtp.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(seg_order)), [s.replace(" ", "\n") for s in seg_order], fontsize=8)
    ax.set_yticks(range(k_best), [f"C{c}" for c in xtp.index])
    for i in range(xtp.shape[0]):
        for j in range(xtp.shape[1]):
            v = xtp.values[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if v > 55 else "#0b0b0b")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% of cluster")
    ax.set_title("How K-Means clusters map onto RFM segments (row %)")
    save(fig, "kmeans_vs_rfm")

    summary = {"customers_clustered": n_cust, "customers_excluded_monetary_le_0": int(n_excluded),
               "features": FEATURES, "k_chosen": k_best, "final_seed": int(best_seed),
               "silhouette_chosen_seed42": float(scan.loc[scan["chosen"], "silhouette"].iloc[0]),
               "mean_pairwise_ari_chosen": float(scan.loc[scan["chosen"], "mean_pairwise_ari"].iloc[0]),
               "selection_rule": f"silhouette within {SIL_TOL} of best (k>=3), then max mean ARI"}
    (TABLES_DIR / "kmeans_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(json.dumps(summary, indent=2))
    print(scan.round(4).to_string(index=False))
    print(prof.round(2).T.to_string())
    print(zprof.round(2).to_string())
    print(xtp.round(0).to_string())
    spark.stop()


if __name__ == "__main__":
    main()
