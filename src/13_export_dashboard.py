"""Export everything the HTML dashboard needs into outputs/dashboard/dashboard_data.json.

Sources: the Hive star schema and model tables (retail.*), plus result files already written by
steps 02-12 in outputs/tables/. Nothing is typed by hand.

Recommendations shown in the dashboard are produced here by REFITTING the two personalised
models of step 10 on ALL effective sales with a customer id (step 10 only evaluated them on a
temporal split):
  - ALS (implicit) with the best hyper-parameters found in step 10 (rec_als_tuning.csv)
  - item-based CF (cosine) - also used for "similar products" and for arbitrary baskets
Lists contain only items the customer has NOT bought yet (discovery use case).
Also written: Hive table retail.customer_recommendations and outputs/tables/customer_recommendations.csv.
"""
import importlib
import json

import numpy as np
import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F

from common import HIVE_DB, ROOT, TABLES_DIR, get_spark, save_table, to_pandas

rec = importlib.import_module("10_recommender")   # reuse ItemCF / ALS helpers of step 10

DB = HIVE_DB
OUT_DIR = ROOT / "outputs" / "dashboard"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOP_N = 10
N_NEIGHBOURS = 15
END = "2011-12-10"


def csv(name):
    return pd.read_csv(TABLES_DIR / f"{name}.csv")


def records(df, round_to=4):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(round_to)
    return json.loads(df.to_json(orient="records", date_format="iso"))


def main():
    spark = get_spark("13_export_dashboard")
    q = lambda s: to_pandas(spark.sql(s))  # noqa: E731
    EFF = "txn_type = 'sale' AND is_product AND NOT is_reversed"

    # ---------------- overview ----------------
    month_country = q(f"""SELECT year_month, country, CAST(SUM(revenue) AS DOUBLE) AS revenue,
                                 COUNT(DISTINCT invoice) AS invoices
                          FROM {DB}.fact_sales WHERE {EFF} GROUP BY year_month, country""")
    country_tot = q(f"""SELECT country, CAST(SUM(revenue) AS DOUBLE) AS revenue,
                               COUNT(DISTINCT invoice) AS invoices, COUNT(DISTINCT customer_id) AS customers,
                               SUM(quantity) AS units
                        FROM {DB}.fact_sales WHERE {EFF} GROUP BY country ORDER BY revenue DESC""")
    kpis = json.loads((TABLES_DIR / "eda_kpis.json").read_text())

    # ---------------- customers ----------------
    cust = q(f"""SELECT r.customer_id, r.country, r.segment, c.cluster, r.recency_days, r.frequency,
                        ROUND(r.monetary, 2) AS monetary, r.r_score, r.f_score, r.m_score,
                        ROUND(k.churn_probability, 4) AS churn_probability, k.risk_band
                 FROM {DB}.customer_rfm r
                 LEFT JOIN {DB}.customer_clusters c ON r.customer_id = c.customer_id
                 LEFT JOIN {DB}.customer_churn_risk k ON r.customer_id = k.customer_id""")
    cust["cluster"] = cust["cluster"].astype("Int64")

    # ---------------- refit recommenders on all data ----------------
    inter_s = rec.interactions(spark, "2009-12-01", END).cache()
    inter = to_pandas(inter_s)
    hist = rec.truth_sets(inter)
    users = sorted(hist)
    tuning = csv("rec_als_tuning").sort_values("map@10", ascending=False).iloc[0]
    best = {"rank": int(tuning["rank"]), "regParam": float(tuning["regParam"]),
            "alpha": float(tuning["alpha"])}
    indexer = StringIndexer(inputCol="stock_code", outputCol="item_idx").fit(inter_s)
    idx_to_item = dict(enumerate(indexer.labels))
    als = rec.fit_als(rec.als_input(spark, inter_s, indexer).cache(), best)
    als_recs = rec.als_recs(als, spark, users, idx_to_item, seen=hist, n=TOP_N)
    cf = rec.ItemCF(inter)
    cf_recs = cf.recommend({u: hist[u] for u in users}, exclude_seen=True, n=TOP_N)

    rows = []
    for u in users:
        for model, lst in (("als", als_recs.get(u, [])), ("item_cf", cf_recs.get(u, []))):
            for rank, it in enumerate(lst, 1):
                rows.append({"customer_id": int(u), "model": model, "rank": rank, "stock_code": it})
    recs_df = pd.DataFrame(rows)
    save_table(recs_df, "customer_recommendations")
    tmp = TABLES_DIR / "customer_recommendations.csv"   # name must not start with "_" or "."
    spark.sql(f"DROP TABLE IF EXISTS {DB}.customer_recommendations")
    spark.sql(f"""CREATE TABLE {DB}.customer_recommendations (customer_id INT, model STRING,
                  `rank` INT, stock_code STRING) STORED AS PARQUET""")
    (spark.read.option("header", True).schema("customer_id INT, model STRING, rank INT, stock_code STRING")
     .csv(str(tmp)).write.insertInto(f"{DB}.customer_recommendations", overwrite=True))
    n_hive_recs = spark.table(f"{DB}.customer_recommendations").count()

    # item neighbours (top-N cosine) for "similar products" and in-browser basket scoring
    sim = cf.sim.tocsr()
    neighbours = {}
    for j, item in enumerate(cf.items):
        row = sim.getrow(j)
        if row.nnz == 0:
            continue
        top = np.argsort(-row.data)[:N_NEIGHBOURS]
        neighbours[item] = [[cf.items[row.indices[t]], round(float(row.data[t]), 4)] for t in top]

    # last purchases per customer (for the recommender tab)
    last = q(f"""SELECT customer_id, stock_code, MAX(invoice_ts) AS last_ts
                 FROM {DB}.fact_sales WHERE {EFF} AND has_customer
                 GROUP BY customer_id, stock_code""")
    last = last.sort_values(["customer_id", "last_ts"], ascending=[True, False])
    recent = last.groupby("customer_id").head(10).groupby("customer_id")["stock_code"].apply(list).to_dict()

    # descriptions for every code that appears anywhere in the payload
    desc = q(f"SELECT stock_code, description FROM {DB}.dim_product")
    desc = dict(zip(desc["stock_code"], desc["description"].fillna("")))

    # every table in the Hive database (star schema + model outputs), counted live
    hive_rows = []
    for t in [r["tableName"] for r in spark.sql(f"SHOW TABLES IN {DB}").collect()]:
        parts = spark.sql(f"SHOW PARTITIONS {DB}.{t}").count() if t == "fact_sales" else 0
        kind = "star schema" if t.startswith(("fact_", "dim_")) else "model output"
        hive_rows.append({"table": f"{DB}.{t}", "rows": spark.table(f"{DB}.{t}").count(),
                          "kind": kind, "partitions": parts})
    hive_tables = pd.DataFrame(hive_rows).sort_values(["kind", "rows"], ascending=[False, False])

    # ---------------- products / seasonality / rules ----------------
    rules = csv("fpgrowth_rules")
    rules = rules[["antecedent", "consequent", "antecedent_desc", "consequent_desc", "support",
                   "support_invoices", "confidence", "lift", "similar_name"]]
    growth = csv("product_growth_decline")
    growth = pd.concat([growth.nsmallest(15, "change"), growth.nlargest(15, "change")])

    payload = {
        "meta": {"source": "UCI Online Retail II, effective product sales (see README)",
                 "period": "2009-12-01 to 2011-12-09", "hive": "Hive local (embedded Derby), no Hadoop cluster",
                 "rec_models_refit_on": "all effective sales with customer id",
                 "als_params": best},
        "kpis": kpis,
        "monthly_country": records(month_country, 2),
        "countries": records(country_tot, 2),
        "weekday_hour": records(csv("eda_weekday_hour")),
        "monthly_yoy": records(csv("eda_yoy_monthly"), 2),
        "cancellations": records(csv("eda_cancellations_monthly"), 3),
        "new_vs_returning": records(csv("eda_new_vs_returning")),
        "top_products": records(csv("eda_product").head(25), 2),
        "rfm_segments": records(csv("rfm_segment_summary"), 2),
        "clusters": records(csv("kmeans_cluster_profile"), 3),
        "churn_metrics": records(csv("churn_metrics"), 4),
        "churn_importance": records(csv("churn_feature_importance"), 4),
        "churn_by_recency": records(csv("churn_rate_by_recency_band"), 4),
        "customers": records(cust, 4),
        "rules": records(rules, 4),
        "growth": records(growth[["stock_code", "description", "rev_2010", "rev_2011", "change", "status"]], 2),
        "growth_summary": records(csv("product_growth_decline_summary"), 2),
        "christmas_share": records(csv("seasonality_christmas_share"), 3),
        "seasonal_index": records(csv("seasonality_monthly_index"), 3),
        "weekly": records(csv("seasonality_weekly"), 2),
        "rec_metrics": records(csv("rec_metrics"), 4),
        "rec_bootstrap": records(csv("rec_bootstrap_map10"), 4),
        "rec_cold_start": records(csv("rec_cold_start_metrics"), 4),
        "recs": {str(u): {"als": als_recs.get(u, []), "item_cf": cf_recs.get(u, []),
                          "recent": recent.get(u, [])} for u in users},
        "neighbours": neighbours,
        "pipeline": {"cleaning_log": records(csv("etl_cleaning_log")),
                     "reconciliation": records(csv("reconciliation")),
                     "hive_tables": records(hive_tables),
                     "quality": records(csv("quality_summary"))},
    }
    used = set(desc) & ({c for v in payload["recs"].values() for lst in v.values() for c in lst}
                        | set(neighbours) | {n for v in neighbours.values() for n, _ in v})
    payload["desc"] = {c: desc[c] for c in sorted(used)}

    out = OUT_DIR / "dashboard_data.json"
    out.write_text(json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8")
    short = sum(1 for u in users if len(als_recs.get(u, [])) < TOP_N)
    summary = {"json_bytes": out.stat().st_size, "customers": len(cust), "customers_with_recs": len(users),
               "customers_with_fewer_than_10_als_recs": short,
               "recommendation_rows": len(recs_df), "recommendations_in_hive": n_hive_recs,
               "products_with_neighbours": len(neighbours), "rules": len(rules), "als_params": best}
    (TABLES_DIR / "dashboard_export_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
