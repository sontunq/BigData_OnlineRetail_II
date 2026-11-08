"""Customer churn prediction with Spark MLlib (temporal, same-season design).

Definition: at snapshot date T, a customer who bought at least once in the LOOKBACK window
[T - 270 days, T) has churned if they make NO effective purchase in [T, T + 90 days).
Features use only data before T (no look-ahead) and the same 270-day lookback at every
snapshot (the data start in Dec 2009, so a 2010 snapshot has only ~9 months of history).

  TRAIN snapshot 2010-09-01 (label window Sep-Nov 2010)
  TEST  snapshot 2011-09-01 (label window Sep-Nov 2011)  -> same season, one year later
  SCORE snapshot 2011-12-10 (end of data, no label)      -> current churn risk, saved to Hive

Models: logistic regression (standardised features), random forest, and a recency-only
baseline (score = days since last purchase). Metrics on the TEST snapshot: ROC AUC, PR AUC,
precision / recall / F1 at threshold 0.5, and precision / recall in the top 20% risk.
K-Means clusters are NOT used as features: they were fitted on the full period (leakage).
No Python UDFs: they crash the Python worker in this environment (pandas 3 + PySpark 3.5);
probabilities are extracted with pyspark.ml.functions.vector_to_array (JVM).
"""
import json

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from common import HIVE_DB, SEED, TABLES_DIR, get_spark, save_table, to_pandas
from plotstyle import NEUTRAL, SERIES, TEXT_2, plt, save

DB = HIVE_DB
LOOKBACK, HORIZON = 270, 90
TRAIN_T, TEST_T, SCORE_T = "2010-09-01", "2011-09-01", "2011-12-10"
FEATURES = ["recency_days", "log_frequency", "log_monetary", "log_avg_invoice", "log_n_products",
            "log_frequency_90d", "log_monetary_90d", "monetary_trend", "active_months",
            "days_since_first", "return_share", "is_uk"]


def snapshot(spark, t, with_label=True):
    """Feature table (and label) for customers active in [t - LOOKBACK, t)."""
    label = f"""
        , CASE WHEN fut.customer_id IS NULL THEN 1.0 ELSE 0.0 END AS churn""" if with_label else ""
    fut_join = f"""
        LEFT JOIN (SELECT DISTINCT customer_id FROM eff
                   WHERE invoice_ts >= TIMESTAMP '{t}'
                     AND invoice_ts < TIMESTAMP '{t}' + INTERVAL {HORIZON} DAYS) fut
          ON f.customer_id = fut.customer_id""" if with_label else ""
    return spark.sql(f"""
    WITH eff AS (
      SELECT customer_id, invoice, invoice_ts, CAST(revenue AS DOUBLE) AS rev, stock_code, country
      FROM {DB}.fact_sales
      WHERE txn_type = 'sale' AND is_product AND NOT is_reversed AND has_customer),
    hist AS (
      SELECT * FROM eff WHERE invoice_ts < TIMESTAMP '{t}'
                         AND invoice_ts >= TIMESTAMP '{t}' - INTERVAL {LOOKBACK} DAYS),
    ret AS (
      SELECT customer_id, -SUM(CAST(revenue AS DOUBLE)) AS returns
      FROM {DB}.fact_sales
      WHERE txn_type = 'cancellation' AND is_product AND NOT is_reversed AND has_customer
        AND invoice_ts < TIMESTAMP '{t}' AND invoice_ts >= TIMESTAMP '{t}' - INTERVAL {LOOKBACK} DAYS
      GROUP BY customer_id),
    firsts AS (SELECT customer_id, MIN(invoice_ts) AS first_ts FROM eff
               WHERE invoice_ts < TIMESTAMP '{t}' GROUP BY customer_id),
    f AS (
      SELECT h.customer_id,
        DATEDIFF(TIMESTAMP '{t}', MAX(h.invoice_ts)) AS recency_days,
        COUNT(DISTINCT h.invoice) AS frequency,
        SUM(h.rev) AS monetary,
        COUNT(DISTINCT h.stock_code) AS n_products,
        COUNT(DISTINCT CASE WHEN h.invoice_ts >= TIMESTAMP '{t}' - INTERVAL 90 DAYS THEN h.invoice END) AS frequency_90d,
        SUM(CASE WHEN h.invoice_ts >= TIMESTAMP '{t}' - INTERVAL 90 DAYS THEN h.rev ELSE 0 END) AS monetary_90d,
        SUM(CASE WHEN h.invoice_ts <  TIMESTAMP '{t}' - INTERVAL 90 DAYS
                  AND h.invoice_ts >= TIMESTAMP '{t}' - INTERVAL 180 DAYS THEN h.rev ELSE 0 END) AS monetary_prev_90d,
        COUNT(DISTINCT DATE_FORMAT(h.invoice_ts, 'yyyy-MM')) AS active_months,
        MAX(CASE WHEN h.country = 'United Kingdom' THEN 1.0 ELSE 0.0 END) AS is_uk
      FROM hist h GROUP BY h.customer_id)
    SELECT f.customer_id, CAST(f.recency_days AS DOUBLE) AS recency_days,
      LN(1 + f.frequency) AS log_frequency, LN(1 + GREATEST(f.monetary, 0)) AS log_monetary,
      LN(1 + GREATEST(f.monetary, 0) / f.frequency) AS log_avg_invoice,
      LN(1 + f.n_products) AS log_n_products, LN(1 + f.frequency_90d) AS log_frequency_90d,
      LN(1 + GREATEST(f.monetary_90d, 0)) AS log_monetary_90d,
      (f.monetary_90d - f.monetary_prev_90d) / (f.monetary_90d + f.monetary_prev_90d + 1) AS monetary_trend,
      CAST(f.active_months AS DOUBLE) AS active_months,
      CAST(DATEDIFF(TIMESTAMP '{t}', fi.first_ts) AS DOUBLE) AS days_since_first,
      COALESCE(r.returns, 0) / (f.monetary + COALESCE(r.returns, 0) + 1) AS return_share,
      f.is_uk, f.frequency, f.monetary {label}
    FROM f JOIN firsts fi ON f.customer_id = fi.customer_id
    LEFT JOIN ret r ON f.customer_id = r.customer_id {fut_join}""")


def top_share_metrics(y, s, share=0.2):
    n = max(1, int(round(len(s) * share)))
    idx = np.argsort(-s)[:n]
    tp = y[idx].sum()
    return {"precision_top20pct": float(tp / n), "recall_top20pct": float(tp / max(y.sum(), 1))}


def threshold_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision@0.5": prec, "recall@0.5": rec, "f1@0.5": f1}


def main():
    spark = get_spark("11_churn")
    tr = snapshot(spark, TRAIN_T).cache()
    te = snapshot(spark, TEST_T).cache()
    sc = snapshot(spark, SCORE_T, with_label=False).cache()
    for d in (tr, te, sc):
        d.count()
    rate = lambda d: float(d.agg(F.avg(F.col("churn").cast("double"))).first()[0])  # noqa: E731
    split = {"train_snapshot": TRAIN_T, "test_snapshot": TEST_T, "score_snapshot": SCORE_T,
             "lookback_days": LOOKBACK, "horizon_days": HORIZON,
             "train_customers": tr.count(), "train_churn_rate": round(rate(tr), 4),
             "test_customers": te.count(), "test_churn_rate": round(rate(te), 4),
             "score_customers": sc.count()}
    print(split, flush=True)

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="raw")
    scaler = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
    lr = LogisticRegression(featuresCol="features", labelCol="churn", maxIter=100,
                            regParam=0.01, elasticNetParam=0.0)
    rf = RandomForestClassifier(featuresCol="raw", labelCol="churn", numTrees=200, maxDepth=6,
                                minInstancesPerNode=20, seed=SEED)
    models = {"logistic_regression": Pipeline(stages=[assembler, scaler, lr]).fit(tr),
              "random_forest": Pipeline(stages=[assembler, rf]).fit(tr)}

    ev = BinaryClassificationEvaluator(labelCol="churn", rawPredictionCol="rawPrediction",
                                       metricName="areaUnderROC")
    rows, curves = [], {}
    y_te = None
    for name, m in models.items():
        pred = m.transform(te)
        spark_auc = ev.evaluate(pred)
        p = to_pandas(pred.select("customer_id", "churn",
                                  vector_to_array("probability")[1].alias("p")))
        y, s = p["churn"].values.astype(int), p["p"].values
        y_te = y
        res = {"model": name, "roc_auc": roc_auc_score(y, s), "roc_auc_spark": spark_auc,
               "pr_auc": average_precision_score(y, s), **threshold_metrics(y, s),
               **top_share_metrics(y, s)}
        rows.append(res)
        curves[name] = (y, s)
    base = to_pandas(te.select("churn", "recency_days"))
    yb, sb = base["churn"].values.astype(int), base["recency_days"].values
    rows.append({"model": "baseline_recency_only", "roc_auc": roc_auc_score(yb, sb),
                 "roc_auc_spark": np.nan, "pr_auc": average_precision_score(yb, sb),
                 "precision@0.5": np.nan, "recall@0.5": np.nan, "f1@0.5": np.nan,
                 **top_share_metrics(yb, sb)})
    curves["baseline_recency_only"] = (yb, sb)
    rows.append({"model": "no_skill (predict churn rate)", "roc_auc": 0.5, "pr_auc": float(yb.mean())})
    metrics = pd.DataFrame(rows)
    save_table(metrics, "churn_metrics")

    # ---------- interpretability ----------
    lr_m = models["logistic_regression"].stages[-1]
    coef = pd.DataFrame({"feature": FEATURES, "lr_coefficient_std": lr_m.coefficients.toArray()})
    rf_m = models["random_forest"].stages[-1]
    coef["rf_importance"] = rf_m.featureImportances.toArray()
    coef = coef.sort_values("rf_importance", ascending=False)
    save_table(coef, "churn_feature_importance")

    # ---------- churn rate by recency band on TEST (sanity / business view) ----------
    band = to_pandas(te.withColumn("recency_band", F.when(F.col("recency_days") <= 30, "0-30")
                                   .when(F.col("recency_days") <= 90, "31-90")
                                   .when(F.col("recency_days") <= 180, "91-180")
                                   .otherwise("181-270"))
                     .groupBy("recency_band").agg(F.count("*").alias("customers"),
                                                  F.avg("churn").alias("churn_rate")))
    save_table(band, "churn_rate_by_recency_band")

    # ---------- score current customers (end of data) ----------
    best_name = metrics.iloc[:2].sort_values("roc_auc", ascending=False)["model"].iloc[0]
    best = models[best_name]
    scored = best.transform(sc).select(
        "customer_id", vector_to_array("probability")[1].alias("churn_probability"),
        "recency_days", "frequency", "monetary")
    # Rank-based bands: the model was trained on a September snapshot and is applied to the
    # end of December (after the peak), so absolute probabilities are shifted by season.
    # Ranks are more robust than fixed probability thresholds: top 20% = high, next 30% = medium.
    from pyspark.sql import Window
    pr = F.percent_rank().over(Window.orderBy(F.desc("churn_probability"), "customer_id"))
    scored = scored.withColumn("pr", pr).withColumn(
        "risk_band", F.when(F.col("pr") < 0.2, "high").when(F.col("pr") < 0.5, "medium")
        .otherwise("low")).drop("pr")
    spark.sql(f"DROP TABLE IF EXISTS {DB}.customer_churn_risk")
    spark.sql(f"""CREATE TABLE {DB}.customer_churn_risk (customer_id INT, churn_probability DOUBLE,
                  recency_days DOUBLE, frequency BIGINT, monetary DOUBLE, risk_band STRING)
                  STORED AS PARQUET""")
    scored.write.insertInto(f"{DB}.customer_churn_risk", overwrite=True)
    risk = to_pandas(spark.sql(f"""SELECT risk_band, COUNT(*) customers, SUM(monetary) monetary_270d
                                   FROM {DB}.customer_churn_risk GROUP BY risk_band"""))
    risk["monetary_270d"] = risk["monetary_270d"].astype(float)
    save_table(risk, "churn_current_risk_bands")

    # ---------- figures ----------
    labels = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
              "baseline_recency_only": "Baseline: recency only"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for i, (name, (y, s)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        auc = metrics.loc[metrics["model"] == name, "roc_auc"].iloc[0]
        ap = metrics.loc[metrics["model"] == name, "pr_auc"].iloc[0]
        axes[0].plot(fpr, tpr, color=SERIES[i], label=f"{labels[name]} (AUC {auc:.3f})")
        axes[1].plot(rc, pr, color=SERIES[i], label=f"{labels[name]} (AP {ap:.3f})")
    axes[0].plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--")
    axes[1].axhline(y_te.mean(), color=NEUTRAL, lw=1, ls="--")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC (test snapshot 2011-09-01)", fontsize=10)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-recall (churn rate {y_te.mean():.0%})", fontsize=10)
    for ax in axes[:2]:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", fontsize=8)
    d = coef.sort_values("rf_importance")
    axes[2].barh(d["feature"], d["rf_importance"], color=SERIES[0], height=0.6)
    axes[2].set_title("Random forest feature importance", fontsize=10)
    axes[2].tick_params(axis="y", labelsize=8)
    fig.suptitle(f"Churn = no purchase in the {HORIZON} days after the snapshot "
                 f"(train {TRAIN_T}, test {TEST_T})", x=0.01, ha="left", fontweight="bold", fontsize=12)
    save(fig, "churn_evaluation")

    summary = {**split, "best_model_for_scoring": best_name}
    (TABLES_DIR / "churn_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(json.dumps(summary, indent=2))
    print(metrics.round(4).to_string(index=False))
    print(coef.round(4).to_string(index=False))
    print(band.round(3).to_string(index=False))
    print(risk.round(0).to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
