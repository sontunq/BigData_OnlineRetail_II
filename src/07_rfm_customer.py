"""Customer RFM segmentation (reads the Hive star schema, writes retail.customer_rfm).

Population: customers with at least one EFFECTIVE product sale (sale AND is_product AND NOT
is_reversed). Lines without customer_id cannot be attributed and are excluded.
Reference date: the day after the last invoice in the data (2011-12-10).
  Recency   = days from the customer's last effective purchase to the reference date
  Frequency = distinct effective invoices
  Monetary  = net revenue = effective sales + remaining product cancellations (returns that
              were not matched as full reversals), so it can be <= 0 for a few customers
Scores 1..5: 1 + floor(5 * percent_rank), capped at 5. percent_rank gives tied values the
same score (NTILE would split the 28% of one-invoice customers arbitrarily). Recency is
ranked descending (recent = 5). Because of ties, score groups are not equal-sized.
Segments: the commonly used R x F grid (10 named segments); M is reported per segment.
"""
import json

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from common import HIVE_DB, TABLES_DIR, get_spark, save_table, to_pandas
from plotstyle import SERIES, TEXT_2, plt, save

DB = HIVE_DB
SEGMENT_ORDER = ["Champions", "Loyal customers", "Potential loyalists", "New customers",
                 "Promising", "Need attention", "About to sleep", "At risk", "Can't lose",
                 "Hibernating"]

RFM_SQL = f"""
WITH ref AS (SELECT DATE_ADD(MAX(TO_DATE(invoice_ts)), 1) AS ref_date FROM {DB}.fact_sales),
eff AS (
  SELECT customer_id, invoice, invoice_ts, CAST(revenue AS DOUBLE) AS rev
  FROM {DB}.fact_sales
  WHERE txn_type = 'sale' AND is_product AND NOT is_reversed AND has_customer
),
returns AS (
  SELECT customer_id, CAST(SUM(revenue) AS DOUBLE) AS returns_value
  FROM {DB}.fact_sales
  WHERE txn_type = 'cancellation' AND is_product AND NOT is_reversed AND has_customer
  GROUP BY customer_id
),
base AS (
  SELECT e.customer_id,
         DATEDIFF(r.ref_date, TO_DATE(MAX(e.invoice_ts))) AS recency_days,
         COUNT(DISTINCT e.invoice) AS frequency,
         SUM(e.rev) AS gross_effective_sales,
         MIN(e.invoice_ts) AS first_purchase_ts,
         MAX(e.invoice_ts) AS last_purchase_ts,
         r.ref_date
  FROM eff e CROSS JOIN ref r
  GROUP BY e.customer_id, r.ref_date
),
m AS (
  SELECT b.*, COALESCE(x.returns_value, 0) AS returns_value,
         b.gross_effective_sales + COALESCE(x.returns_value, 0) AS monetary
  FROM base b LEFT JOIN returns x ON b.customer_id = x.customer_id
),
scored AS (
  SELECT m.*,
    LEAST(5, 1 + FLOOR(5 * PERCENT_RANK() OVER (ORDER BY recency_days DESC))) AS r_score,
    LEAST(5, 1 + FLOOR(5 * PERCENT_RANK() OVER (ORDER BY frequency)))        AS f_score,
    LEAST(5, 1 + FLOOR(5 * PERCENT_RANK() OVER (ORDER BY monetary)))         AS m_score
  FROM m
)
SELECT s.*, c.country,
  CASE
    WHEN r_score = 5 AND f_score >= 4                    THEN 'Champions'
    WHEN r_score BETWEEN 3 AND 4 AND f_score >= 4        THEN 'Loyal customers'
    WHEN r_score >= 4 AND f_score BETWEEN 2 AND 3        THEN 'Potential loyalists'
    WHEN r_score = 5 AND f_score = 1                     THEN 'New customers'
    WHEN r_score = 4 AND f_score = 1                     THEN 'Promising'
    WHEN r_score = 3 AND f_score = 3                     THEN 'Need attention'
    WHEN r_score = 3 AND f_score <= 2                    THEN 'About to sleep'
    WHEN r_score <= 2 AND f_score BETWEEN 3 AND 4        THEN 'At risk'
    WHEN r_score <= 2 AND f_score = 5                    THEN "Can't lose"
    WHEN r_score <= 2 AND f_score <= 2                   THEN 'Hibernating'
  END AS segment
FROM scored s JOIN {DB}.dim_customer c ON s.customer_id = c.customer_id
"""

DDL = f"""
CREATE TABLE {DB}.customer_rfm (
  customer_id INT, recency_days INT, frequency BIGINT, gross_effective_sales DOUBLE,
  returns_value DOUBLE, monetary DOUBLE, first_purchase_ts TIMESTAMP, last_purchase_ts TIMESTAMP,
  ref_date DATE, r_score INT, f_score INT, m_score INT, country STRING, segment STRING)
STORED AS PARQUET"""
COLS = ["customer_id", "recency_days", "frequency", "gross_effective_sales", "returns_value",
        "monetary", "first_purchase_ts", "last_purchase_ts", "ref_date", "r_score", "f_score",
        "m_score", "country", "segment"]


def main():
    spark = get_spark("07_rfm_customer")
    rfm_s = spark.sql(RFM_SQL)
    for c in ["r_score", "f_score", "m_score"]:
        rfm_s = rfm_s.withColumn(c, rfm_s[c].cast("int"))
    spark.sql(f"DROP TABLE IF EXISTS {DB}.customer_rfm")
    spark.sql(DDL)
    rfm_s.select(*COLS).write.insertInto(f"{DB}.customer_rfm", overwrite=True)
    rfm = to_pandas(spark.table(f"{DB}.customer_rfm"))
    spark.stop()

    if rfm["segment"].isna().any():
        raise RuntimeError(f"{rfm['segment'].isna().sum()} customers without a segment")
    save_table(rfm, "rfm_customers")

    # ---------- score boundaries actually produced (ties make groups unequal) ----------
    bounds = []
    for metric, score in [("recency_days", "r_score"), ("frequency", "f_score"), ("monetary", "m_score")]:
        g = rfm.groupby(score)[metric].agg(["min", "max", "size"]).reset_index()
        g.insert(0, "metric", metric)
        g.columns = ["metric", "score", "min", "max", "customers"]
        bounds.append(g)
    bounds = pd.concat(bounds)
    save_table(bounds, "rfm_score_bounds")

    # ---------- segment profile ----------
    tot_c, tot_m = len(rfm), rfm["monetary"].sum()
    seg = (rfm.groupby("segment").agg(
        customers=("customer_id", "size"), revenue=("monetary", "sum"),
        recency_median=("recency_days", "median"), frequency_median=("frequency", "median"),
        monetary_median=("monetary", "median"), monetary_mean=("monetary", "mean"),
        uk_pct=("country", lambda s: 100 * (s == "United Kingdom").mean()))
        .reindex(SEGMENT_ORDER).reset_index())
    seg["customers_pct"] = 100 * seg["customers"] / tot_c
    seg["revenue_pct"] = 100 * seg["revenue"] / tot_m
    save_table(seg, "rfm_segment_summary")

    grid = rfm.pivot_table(index="r_score", columns="f_score", values="customer_id",
                           aggfunc="count", fill_value=0)
    save_table(grid.reset_index(), "rfm_rf_grid_counts")

    summary = {
        "customers": tot_c,
        "reference_date": str(rfm["ref_date"].iloc[0]),
        "net_monetary_total": round(float(tot_m), 2),
        "customers_monetary_le_0": int((rfm["monetary"] <= 0).sum()),
        "recency_median_days": float(rfm["recency_days"].median()),
        "frequency_median": float(rfm["frequency"].median()),
        "monetary_median": round(float(rfm["monetary"].median()), 2),
        "corr_spearman_f_m": round(float(rfm["frequency"].corr(rfm["monetary"], method="spearman")), 3),
        "corr_spearman_r_f": round(float(rfm["recency_days"].corr(rfm["frequency"], method="spearman")), 3),
    }
    (TABLES_DIR / "rfm_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- figure 1: customers vs revenue share by segment ----------
    d = seg.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, col, title in [(axes[0], "customers_pct", "Share of customers"),
                           (axes[1], "revenue_pct", "Share of net revenue")]:
        ax.barh(d["segment"], d[col], color=SERIES[0], height=0.6)
        for y, (v, n) in enumerate(zip(d[col], d["customers"])):
            lab = f"  {v:.1f}%" + (f"  (n={n:,})" if col == "customers_pct" else "")
            ax.text(max(v, 0), y, lab, va="center", fontsize=8, color=TEXT_2)
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.set_xlim(min(0, d[col].min() * 1.2), d[col].max() * 1.35)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Customer RFM segments ({tot_c:,} customers, reference date "
                 f"{summary['reference_date']})", x=0.01, ha="left", fontweight="bold", fontsize=12)
    save(fig, "rfm_segments")

    # ---------- figure 2: R x F grid ----------
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(grid.values[::-1], cmap="Blues", aspect="auto")
    ax.set_xticks(range(grid.shape[1]), [f"F{c}" for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"R{r}" for r in grid.index[::-1]])
    seg_of = rfm.groupby(["r_score", "f_score"])["segment"].first()
    vmax = grid.values.max()
    for i, r in enumerate(grid.index[::-1]):
        for j, f in enumerate(grid.columns):
            n = grid.loc[r, f]
            if n:
                ax.text(j, i, f"{seg_of.get((r, f), '')}\n{n:,}", ha="center", va="center",
                        fontsize=7, color="white" if n > vmax * 0.55 else "#0b0b0b")
    ax.grid(False)
    ax.set_xlabel("Frequency score (5 = most invoices)")
    ax.set_ylabel("Recency score (5 = most recent)")
    fig.colorbar(im, ax=ax, label="Customers")
    ax.set_title("Customers by recency and frequency score")
    save(fig, "rfm_rf_grid")

    # ---------- figure 3: R, F, M distributions ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(rfm["recency_days"], bins=40, color=SERIES[0], edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Days since last purchase")
    axes[0].set_title("Recency", fontsize=10)
    fb = np.logspace(0, np.log10(rfm["frequency"].max()), 30)
    axes[1].hist(rfm["frequency"], bins=fb, color=SERIES[0], edgecolor="white", linewidth=0.4)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Invoices (log scale)")
    axes[1].set_title("Frequency", fontsize=10)
    pos = rfm.loc[rfm["monetary"] > 0, "monetary"]
    mb = np.logspace(0, np.log10(pos.max()), 40)
    axes[2].hist(pos, bins=mb, color=SERIES[0], edgecolor="white", linewidth=0.4)
    axes[2].set_xscale("log")
    axes[2].set_xlabel(f"Net revenue £ (log; {summary['customers_monetary_le_0']} customers <= 0 not shown)")
    axes[2].set_title("Monetary", fontsize=10)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:,.0f}"))
    axes[2].set_xticks([10, 100, 1e3, 1e4, 1e5])
    axes[2].xaxis.set_major_formatter(FuncFormatter(
        lambda x, _p: f"£{x / 1e3:,.0f}k" if x >= 1e3 else f"£{x:,.0f}"))
    axes[0].set_ylabel("Customers")
    save(fig, "rfm_distributions")

    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=2))
    print(bounds.to_string(index=False))
    print(seg.round(1).to_string(index=False))
    print(grid.to_string())


if __name__ == "__main__":
    main()
