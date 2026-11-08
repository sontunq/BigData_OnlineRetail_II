"""Generate the data dictionary (outputs/data_dictionary.md + data_dictionary.csv).

Row counts, columns and Hive column types are READ from the files and from Hive; only the
descriptions below are written by hand. The script fails if an output file has no
description, or a description points to a file that does not exist, so the dictionary
cannot silently drift from the outputs.
"""
import json
import sys

import pandas as pd

from common import FIGURES_DIR, HIVE_DB, ROOT, TABLES_DIR, get_spark, to_pandas

OUT_MD = ROOT / "outputs" / "data_dictionary.md"
OUT_CSV = ROOT / "outputs" / "data_dictionary.csv"
DASH_DIR = ROOT / "outputs" / "dashboard"

# "Effective product sales" = txn_type 'sale' AND is_product AND NOT is_reversed (see 03_etl_hive.py)
TABLES = {
    # 00-01
    "environment.json": ("00", "Tool versions and the Hive smoke test result."),
    "ingest_manifest.json": ("01", "Source workbook, MD5, landing CSV, row counts per sheet, explicit schema."),
    # 02 quality (raw data, before cleaning)
    "quality_summary.csv": ("02", "One row per data-quality check on the raw data: value, PASS/WARN/FAIL, note."),
    "quality_nulls.csv": ("02", "Null or blank count and share per raw column."),
    "quality_sheet_overlap.csv": ("02", "The 2010-12-01..09 window as it appears in each Excel sheet (rows, invoices, time range)."),
    "quality_invoice_types.csv": ("02", "Rows, invoices and units by invoice type (numeric, cancellation C..., adjustment A...)."),
    "quality_non_product_codes.csv": ("02", "Service / fee / voucher stock codes flagged as non-products, with rows and amount."),
    "quality_monthly_coverage.csv": ("02", "Rows and invoices per month and sheet (date coverage)."),
    # 03-05 ETL, Hive, reconciliation
    "etl_cleaning_log.csv": ("03", "Rows before / removed / after each cleaning step (overlap, exact duplicates)."),
    "etl_fact_flag_breakdown.csv": ("03", "fact_sales rows, invoices, units and revenue by txn_type, is_product, has_customer, is_reversed."),
    "hive_tables.csv": ("03", "Star-schema tables written by the ETL: rows, table type, location, SerDe, partitions."),
    "hive_referential_integrity.csv": ("03", "Orphan foreign keys in fact_sales per dimension (all must be 0)."),
    "hive_queries.sql": ("04", "HiveQL text of the validation queries."),
    "hive_q_revenue_by_month.csv": ("04", "HiveQL: gross product sales, invoices, customers, units by month."),
    "hive_q_revenue_by_country_top15.csv": ("04", "HiveQL: top 15 countries by gross product sales with revenue share."),
    "hive_q_top_products_by_revenue.csv": ("04", "HiveQL: top 15 products by gross product sales."),
    "hive_q_invoices_by_weekday_hour.csv": ("04", "HiveQL: invoices and revenue by weekday and hour."),
    "hive_q_cancellation_rate_by_month.csv": ("04", "HiveQL: cancelled value as % of product sales by month."),
    "hive_q_customers_by_year_and_repeat.csv": ("04", "HiveQL: customers by first-purchase year and how many bought again."),
    "hive_q_partitions_fact.csv": ("04", "HiveQL: SHOW PARTITIONS of fact_sales (one per month)."),
    "reconciliation.csv": ("05", "Independent pandas re-implementation vs Spark/Hive for each key metric; match flag."),
    "reconciliation_monthly.csv": ("05", "Monthly invoices and revenue, pandas vs Spark/Hive; match flag."),
    # 06 EDA (effective product sales)
    "eda_kpis.json": ("06", "Headline KPIs: revenue, invoices, customers, concentration, invoice size, YoY growth."),
    "eda_monthly.csv": ("06", "Effective sales, invoices, customers and units by year and month."),
    "eda_yoy_monthly.csv": ("06", "Monthly revenue 2009/2010/2011 side by side and 2011-vs-2010 growth %."),
    "eda_weekday.csv": ("06", "Revenue, invoices and trading days by weekday; revenue per trading day."),
    "eda_hour.csv": ("06", "Invoices and revenue by hour of day."),
    "eda_weekday_hour.csv": ("06", "Invoices by weekday x hour."),
    "eda_country.csv": ("06", "Revenue, customers, invoices, revenue share and revenue per customer by country."),
    "eda_product.csv": ("06", "Revenue, units, invoices, customers by product, with cumulative revenue share (Pareto)."),
    "eda_customer_frequency.csv": ("06", "Number of customers by invoices per customer (capped at 20)."),
    "eda_customer_pareto.csv": ("06", "Customer concentration curve: % of customers vs cumulative % of revenue."),
    "eda_invoice_summary.csv": ("06", "Lines and value per invoice, registered vs no customer id."),
    "eda_cancellations_monthly.csv": ("06", "Gross product sales, fully reversed sales and other cancellations by month."),
    "eda_new_vs_returning.csv": ("06", "Active customers per month split into new (first purchase) and returning."),
    # 07 RFM
    "rfm_customers.csv": ("07", "One row per customer: recency, frequency, monetary (net), R/F/M scores, country, segment."),
    "rfm_segment_summary.csv": ("07", "Customers, revenue and medians per RFM segment; shares of customers and revenue."),
    "rfm_score_bounds.csv": ("07", "Actual min/max of recency, frequency, monetary inside each score (ties make groups unequal)."),
    "rfm_rf_grid_counts.csv": ("07", "Customers per recency score x frequency score cell."),
    "rfm_summary.json": ("07", "Reference date, medians and rank correlations of R, F, M."),
    # 08 K-Means
    "kmeans_k_selection.csv": ("08", "WSSSE, silhouette and seed stability (mean/min pairwise ARI) per k; chosen k."),
    "kmeans_stability_runs.csv": ("08", "Each (k, seed) fit: WSSSE, silhouette, cluster sizes."),
    "kmeans_cluster_profile.csv": ("08", "Per cluster: size, revenue, medians of the raw features, highest/lowest feature."),
    "kmeans_cluster_zprofile.csv": ("08", "Per cluster: mean standardised value of each feature."),
    "kmeans_cluster_vs_rfm.csv": ("08", "Customers per cluster x RFM segment."),
    "kmeans_customer_assignments.csv": ("08", "One row per clustered customer: features and assigned cluster."),
    "kmeans_summary.json": ("08", "Features, chosen k, final seed, silhouette, stability and selection rule."),
    # 09 FP-Growth
    "fpgrowth_support_sensitivity.csv": ("09", "Frequent itemsets and rules produced at each minSupport tried, with runtime."),
    "fpgrowth_frequent_itemsets.csv": ("09", "Frequent itemsets at minSupport 1%: items, descriptions, frequency, support."),
    "fpgrowth_rules.csv": ("09", "Association rules: antecedent, consequent, descriptions, support, confidence, lift, similar_name flag."),
    "fpgrowth_summary.json": ("09", "Basket statistics, thresholds and rule counts."),
    # 10 recommender
    "rec_als_tuning.csv": ("10", "ALS grid on the validation split inside the training period, sorted by MAP@10."),
    "rec_metrics.csv": ("10", "Hit rate, precision, recall, MAP @5/10/20, catalogue coverage by model and mode (all / new items)."),
    "rec_bootstrap_map10.csv": ("10", "Paired bootstrap of MAP@10 differences between models (mean, 95% CI)."),
    "rec_cold_start_metrics.csv": ("10", "Metrics for customers first seen in the test period (first invoice as history)."),
    "rec_popularity_bias.csv": ("10", "Median popularity rank of recommended items and share in the top-100 popular items."),
    "rec_examples.csv": ("10", "Example top-5 recommendations for three evaluation customers."),
    "rec_summary.json": ("10", "Train/test split sizes, matrix density, best ALS parameters, runtime."),
    # 11 churn
    "churn_metrics.csv": ("11", "ROC AUC, PR AUC, precision/recall/F1 @0.5 and top-20% precision/recall per model."),
    "churn_feature_importance.csv": ("11", "Logistic-regression coefficients (standardised) and random-forest importance."),
    "churn_rate_by_recency_band.csv": ("11", "Actual churn rate by recency band at the test snapshot."),
    "churn_current_risk_bands.csv": ("11", "Customers and 270-day spend per current risk band (rank-based)."),
    "churn_summary.json": ("11", "Snapshots, windows, churn rates and the model used for scoring."),
    # 12 seasonality
    "seasonality_monthly_index.csv": ("12", "Jan-Nov revenue 2010 and 2011, seasonal index and YoY %."),
    "seasonality_weekly.csv": ("12", "Revenue, invoices, customers by ISO year and week."),
    "seasonality_christmas_share.csv": ("12", "Monthly revenue and share from Christmas-named products (keyword rule)."),
    "seasonality_summary.json": ("12", "Profile correlation, peak weeks, Sep-Nov share, Christmas share, product status counts."),
    "product_growth_decline.csv": ("12", "Products with >= GBP 2,000 in Jan-Nov 2010 or 2011: revenue both years, change, status."),
    "product_growth_decline_summary.csv": ("12", "Products and revenue by status (stable, growing, declining, new, gone)."),
    # 13 export
    "customer_recommendations.csv": ("13", "Top-10 ALS and item-CF recommendations per customer (models refitted on all data)."),
    "dashboard_export_summary.json": ("13", "Size of the dashboard payload and recommendation counts."),
}
DASHBOARD = {
    "dashboard_data.json": ("13", "All data embedded in the dashboard (aggregates, customers, rules, recommendations)."),
    "retail_dashboard.html": ("14", "Self-contained HTML dashboard (plotly.js inlined); open in any browser."),
}
FIGURES = {
    "eda_monthly_yoy.png": ("06", "Monthly effective sales, 2010 vs 2011."),
    "eda_weekday_hour.png": ("06", "Revenue per trading day by weekday; invoices by weekday x hour."),
    "eda_countries.png": ("06", "Top 10 non-UK countries."),
    "eda_top_products.png": ("06", "Top 15 products."),
    "eda_customers.png": ("06", "Revenue concentration (customers, products) and purchase frequency."),
    "eda_invoices.png": ("06", "Lines per invoice and invoice value, registered vs no customer id."),
    "eda_cancellations.png": ("06", "Cancelled value as % of gross product sales by month."),
    "eda_new_vs_returning.png": ("06", "Active customers per month, new vs returning."),
    "rfm_segments.png": ("07", "Share of customers and of revenue per RFM segment."),
    "rfm_rf_grid.png": ("07", "Customers by recency x frequency score with segment labels."),
    "rfm_distributions.png": ("07", "Distributions of recency, frequency and monetary."),
    "kmeans_k_selection.png": ("08", "Elbow, silhouette and seed stability by k."),
    "kmeans_cluster_profile.png": ("08", "Cluster profiles (mean z-scores)."),
    "kmeans_vs_rfm.png": ("08", "K-Means clusters vs RFM segments (row %)."),
    "fpgrowth_top_rules_all.png": ("09", "Top 15 product pairs by lift."),
    "fpgrowth_top_rules_cross.png": ("09", "Top 15 product pairs by lift without a similar name."),
    "fpgrowth_rules_scatter.png": ("09", "All rules: support vs confidence, coloured by lift."),
    "rec_metrics.png": ("10", "Recommender metrics @10 by model, all items vs new items."),
    "churn_evaluation.png": ("11", "Churn ROC, precision-recall and feature importance."),
    "seasonality_weekly_index.png": ("12", "Weekly sales 2010 vs 2011 and monthly seasonal index."),
    "seasonality_christmas_share.png": ("12", "Christmas-named products, share of revenue by month."),
    "product_growth_decline.png": ("12", "Biggest product revenue changes 2011 vs 2010."),
    "dashboard_overview.png": ("15", "Dashboard screenshot: Overview tab."),
    "dashboard_customers.png": ("15", "Dashboard screenshot: Customers tab."),
    "dashboard_products.png": ("15", "Dashboard screenshot: Products & baskets tab."),
    "dashboard_recommender.png": ("15", "Dashboard screenshot: Recommender tab."),
    "dashboard_pipeline.png": ("15", "Dashboard screenshot: Data pipeline tab."),
}
HIVE = {
    "fact_sales": "One row per cleaned invoice line (partitioned by year_month). Flags: txn_type, is_product, has_customer, is_reversed.",
    "dim_product": "One row per stock code: most frequent description, is_product, price range and first/last sale date.",
    "dim_customer": "One row per customer: most frequent country, first and last purchase timestamps.",
    "dim_date": "One row per calendar date: year, quarter, month, weekday, week, has_transactions.",
    "dim_country": "One row per country with is_uk flag.",
    "customer_rfm": "RFM values, scores and segment per customer (step 07).",
    "customer_clusters": "K-Means cluster per customer (step 08).",
    "assoc_rules": "FP-Growth association rules with descriptions (step 09).",
    "customer_churn_risk": "Churn probability and rank-based risk band per active customer at 2011-12-10 (step 11).",
    "customer_recommendations": "Top-10 ALS and item-CF recommendations per customer (step 13).",
}
COLUMN_NOTES = {
    "fact_sales": {
        "line_id": "Surrogate key (row number ordered by time, invoice, stock code).",
        "invoice": "Invoice number; prefix C = cancellation, A = bad-debt adjustment.",
        "invoice_ts": "Invoice timestamp (UK wall-clock time, stored as UTC).",
        "date_key": "FK to dim_date (yyyyMMdd).", "hour": "Hour of invoice_ts.",
        "customer_id": "FK to dim_customer; NULL when the customer is unknown.",
        "stock_code": "FK to dim_product (trimmed, upper-cased).", "country": "FK to dim_country.",
        "quantity": "Units; negative for cancellations.", "price": "Unit price in GBP (3 decimals).",
        "revenue": "quantity x price (negative for cancellations).",
        "txn_type": "sale / cancellation / adjustment.", "is_product": "False for postage, fees, vouchers, tests.",
        "has_customer": "customer_id is not NULL.",
        "is_reversed": "Sale line fully cancelled later, or the cancellation that reverses it.",
        "source_sheet": "Excel sheet the row came from.", "year_month": "Partition column (yyyy-MM).",
    },
}


def describe_file(path):
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return len(df), ", ".join(df.columns)
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return (len(obj) if isinstance(obj, list) else None), ", ".join(list(obj)[:25]) if isinstance(obj, dict) else ""
    return None, ""


def main():
    problems = []
    rows = []
    for folder, spec, area in [(TABLES_DIR, TABLES, "outputs/tables"), (DASH_DIR, DASHBOARD, "outputs/dashboard"),
                               (FIGURES_DIR, FIGURES, "outputs/figures")]:
        present = {p.name for p in folder.iterdir() if p.is_file()}
        for name in sorted(present - set(spec)):
            problems.append(f"{area}/{name}: no description")
        for name in sorted(set(spec) - present):
            problems.append(f"{area}/{name}: described but missing")
        for name, (step, text) in spec.items():
            p = folder / name
            if not p.exists():
                continue
            n, cols = describe_file(p) if folder != FIGURES_DIR else (None, "")
            rows.append({"area": area, "file": name, "step": step, "description": text,
                         "rows": n, "columns_or_keys": cols, "bytes": p.stat().st_size})
    if problems:
        print("DATA DICTIONARY OUT OF SYNC:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)

    spark = get_spark("16_data_dictionary")
    hive = []
    present = {r["tableName"] for r in spark.sql(f"SHOW TABLES IN {HIVE_DB}").collect()}
    for t in sorted(set(HIVE) - present):
        problems.append(f"Hive {HIVE_DB}.{t}: described but missing")
    for t in sorted(present - set(HIVE)):
        problems.append(f"Hive {HIVE_DB}.{t}: no description")
    for t in HIVE:
        if t not in present:
            continue
        cols = to_pandas(spark.sql(f"DESCRIBE {HIVE_DB}.{t}"))
        cols = cols[~cols["col_name"].str.startswith("#") & (cols["col_name"].str.strip() != "")]
        cols = cols.drop_duplicates("col_name")
        hive.append({"table": t, "rows": spark.table(f"{HIVE_DB}.{t}").count(),
                     "columns": list(zip(cols["col_name"], cols["data_type"]))})
    spark.stop()
    if problems:
        print("DATA DICTIONARY OUT OF SYNC:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    md = ["# Data dictionary", "",
          "Generated by `src/16_data_dictionary.py`. Row counts, columns and Hive types are read from the",
          "files and from Hive; descriptions are written in the script. Step = numbered script in `src/`.", "",
          "**Effective product sales** = `txn_type = 'sale' AND is_product AND NOT is_reversed` "
          "(see `src/03_etl_hive.py`); it is the default measure in steps 06-14.", "",
          f"## Hive tables (database `{HIVE_DB}`, Hive local, embedded Derby metastore)", ""]
    for h in hive:
        md += [f"### {HIVE_DB}.{h['table']} ({h['rows']:,} rows)", "", HIVE[h["table"]], "",
               "| Column | Type | Meaning |", "|---|---|---|"]
        notes = COLUMN_NOTES.get(h["table"], {})
        md += [f"| {c} | {t} | {notes.get(c, '')} |" for c, t in h["columns"]]
        md.append("")
    for area, title in [("outputs/tables", "Result tables"), ("outputs/dashboard", "Dashboard"),
                        ("outputs/figures", "Figures (PNG, 200 dpi)")]:
        md += [f"## {title} (`{area}/`)", ""]
        sub = df[df["area"] == area].sort_values(["step", "file"])
        if area == "outputs/figures":
            md += ["| File | Step | Description |", "|---|---|---|"]
            md += [f"| {r.file} | {r.step} | {r.description} |" for r in sub.itertuples()]
        else:
            md += ["| File | Step | Rows | Description | Columns / keys |", "|---|---|---|---|---|"]
            md += [f"| {r.file} | {r.step} | {'' if pd.isna(r.rows) else f'{int(r.rows):,}'} | "
                   f"{r.description} | {r.columns_or_keys if len(r.columns_or_keys) < 400 else r.columns_or_keys[:400] + ' ...'} |"
                   for r in sub.itertuples()]
        md.append("")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_MD} ({len(df)} files, {len(hive)} Hive tables)")


if __name__ == "__main__":
    main()
