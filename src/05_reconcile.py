"""Reconcile an independent pandas implementation of the cleaning rules against the
Spark/Hive star schema. Mismatches are REPORTED, never adjusted; exit code 1 on any mismatch.

pandas reads the landing CSV directly (not Hive) and re-implements the rules of 03_etl_hive.py.
Money is summed in integer thousandths (price has 3 decimals) to avoid float drift.

Output: outputs/tables/reconciliation.csv, reconciliation_monthly.csv
"""
import sys

import numpy as np
import pandas as pd

from common import (HIVE_DB, LANDING_CSV, NON_PRODUCT_CODES, NON_PRODUCT_PREFIXES,
                    OVERLAP_START, SHEET1_NAME, get_spark, save_table, to_pandas)

KEY_COLS = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price",
            "customer_id", "country"]


def pandas_side():
    d = pd.read_csv(LANDING_CSV, dtype={"invoice": str, "stock_code": str, "description": str,
                                        "country": str, "customer_id": "Int64"})
    m = {"raw_rows": len(d)}
    ts = pd.to_datetime(d["invoice_date"])
    d = d[~((d["source_sheet"] == SHEET1_NAME) & (ts >= pd.Timestamp(OVERLAP_START)))]
    m["rows_after_overlap"] = len(d)
    d = d.drop_duplicates(KEY_COLS)
    m["fact_rows"] = len(d)

    code = d["stock_code"].str.strip().str.upper()
    non_prod = code.isin(NON_PRODUCT_CODES)
    for p in NON_PRODUCT_PREFIXES:
        non_prod |= code.str.startswith(p)
    inv = d["invoice"]
    txn = np.where(inv.str.startswith("C"), "cancellation",
                   np.where(inv.str.startswith("A") | (d["quantity"] <= 0) | (d["price"] <= 0),
                            "adjustment", "sale"))
    rev_milli = d["quantity"].astype("int64") * (d["price"] * 1000).round().astype("int64")
    d = d.assign(code=code, is_product=~non_prod, txn_type=txn, rev_milli=rev_milli,
                 ym=pd.to_datetime(d["invoice_date"]).dt.strftime("%Y-%m"))

    # Reversed sale/cancellation pairs, re-implemented independently (see 03 mark_reversed).
    d = d.assign(ts=pd.to_datetime(d["invoice_date"])).reset_index(drop=True)
    d["rid"] = d.sort_values(["ts", "invoice", "code", "quantity", "price", "customer_id"],
                             kind="mergesort").index.argsort() + 1
    keys = ["customer_id", "code"]
    s = d[(d["txn_type"] == "sale") & d["customer_id"].notna()][["rid", "ts", "quantity", *keys]]
    c = d[(d["txn_type"] == "cancellation") & d["customer_id"].notna()][["rid", "ts", "quantity", *keys]]
    c = c.assign(quantity=-c["quantity"])
    p = c.merge(s, on=keys + ["quantity"], suffixes=("_c", "_s"))
    p = p[p["ts_s"] <= p["ts_c"]]
    p = p.sort_values(["rid_c", "ts_s", "rid_s"], ascending=[True, False, False]).drop_duplicates("rid_c")
    p = p.sort_values(["rid_s", "ts_c", "rid_c"]).drop_duplicates("rid_s")
    rev_ids = set(p["rid_s"]) | set(p["rid_c"])
    d["is_reversed"] = d["rid"].isin(rev_ids)
    m["reversed_lines"] = int(d["is_reversed"].sum())

    ps = d[(d["txn_type"] == "sale") & d["is_product"]]
    eff = ps[~ps["is_reversed"]]
    m["reversed_product_sale_lines"] = int(ps["is_reversed"].sum())
    m["effective_sales_rows"] = len(eff)
    m["effective_sales_revenue"] = eff["rev_milli"].sum() / 1000
    m["effective_sales_customers"] = eff["customer_id"].nunique()
    m.update({
        "rows_sale": int((d["txn_type"] == "sale").sum()),
        "rows_cancellation": int((d["txn_type"] == "cancellation").sum()),
        "rows_adjustment": int((d["txn_type"] == "adjustment").sum()),
        "rows_non_product": int((~d["is_product"]).sum()),
        "rows_missing_customer": int(d["customer_id"].isna().sum()),
        "product_sales_rows": len(ps),
        "product_sales_invoices": ps["invoice"].nunique(),
        "product_sales_units": int(ps["quantity"].sum()),
        "product_sales_revenue": ps["rev_milli"].sum() / 1000,
        "product_sales_customers": ps["customer_id"].nunique(),
        "product_sales_with_customer_revenue": ps.loc[ps["customer_id"].notna(), "rev_milli"].sum() / 1000,
        "all_lines_net_revenue": d["rev_milli"].sum() / 1000,
        "distinct_stock_codes": d["code"].nunique(),
        "distinct_customers": d["customer_id"].nunique(),
        "distinct_countries": d["country"].nunique(),
        "days_with_transactions": pd.to_datetime(d["invoice_date"]).dt.date.nunique(),
    })
    monthly = (ps.groupby("ym").agg(revenue_milli=("rev_milli", "sum"), invoices=("invoice", "nunique"))
               .reset_index())
    monthly["revenue"] = monthly["revenue_milli"] / 1000
    return m, monthly[["ym", "invoices", "revenue"]]


def spark_side():
    spark = get_spark("05_reconcile")
    f = f"{HIVE_DB}.fact_sales"
    log = pd.read_csv("../outputs/tables/etl_cleaning_log.csv")
    m = {"raw_rows": int(log["rows_after"].iloc[0]),
         "rows_after_overlap": int(log["rows_after"].iloc[1])}
    r = spark.sql(f"""
        SELECT COUNT(*) AS fact_rows,
               SUM(CASE WHEN txn_type = 'sale' THEN 1 ELSE 0 END) AS rows_sale,
               SUM(CASE WHEN txn_type = 'cancellation' THEN 1 ELSE 0 END) AS rows_cancellation,
               SUM(CASE WHEN txn_type = 'adjustment' THEN 1 ELSE 0 END) AS rows_adjustment,
               SUM(CASE WHEN NOT is_product THEN 1 ELSE 0 END) AS rows_non_product,
               SUM(CASE WHEN NOT has_customer THEN 1 ELSE 0 END) AS rows_missing_customer,
               CAST(SUM(revenue) AS DOUBLE) AS all_lines_net_revenue,
               COUNT(DISTINCT stock_code) AS distinct_stock_codes,
               COUNT(DISTINCT customer_id) AS distinct_customers,
               COUNT(DISTINCT country) AS distinct_countries,
               COUNT(DISTINCT date_key) AS days_with_transactions
        FROM {f}""").first().asDict()
    m.update(r)
    s = spark.sql(f"""
        SELECT COUNT(*) AS product_sales_rows,
               COUNT(DISTINCT invoice) AS product_sales_invoices,
               SUM(quantity) AS product_sales_units,
               CAST(SUM(revenue) AS DOUBLE) AS product_sales_revenue,
               COUNT(DISTINCT customer_id) AS product_sales_customers,
               CAST(SUM(CASE WHEN has_customer THEN revenue ELSE 0 END) AS DOUBLE)
                   AS product_sales_with_customer_revenue
        FROM {f} WHERE txn_type = 'sale' AND is_product""").first().asDict()
    m.update(s)
    e = spark.sql(f"""
        SELECT SUM(CASE WHEN is_reversed THEN 1 ELSE 0 END) AS reversed_lines,
               SUM(CASE WHEN is_reversed AND txn_type = 'sale' AND is_product THEN 1 ELSE 0 END)
                   AS reversed_product_sale_lines,
               SUM(CASE WHEN txn_type = 'sale' AND is_product AND NOT is_reversed THEN 1 ELSE 0 END)
                   AS effective_sales_rows,
               CAST(SUM(CASE WHEN txn_type = 'sale' AND is_product AND NOT is_reversed
                             THEN revenue ELSE 0 END) AS DOUBLE) AS effective_sales_revenue,
               COUNT(DISTINCT CASE WHEN txn_type = 'sale' AND is_product AND NOT is_reversed
                                   THEN customer_id END) AS effective_sales_customers
        FROM {f}""").first().asDict()
    m.update(e)
    monthly = to_pandas(spark.sql(f"""
        SELECT year_month AS ym, COUNT(DISTINCT invoice) AS invoices,
               CAST(SUM(revenue) AS DOUBLE) AS revenue
        FROM {f} WHERE txn_type = 'sale' AND is_product GROUP BY year_month"""))
    spark.stop()
    return m, monthly


def main():
    pm, pmon = pandas_side()
    sm, smon = spark_side()
    rows = []
    for k in pm:
        a, b = pm[k], sm[k]
        ok = abs(float(a) - float(b)) <= 0.0005 if isinstance(a, float) else int(a) == int(b)
        rows.append({"metric": k, "pandas": a, "spark_hive": b, "match": ok})
    rec = pd.DataFrame(rows, dtype=object)
    save_table(rec, "reconciliation")

    mon = pmon.merge(smon, on="ym", how="outer", suffixes=("_pandas", "_spark")).sort_values("ym")
    mon["match"] = (mon["invoices_pandas"] == mon["invoices_spark"]) & \
                   ((mon["revenue_pandas"] - mon["revenue_spark"]).abs() <= 0.0005)
    save_table(mon, "reconciliation_monthly")

    pd.set_option("display.width", 200)
    print(rec.to_string(index=False))
    print(f"\nmonthly rows: {len(mon)}, all match: {bool(mon['match'].all())}")
    if not (rec["match"].all() and mon["match"].all()):
        print("RECONCILIATION FAILED: pandas and Spark/Hive disagree", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
