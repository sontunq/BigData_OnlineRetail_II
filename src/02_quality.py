"""Data quality report on the RAW landing data (before cleaning), with Spark.

Outputs (outputs/tables/):
  quality_summary.csv            one row per check: value, status (PASS/WARN/FAIL), note
  quality_nulls.csv              null / blank count per column
  quality_sheet_overlap.csv      the 2010-12-01..09 window as it appears in each sheet
  quality_non_product_codes.csv  service / fee / voucher codes with line counts
  quality_invoice_types.csv      sale / cancellation (C...) / adjustment (A...) invoices
  quality_monthly_coverage.csv   rows and invoices per month and sheet

WARN means "real issue that the ETL handles"; FAIL means the data cannot be used as is.
"""
import pandas as pd
from pyspark.sql import functions as F

from common import (NON_PRODUCT_CODES, NON_PRODUCT_PREFIXES, OVERLAP_START, SHEET1_NAME,
                    SHEET2_NAME, get_spark, read_raw, save_table, to_pandas)

KEY_COLS = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price",
            "customer_id", "country"]


def non_product_expr(col="stock_code"):
    code = F.upper(F.trim(F.col(col)))
    cond = code.isin(NON_PRODUCT_CODES)
    for p in NON_PRODUCT_PREFIXES:
        cond = cond | code.startswith(p)
    return cond


def main():
    spark = get_spark("02_quality")
    df = read_raw(spark).cache()
    n = df.count()
    checks = []

    def add(check, value, ok, note=""):
        status = "WARN" if ok is None else ("PASS" if bool(ok) else "FAIL")
        checks.append({"check": check, "value": value, "status": status, "note": note})

    add("row_count", n, n > 0)
    add("rows_sheet_2009_2010", df.filter(F.col("source_sheet") == SHEET1_NAME).count(), True)
    add("rows_sheet_2010_2011", df.filter(F.col("source_sheet") == SHEET2_NAME).count(), True)

    # ---------- nulls ----------
    rows = []
    for c, t in df.dtypes:
        cond = F.col(c).isNull()
        if t == "string":
            cond = cond | (F.trim(F.col(c)) == "")
        rows.append({"column": c, "dtype": t, "null_or_blank": df.filter(cond).count()})
    nulls = pd.DataFrame(rows)
    nulls["pct"] = 100 * nulls["null_or_blank"] / n
    save_table(nulls, "quality_nulls")
    miss_cust = int(nulls.loc[nulls["column"] == "customer_id", "null_or_blank"].iloc[0])
    miss_desc = int(nulls.loc[nulls["column"] == "description", "null_or_blank"].iloc[0])
    add("missing_customer_id_rows", miss_cust, None,
        f"{100 * miss_cust / n:.2f}% of rows; kept for revenue, excluded from customer models")
    add("missing_description_rows", miss_desc, None, "description filled from dim_product mode")
    for c in ["invoice", "stock_code", "quantity", "invoice_date", "price", "country"]:
        v = int(nulls.loc[nulls["column"] == c, "null_or_blank"].iloc[0])
        add(f"missing_{c}", v, v == 0)

    # ---------- sheet overlap ----------
    win = df.filter(F.col("invoice_date") >= F.lit(OVERLAP_START).cast("timestamp")) \
            .filter(F.col("invoice_date") < F.lit("2010-12-10").cast("timestamp"))
    ov = to_pandas(win.groupBy("source_sheet").agg(
        F.count("*").alias("rows"), F.countDistinct("invoice").alias("invoices"),
        F.min("invoice_date").alias("first_ts"), F.max("invoice_date").alias("last_ts")))
    for c in ["first_ts", "last_ts"]:
        ov[c] = ov[c].astype(str)
    save_table(ov, "quality_sheet_overlap")
    s1 = win.filter(F.col("source_sheet") == SHEET1_NAME).select(*KEY_COLS)
    s2 = win.filter(F.col("source_sheet") == SHEET2_NAME).select(*KEY_COLS)
    only_s1, only_s2 = s1.exceptAll(s2).count(), s2.exceptAll(s1).count()
    add("overlap_window_rows_sheet1", s1.count(), None,
        "2010-12-01..09 appears in both sheets; ETL drops the sheet-1 copy")
    add("overlap_rows_only_in_sheet1", only_s1, only_s1 == 0, "0 = sheet-1 copy is fully contained in sheet 2")
    add("overlap_rows_only_in_sheet2", only_s2, only_s2 == 0)

    # ---------- duplicates ----------
    no_overlap = df.filter(~((F.col("source_sheet") == SHEET1_NAME) &
                             (F.col("invoice_date") >= F.lit(OVERLAP_START).cast("timestamp"))))
    n_no = no_overlap.count()
    dup = n_no - no_overlap.dropDuplicates(KEY_COLS).count()
    add("exact_duplicate_rows_after_overlap_removal", dup, None,
        "identical in every column incl. timestamp; dropped by ETL (assumption)")

    # ---------- invoice / transaction types ----------
    typed = df.withColumn("inv_type", F.when(F.col("invoice").startswith("C"), "cancellation")
                          .when(F.col("invoice").startswith("A"), "adjustment_A")
                          .when(F.col("invoice").rlike("^[0-9]+$"), "numeric")
                          .otherwise("other"))
    it = to_pandas(typed.groupBy("inv_type").agg(F.count("*").alias("rows"),
                                                 F.countDistinct("invoice").alias("invoices"),
                                                 F.sum("quantity").alias("units"))
                   .orderBy(F.desc("rows")))
    save_table(it, "quality_invoice_types")
    other_inv = int(it.loc[it["inv_type"] == "other", "rows"].sum())
    add("invoices_with_unknown_format", other_inv, other_inv == 0)
    add("cancellation_rows", int(it.loc[it["inv_type"] == "cancellation", "rows"].sum()), None,
        "invoice starts with C; negative quantities")
    canc_pos = typed.filter((F.col("inv_type") == "cancellation") & (F.col("quantity") > 0)).count()
    add("cancellation_rows_with_positive_qty", canc_pos, True if canc_pos == 0 else None,
        "C496350: one 'Manual' (code M) line, a non-product adjustment")

    # ---------- invalid values ----------
    q0 = df.filter((F.col("quantity") <= 0) & ~F.col("invoice").startswith("C")).count()
    add("non_cancellation_rows_qty_le_0", q0, None, "stock write-offs, mostly without customer")
    p0 = df.filter(F.col("price") == 0).count()
    pneg = df.filter(F.col("price") < 0).count()
    add("price_eq_0_rows", p0, None, "free items / write-offs; excluded from sales")
    add("price_lt_0_rows", pneg, None, "'Adjust bad debt' (invoice A...)")
    stats = df.agg(F.min("quantity"), F.max("quantity"), F.min("price"), F.max("price")).first()
    add("quantity_min", stats[0], True)
    add("quantity_max", stats[1], True)
    add("price_min", float(stats[2]), True)
    add("price_max", float(stats[3]), True)
    add("price_with_3_decimals_rows",
        df.filter(F.col("price") != F.round("price", 2)).count(), True, "kept: price is DECIMAL(12,3)")

    # ---------- stock codes ----------
    npc = to_pandas(df.filter(non_product_expr()).groupBy(F.upper(F.trim("stock_code")).alias("code"))
                    .agg(F.count("*").alias("rows"), F.first("description", True).alias("description"),
                         F.sum(F.col("quantity") * F.col("price")).alias("amount"))
                    .orderBy(F.desc("rows")))
    npc["amount"] = npc["amount"].astype(float)
    save_table(npc, "quality_non_product_codes")
    add("non_product_rows", int(npc["rows"].sum()), None,
        f"{len(npc)} service/fee/voucher codes; flagged is_product = false")
    lower = df.filter(F.col("stock_code") != F.upper("stock_code")) \
              .select("stock_code").distinct().count()
    add("stock_codes_with_lower_case", lower, None, "upper-cased in ETL")
    prod = df.filter(~non_product_expr()).withColumn("code", F.upper(F.trim("stock_code")))
    n_codes = prod.select("code").distinct().count()
    multi_desc = (prod.filter(F.col("description").isNotNull()).groupBy("code")
                  .agg(F.countDistinct("description").alias("n")).filter("n > 1").count())
    multi_price = (prod.filter(F.col("price") > 0).groupBy("code")
                   .agg(F.countDistinct("price").alias("n")).filter("n > 1").count())
    add("distinct_product_codes", n_codes, True)
    add("product_codes_with_multiple_descriptions", multi_desc, None, "dim_product uses the most frequent")
    add("product_codes_with_multiple_prices", multi_price, None, "price varies by time and quantity tier")

    # ---------- customers, countries ----------
    cust = df.filter(F.col("customer_id").isNotNull())
    add("distinct_customers", cust.select("customer_id").distinct().count(), True)
    multi_cty = (cust.groupBy("customer_id").agg(F.countDistinct("country").alias("n"))
                 .filter("n > 1").count())
    add("customers_with_multiple_countries", multi_cty, None, "dim_customer uses the most frequent")
    add("distinct_countries", df.select("country").distinct().count(), True)

    # ---------- date coverage ----------
    rng = to_pandas(df.agg(F.min("invoice_date").alias("a"), F.max("invoice_date").alias("b")))
    first_ts, last_ts = rng["a"].iloc[0], rng["b"].iloc[0]
    add("first_timestamp", str(first_ts), True)
    add("last_timestamp", str(last_ts), True, "December 2011 covers only 9 days")
    days = df.select(F.to_date("invoice_date").alias("d")).distinct()
    n_days = days.count()
    span = (last_ts.date() - first_ts.date()).days + 1
    add("calendar_days_in_range", span, True)
    add("days_with_transactions", n_days, True)
    add("days_without_transactions", span - n_days, None,
        "includes Saturdays (shop does not trade) and the Christmas closures")
    sat = days.filter(F.dayofweek("d") == 7).count()
    add("saturdays_with_transactions", sat, True)
    monthly = to_pandas(df.groupBy(F.date_format("invoice_date", "yyyy-MM").alias("month"),
                                   "source_sheet")
                        .agg(F.count("*").alias("rows"), F.countDistinct("invoice").alias("invoices"))
                        .orderBy("month", "source_sheet"))
    save_table(monthly, "quality_monthly_coverage")

    summary = pd.DataFrame(checks)
    save_table(summary, "quality_summary")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print(summary.to_string(index=False))
    print(ov.to_string(index=False))
    print(it.to_string(index=False))
    print(npc.to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
