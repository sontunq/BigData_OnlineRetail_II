"""Clean/transform Online Retail II with Spark and load a star schema into Hive.

Hive here = Spark's built-in Hive support with an embedded Derby metastore and a local
warehouse folder. It is NOT a Hadoop/HDFS cluster.

Cleaning (rows removed only in steps 1-2; everything else is flagged, not deleted):
  1. Sheet overlap: invoices of 2010-12-01..09 appear in both sheets (verified identical in
     02_quality). The sheet-1 copy is dropped.
  2. Exact duplicates (identical in all 8 source columns) are dropped. Assumption: they are
     double-recorded lines, not two real purchases in the same second.
  3. stock_code is trimmed and upper-cased; description trimmed.
  4. Flags:
       txn_type     'cancellation' invoice starts with C
                    'adjustment'   invoice starts with A, or quantity <= 0, or price <= 0
                    'sale'         everything else
       is_product   false for service / fee / voucher codes (common.NON_PRODUCT_CODES)
       has_customer customer_id is not null
  5. revenue = quantity x price (negative for cancellations).
  6. is_reversed (added after the EDA found orders such as 80,995 x "PAPER CRAFT,
     LITTLE BIRDIE" that were cancelled 12 minutes later): each cancellation line is matched
     to the latest earlier sale line with the same customer_id, stock_code and the opposite
     quantity; both lines of a matched pair get is_reversed = true. Each sale line is used by
     at most one cancellation. Lines without customer_id cannot be matched.
"Gross product sales"     = txn_type = 'sale' AND is_product.
"Effective product sales" = gross product sales AND NOT is_reversed  (main analytics measure).

Tables (database `retail`):
  fact_sales    one row per invoice line, partitioned by year_month
  dim_product   one row per stock_code
  dim_customer  one row per customer_id
  dim_date      one row per calendar date from first to last invoice date
  dim_country   one row per country
"""
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from common import (HIVE_DB, NON_PRODUCT_CODES, NON_PRODUCT_PREFIXES, OVERLAP_START,
                    SHEET1_NAME, get_spark, read_raw, save_table, to_pandas)

KEY_COLS = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price",
            "customer_id", "country"]


def is_non_product(code_col):
    cond = code_col.isin(NON_PRODUCT_CODES)
    for p in NON_PRODUCT_PREFIXES:
        cond = cond | code_col.startswith(p)
    return cond


def clean(raw, log):
    def step(name, df, before):
        after = df.count()
        log.append({"step": name, "rows_before": before, "rows_removed": before - after,
                    "rows_after": after})
        return after

    n0 = raw.count()
    log.append({"step": "0 raw landing rows", "rows_before": n0, "rows_removed": 0, "rows_after": n0})
    df = raw.filter(~((F.col("source_sheet") == SHEET1_NAME) &
                      (F.col("invoice_date") >= F.lit(OVERLAP_START).cast("timestamp"))))
    n1 = step("1 drop sheet-1 copy of the 2010-12-01..09 overlap", df, n0)
    df = df.dropDuplicates(KEY_COLS)
    step("2 drop exact duplicate rows", df, n1)

    code = F.upper(F.trim("stock_code"))
    return (
        df.withColumn("stock_code", code)
        .withColumn("description", F.trim("description"))
        .withColumn("is_product", ~is_non_product(F.col("stock_code")))
        .withColumn("txn_type",
                    F.when(F.col("invoice").startswith("C"), "cancellation")
                    .when(F.col("invoice").startswith("A") | (F.col("quantity") <= 0) |
                          (F.col("price") <= 0), "adjustment")
                    .otherwise("sale"))
        .withColumn("has_customer", F.col("customer_id").isNotNull())
        .withColumn("revenue", (F.col("quantity") * F.col("price")).cast("decimal(18,3)"))
        .withColumn("invoice_ts", F.col("invoice_date"))
        .withColumn("date", F.to_date("invoice_ts"))
        .withColumn("date_key", F.date_format("date", "yyyyMMdd").cast("int"))
        .withColumn("hour", F.hour("invoice_ts"))
        .withColumn("year_month", F.date_format("date", "yyyy-MM"))
    )


def mode_by(df, key, value, out):
    """Most frequent non-null value per key; ties broken alphabetically (deterministic)."""
    w = Window.partitionBy(key).orderBy(F.desc("n"), F.asc(value))
    return (df.filter(F.col(value).isNotNull()).groupBy(key, value).agg(F.count("*").alias("n"))
            .withColumn("r", F.row_number().over(w)).filter("r = 1")
            .select(key, F.col(value).alias(out)))


DDL = {
    "dim_product": """
        CREATE TABLE {db}.dim_product (
            stock_code STRING, description STRING, is_product BOOLEAN,
            n_descriptions INT, n_prices INT, min_price DECIMAL(12,3), max_price DECIMAL(12,3),
            first_sale_date DATE, last_sale_date DATE)
        STORED AS PARQUET""",
    "dim_customer": """
        CREATE TABLE {db}.dim_customer (
            customer_id INT, country STRING, n_countries INT,
            first_purchase_ts TIMESTAMP, last_purchase_ts TIMESTAMP)
        STORED AS PARQUET""",
    "dim_country": """
        CREATE TABLE {db}.dim_country (country STRING, is_uk BOOLEAN)
        STORED AS PARQUET""",
    "dim_date": """
        CREATE TABLE {db}.dim_date (
            date_key INT, `date` DATE, year INT, quarter INT, month INT, month_name STRING,
            year_month STRING, day_of_month INT, weekday_num INT, weekday_name STRING,
            is_weekend BOOLEAN, week_of_year INT, has_transactions BOOLEAN)
        STORED AS PARQUET""",
    "fact_sales": """
        CREATE TABLE {db}.fact_sales (
            line_id BIGINT, invoice STRING, invoice_ts TIMESTAMP, date_key INT, hour INT,
            customer_id INT, stock_code STRING, country STRING, quantity INT,
            price DECIMAL(12,3), revenue DECIMAL(18,3), txn_type STRING, is_product BOOLEAN,
            has_customer BOOLEAN, is_reversed BOOLEAN, source_sheet STRING)
        PARTITIONED BY (year_month STRING)
        STORED AS PARQUET""",
}

FACT_COLS = ["line_id", "invoice", "invoice_ts", "date_key", "hour", "customer_id", "stock_code",
             "country", "quantity", "price", "revenue", "txn_type", "is_product", "has_customer",
             "is_reversed", "source_sheet", "year_month"]   # partition column last


def mark_reversed(fact):
    """Flag sale lines fully cancelled later, and the cancellation lines that cancel them.

    Two-step 1:1 matching on (customer_id, stock_code, opposite quantity, sale not after the
    cancellation): (1) each cancellation picks the latest candidate sale line; (2) if several
    cancellations picked the same sale line, the earliest cancellation keeps it. This is not a
    full assignment: a cancellation that loses in step 2 stays unmatched (conservative).
    """
    keys = ["customer_id", "stock_code"]
    s = fact.filter((F.col("txn_type") == "sale") & F.col("has_customer")).select(
        F.col("line_id").alias("s_id"), F.col("invoice_ts").alias("s_ts"),
        F.col("quantity").alias("s_qty"), *keys)
    c = fact.filter((F.col("txn_type") == "cancellation") & F.col("has_customer")).select(
        F.col("line_id").alias("c_id"), F.col("invoice_ts").alias("c_ts"),
        F.col("quantity").alias("c_qty"), *keys)
    pairs = c.join(s, keys).filter((F.col("s_qty") == -F.col("c_qty")) & (F.col("s_ts") <= F.col("c_ts")))
    per_c = Window.partitionBy("c_id").orderBy(F.desc("s_ts"), F.desc("s_id"))
    pairs = pairs.withColumn("r", F.row_number().over(per_c)).filter("r = 1")
    per_s = Window.partitionBy("s_id").orderBy("c_ts", "c_id")
    pairs = pairs.withColumn("r", F.row_number().over(per_s)).filter("r = 1")
    ids = (pairs.select(F.col("s_id").alias("line_id"))
           .union(pairs.select(F.col("c_id").alias("line_id")))
           .withColumn("is_reversed", F.lit(True)))
    return (fact.join(F.broadcast(ids), "line_id", "left")
            .withColumn("is_reversed", F.coalesce("is_reversed", F.lit(False))))


def build_tables(df):
    w = Window.orderBy("invoice_ts", "invoice", "stock_code", "quantity", "price", "customer_id")
    fact = mark_reversed(df.withColumn("line_id", F.row_number().over(w).cast("bigint")))

    sales = df.filter((F.col("txn_type") == "sale") & F.col("is_product"))
    desc = mode_by(df, "stock_code", "description", "description")
    pstats = (df.groupBy("stock_code").agg(
        F.first("is_product").alias("is_product"),
        F.countDistinct("description").cast("int").alias("n_descriptions")))
    sstats = (sales.groupBy("stock_code").agg(
        F.countDistinct("price").cast("int").alias("n_prices"),
        F.min("price").alias("min_price"), F.max("price").alias("max_price"),
        F.min("date").alias("first_sale_date"), F.max("date").alias("last_sale_date")))
    dim_product = (pstats.join(desc, "stock_code", "left").join(sstats, "stock_code", "left")
                   .select("stock_code", "description", "is_product", "n_descriptions", "n_prices",
                           "min_price", "max_price", "first_sale_date", "last_sale_date"))

    cust = df.filter(F.col("has_customer"))
    cty = mode_by(cust, "customer_id", "country", "country")
    cstats = cust.groupBy("customer_id").agg(F.countDistinct("country").cast("int").alias("n_countries"))
    ptimes = (cust.filter(F.col("txn_type") == "sale").groupBy("customer_id")
              .agg(F.min("invoice_ts").alias("first_purchase_ts"),
                   F.max("invoice_ts").alias("last_purchase_ts")))
    dim_customer = (cstats.join(cty, "customer_id").join(ptimes, "customer_id", "left")
                    .select("customer_id", "country", "n_countries", "first_purchase_ts",
                            "last_purchase_ts"))

    dim_country = (df.select("country").distinct()
                   .withColumn("is_uk", F.col("country") == "United Kingdom"))

    active = df.select("date").distinct().withColumn("has_transactions", F.lit(True))
    bounds = df.agg(F.min("date").alias("a"), F.max("date").alias("b"))
    dim_date = (bounds.select(F.explode(F.sequence("a", "b")).alias("date"))
                .join(active, "date", "left")
                .select(F.date_format("date", "yyyyMMdd").cast("int").alias("date_key"), "date",
                        F.year("date").alias("year"), F.quarter("date").alias("quarter"),
                        F.month("date").alias("month"),
                        F.date_format("date", "MMMM").alias("month_name"),
                        F.date_format("date", "yyyy-MM").alias("year_month"),
                        F.dayofmonth("date").alias("day_of_month"),
                        (((F.dayofweek("date") + 5) % 7) + 1).alias("weekday_num"),
                        F.date_format("date", "EEEE").alias("weekday_name"),
                        F.dayofweek("date").isin(1, 7).alias("is_weekend"),
                        F.weekofyear("date").alias("week_of_year"),
                        F.coalesce("has_transactions", F.lit(False)).alias("has_transactions")))
    return fact, dim_product, dim_customer, dim_country, dim_date


def main():
    spark = get_spark("03_etl_hive")
    spark.conf.set("hive.exec.dynamic.partition", "true")
    spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
    spark.conf.set("hive.exec.max.dynamic.partitions", "1000")

    log = []
    df = clean(read_raw(spark), log).cache()
    fact, dim_product, dim_customer, dim_country, dim_date = build_tables(df)

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {HIVE_DB} COMMENT 'Online Retail II star schema'")
    for name in DDL:
        spark.sql(f"DROP TABLE IF EXISTS {HIVE_DB}.{name}")
        spark.sql(DDL[name].format(db=HIVE_DB))
    dim_product.write.insertInto(f"{HIVE_DB}.dim_product", overwrite=True)
    dim_customer.write.insertInto(f"{HIVE_DB}.dim_customer", overwrite=True)
    dim_country.write.insertInto(f"{HIVE_DB}.dim_country", overwrite=True)
    dim_date.write.insertInto(f"{HIVE_DB}.dim_date", overwrite=True)
    fact.select(*FACT_COLS).write.insertInto(f"{HIVE_DB}.fact_sales", overwrite=True)

    log_pd = pd.DataFrame(log)
    save_table(log_pd, "etl_cleaning_log")

    # Row counts by flag in the loaded fact table
    flags = to_pandas(spark.sql(f"""
        SELECT txn_type, is_product, has_customer, is_reversed, COUNT(*) AS rows,
               COUNT(DISTINCT invoice) AS invoices, SUM(quantity) AS units,
               CAST(SUM(revenue) AS DOUBLE) AS revenue
        FROM {HIVE_DB}.fact_sales GROUP BY txn_type, is_product, has_customer, is_reversed
        ORDER BY txn_type, is_product DESC, has_customer DESC, is_reversed"""))
    save_table(flags, "etl_fact_flag_breakdown")

    rows = []
    for name in DDL:
        cnt = spark.table(f"{HIVE_DB}.{name}").count()
        desc = to_pandas(spark.sql(f"DESCRIBE FORMATTED {HIVE_DB}.{name}"))
        meta = dict(zip(desc["col_name"].str.strip(), desc["data_type"]))
        rows.append({"table": f"{HIVE_DB}.{name}", "rows": cnt, "type": meta.get("Type", ""),
                     "provider": meta.get("Provider", ""), "location": meta.get("Location", ""),
                     "serde": meta.get("Serde Library", "")})
    inv = pd.DataFrame(rows)
    n_parts = spark.sql(f"SHOW PARTITIONS {HIVE_DB}.fact_sales").count()
    inv["partitions"] = [n_parts if t.endswith("fact_sales") else 0 for t in inv["table"]]
    save_table(inv, "hive_tables")

    ri = to_pandas(spark.sql(f"""
        SELECT
          SUM(CASE WHEN p.stock_code IS NULL THEN 1 ELSE 0 END) AS orphan_product,
          SUM(CASE WHEN f.customer_id IS NOT NULL AND c.customer_id IS NULL THEN 1 ELSE 0 END)
              AS orphan_customer,
          SUM(CASE WHEN d.date_key IS NULL THEN 1 ELSE 0 END) AS orphan_date,
          SUM(CASE WHEN k.country IS NULL THEN 1 ELSE 0 END) AS orphan_country,
          COUNT(*) - COUNT(DISTINCT f.line_id) AS duplicate_line_ids
        FROM {HIVE_DB}.fact_sales f
        LEFT JOIN {HIVE_DB}.dim_product p  ON f.stock_code = p.stock_code
        LEFT JOIN {HIVE_DB}.dim_customer c ON f.customer_id = c.customer_id
        LEFT JOIN {HIVE_DB}.dim_date d     ON f.date_key = d.date_key
        LEFT JOIN {HIVE_DB}.dim_country k  ON f.country = k.country"""))
    save_table(ri, "hive_referential_integrity")

    pd.set_option("display.width", 220)
    print(log_pd.to_string(index=False))
    print(flags.round(2).to_string(index=False))
    print(inv.drop(columns=["location", "serde"]).to_string(index=False))
    print(ri.to_string(index=False))
    if int(ri.iloc[0].fillna(0).sum()):
        raise RuntimeError("referential integrity check failed")
    spark.stop()


if __name__ == "__main__":
    main()
