"""Shared paths, schema and SparkSession factory for the Online Retail II pipeline.

Spark runs in local mode with Hive support enabled. The Hive metastore is an embedded
Derby database (metastore/metastore_db) and the warehouse is a local folder (warehouse/).
This is *Hive local*, not a Hadoop/HDFS cluster.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_XLSX = DATA_DIR / "online_retail_II.xlsx"          # original download, never modified
LANDING_DIR = DATA_DIR / "landing"
LANDING_CSV = LANDING_DIR / "online_retail_II.csv"      # both sheets, converted once by 01
WAREHOUSE_DIR = ROOT / "warehouse"
METASTORE_DIR = ROOT / "metastore" / "metastore_db"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
HIVE_DB = "retail"

SEED = 42
SPARK_DRIVER_MEMORY = "2g"   # the development laptop has 8 GB RAM

for d in (LANDING_DIR, WAREHOUSE_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Make Spark workers use the same Python interpreter as the driver.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# On Windows, Hadoop client libraries need winutils.exe/hadoop.dll.
# If tools/hadoop/bin exists in the repo, point HADOOP_HOME at it.
_local_hadoop = ROOT / "tools" / "hadoop"
if os.name == "nt" and (_local_hadoop / "bin" / "winutils.exe").exists():
    os.environ.setdefault("HADOOP_HOME", str(_local_hadoop))
    os.environ["PATH"] = str(_local_hadoop / "bin") + os.pathsep + os.environ["PATH"]


def get_spark(app_name: str = "online-retail"):
    """Return a local SparkSession with Hive support (embedded Derby metastore)."""
    from pyspark.sql import SparkSession

    derby_url = f"jdbc:derby:;databaseName={METASTORE_DIR.as_posix()};create=true"
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        # Plain path, not as_uri(): a URI-encoded space ("%20") would create a folder literally
        # named "Big%20Data" because the project path contains a space.
        .config("spark.sql.warehouse.dir", WAREHOUSE_DIR.as_posix())
        .config("spark.hadoop.javax.jdo.option.ConnectionURL", derby_url)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .enableHiveSupport()
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# Raw columns as delivered by UCI (sheet names "Year 2009-2010" and "Year 2010-2011").
RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
               "Customer ID", "Country"]
# Column names in the landing CSV (snake_case, plus the sheet each row came from).
LANDING_COLUMNS = ["invoice", "stock_code", "description", "quantity", "invoice_date",
                   "price", "customer_id", "country", "source_sheet"]


def raw_schema():
    """Explicit schema for the landing CSV. Invoice and StockCode are strings: they mix
    digits and letters (cancellations start with 'C', some stock codes end with a letter)."""
    from pyspark.sql import types as T

    return T.StructType([
        T.StructField("invoice", T.StringType(), False),
        T.StructField("stock_code", T.StringType(), True),
        T.StructField("description", T.StringType(), True),
        T.StructField("quantity", T.IntegerType(), True),
        T.StructField("invoice_date", T.TimestampType(), True),   # yyyy-MM-dd HH:mm:ss
        # 3 decimals: 18 rows have price 0.001, which DECIMAL(12,2) would round to 0.00
        T.StructField("price", T.DecimalType(12, 3), True),
        T.StructField("customer_id", T.IntegerType(), True),      # null = guest/unknown
        T.StructField("country", T.StringType(), True),
        T.StructField("source_sheet", T.StringType(), True),
    ])


def read_raw(spark, path=None):
    """Read the landing CSV with the explicit schema. FAILFAST makes malformed rows raise."""
    return (
        spark.read.option("header", True)
        .option("mode", "FAILFAST")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .option("multiLine", False)
        .option("escape", '"')
        .schema(raw_schema())
        .csv(str(path or LANDING_CSV))
    )


# ---------------- Cleaning rules shared by ETL (Spark) and reconciliation (pandas) -------------
# Stock codes that are services, fees, adjustments or vouchers, not merchandise.
NON_PRODUCT_CODES = ["POST", "DOT", "M", "C2", "C3", "D", "S", "B", "BANK CHARGES", "ADJUST",
                     "ADJUST2", "AMAZONFEE", "CRUK", "TEST001", "TEST002", "GIFT"]
NON_PRODUCT_PREFIXES = ["GIFT_0001"]          # gift vouchers (codes compared upper-cased)
# The two sheets overlap: sheet 1 runs to 2010-12-09 and sheet 2 starts on 2010-12-01, and the
# invoices of that window appear in both. Rows of sheet 1 on/after this date are dropped.
SHEET1_NAME, SHEET2_NAME = "Year 2009-2010", "Year 2010-2011"
OVERLAP_START = "2010-12-01"


def to_pandas(sdf):
    """Collect a SMALL Spark result to pandas without relying on Arrow/toPandas compatibility
    (toPandas / createDataFrame(pandas) crash the Python worker with pandas 3 + PySpark 3.5)."""
    import pandas as pd
    from pyspark.sql import functions as F

    # Collecting a TimestampType converts it to the OS time zone (e.g. +7h on a Vietnamese
    # laptop). Format timestamps as text inside Spark (session time zone = UTC = wall clock of
    # the data) and parse them back as naive datetimes.
    ts_cols = [c for c, t in sdf.dtypes if t == "timestamp"]
    if ts_cols:
        sdf = sdf.select(*[F.date_format(c, "yyyy-MM-dd HH:mm:ss").alias(c) if c in ts_cols
                           else F.col(c) for c in sdf.columns])
    rows = [r.asDict() for r in sdf.collect()]
    out = pd.DataFrame(rows, columns=sdf.columns)
    for c in ts_cols:
        out[c] = pd.to_datetime(out[c])
    return out


def save_table(df_pandas, name: str):
    """Save a pandas DataFrame to outputs/tables/<name>.csv and return the path."""
    path = TABLES_DIR / f"{name}.csv"
    df_pandas.to_csv(path, index=False)
    return path
