"""Ingest Online Retail II into a local landing zone and read it with Spark.

Source: UCI Machine Learning Repository, "Online Retail II" (id 502), CC BY 4.0.
The original workbook data/online_retail_II.xlsx is never modified. Spark cannot read .xlsx
natively, so both sheets are converted ONCE with pandas/openpyxl into one CSV in the local
landing folder (data/landing/). HDFS is NOT used (see scripts/hdfs_put.sh for a cluster).

The conversion is skipped when the landing CSV already exists and the workbook MD5 matches
the one recorded in the manifest.
"""
import hashlib
import json
import time

import pandas as pd

from common import (LANDING_COLUMNS, LANDING_CSV, RAW_COLUMNS, RAW_XLSX, ROOT, TABLES_DIR,
                    get_spark, raw_schema, read_raw)

MANIFEST = TABLES_DIR / "ingest_manifest.json"


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(xlsx_md5: str) -> dict:
    t0 = time.time()
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None, engine="openpyxl",
                           dtype={"Invoice": str, "StockCode": str, "Description": str,
                                  "Country": str})
    parts, sheet_rows = [], {}
    for name, df in sheets.items():
        missing = set(RAW_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"sheet {name!r} is missing columns {missing}")
        df = df[RAW_COLUMNS].copy()
        df["source_sheet"] = name
        sheet_rows[name] = len(df)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df.columns = LANDING_COLUMNS
    # Customer ID is stored as a float in Excel (e.g. 13085.0); it is an integer key.
    if not (df["customer_id"].dropna() % 1 == 0).all():
        raise ValueError("non-integer customer ids found")
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(LANDING_CSV, index=False, lineterminator="\n")
    return {"sheets": sheet_rows, "rows_written": len(df),
            "conversion_seconds": round(time.time() - t0, 1)}


def main():
    xlsx_md5 = md5(RAW_XLSX)
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    if LANDING_CSV.exists() and old.get("source_md5") == xlsx_md5:
        conv = old["conversion"]
        print("Landing CSV is up to date; conversion skipped.")
    else:
        print("Converting workbook to CSV (one-off, may take a few minutes)...", flush=True)
        conv = convert(xlsx_md5)

    spark = get_spark("01_ingest")
    df = read_raw(spark)
    n_rows = df.count()          # FAILFAST: raises here if any row violates the schema
    df.printSchema()
    df.show(5, truncate=False)
    by_sheet = {r["source_sheet"]: r["count"] for r in df.groupBy("source_sheet").count().collect()}
    spark.stop()

    if n_rows != conv["rows_written"]:
        raise RuntimeError(f"Spark read {n_rows} rows, conversion wrote {conv['rows_written']}")

    manifest = {
        "source_file": RAW_XLSX.relative_to(ROOT).as_posix(),
        "source": "UCI Machine Learning Repository, Online Retail II (id 502), CC BY 4.0",
        "source_md5": xlsx_md5,
        "source_bytes": RAW_XLSX.stat().st_size,
        "landing_file": LANDING_CSV.relative_to(ROOT).as_posix(),
        "landing_bytes": LANDING_CSV.stat().st_size,
        "storage": "local filesystem landing folder (no HDFS)",
        "conversion": conv,
        "rows_read_by_spark": n_rows,
        "rows_by_sheet_spark": by_sheet,
        "schema": {f.name: f.dataType.simpleString() for f in raw_schema().fields},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
