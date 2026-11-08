"""Verify the environment and that Spark runs with Hive support (local)."""
import json
import platform
import subprocess
import sys

from common import TABLES_DIR, get_spark


def java_version() -> str:
    out = subprocess.run(["java", "-version"], capture_output=True, text=True)
    return (out.stderr or out.stdout).splitlines()[0]


def main():
    import matplotlib
    import pandas
    import pyspark
    import sklearn

    spark = get_spark("00_check_env")
    catalog = spark.conf.get("spark.sql.catalogImplementation")

    # Smoke test: create a managed Hive table, write, read back, drop.
    spark.sql("CREATE DATABASE IF NOT EXISTS env_check")
    spark.range(5).write.mode("overwrite").saveAsTable("env_check.smoke")
    n = spark.sql("SELECT COUNT(*) AS n FROM env_check.smoke").collect()[0]["n"]
    spark.sql("DROP TABLE env_check.smoke")
    spark.sql("DROP DATABASE env_check")

    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "java": java_version(),
        "pyspark": pyspark.__version__,
        "spark": spark.version,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit_learn": sklearn.__version__,
        "spark_master": spark.sparkContext.master,
        "catalog_implementation": catalog,
        "warehouse_dir": spark.conf.get("spark.sql.warehouse.dir"),
        "hive_smoke_test_rows": int(n),
        "hive_smoke_test_ok": catalog == "hive" and n == 5,
    }
    spark.stop()
    (TABLES_DIR / "environment.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
