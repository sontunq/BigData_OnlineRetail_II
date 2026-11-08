#!/usr/bin/env bash
# OPTIONAL - only for a machine that has an HDFS cluster. NOT used in this project run:
# the pipeline reads from the local landing folder data/landing/.
# Run 01_ingest.py first so that the landing CSV exists.
set -euo pipefail
HDFS_DIR=${HDFS_DIR:-/user/$USER/online_retail/landing}
hdfs dfs -mkdir -p "$HDFS_DIR"
hdfs dfs -put -f "data/landing/online_retail_II.csv" "$HDFS_DIR/online_retail_II.csv"
hdfs dfs -ls "$HDFS_DIR"
