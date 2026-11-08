"""Market-basket analysis with FP-Growth (Spark MLlib) on real invoices.

Transactions: one per invoice of EFFECTIVE product sales (sale AND is_product AND NOT
is_reversed), items = distinct stock codes in the invoice. All invoices are used, including
those without customer_id (a basket is valid without knowing the buyer).

Support threshold: a sensitivity run over several minSupport values records how many
itemsets/rules each produces; the main model uses MIN_SUPPORT = 0.01 (an item set must appear
in >= 1% of invoices, ~390 invoices) and MIN_CONFIDENCE = 0.2. Both are judgement calls.

A single-item rule is flagged similar_name (a simple text heuristic) when the antecedent and
consequent descriptions share at least two meaningful words (stop words such as SET, OF, THE,
AND, WITH and numbers removed). Examples: "POPPY'S PLAYHOUSE BEDROOM" / "... KITCHEN" (one
collection), "STRAWBERRY CHARLOTTE BAG" / "CHARLOTTE BAG PINK POLKADOT" (one item, two designs),
"SET OF 20 VINTAGE CHRISTMAS NAPKINS" / "60 CAKE CASES VINTAGE CHRISTMAS" (one theme).
Such rules show collections or themes bought together; rules without a shared name are the
"cross-product" affinities. The heuristic can miss or over-match and is reported as such.
"""
import json
import re
import time
from string import capwords

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F

from common import HIVE_DB, TABLES_DIR, get_spark, save_table, to_pandas
from plotstyle import SERIES, TEXT_2, plt, save

DB = HIVE_DB
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.2
SENSITIVITY = [0.03, 0.02, 0.015, 0.01]

BASKET_SQL = f"""
SELECT invoice, COLLECT_SET(stock_code) AS items
FROM {DB}.fact_sales
WHERE txn_type = 'sale' AND is_product AND NOT is_reversed
GROUP BY invoice
"""


STOP = {"SET", "OF", "THE", "AND", "WITH", "IN", "A", "FOR", "TO", "DESIGN", "S"}


def words(desc: str) -> set:
    toks = re.sub(r"[^A-Z ]", " ", (desc or "").upper()).split()
    return {t for t in toks if t not in STOP and len(t) > 1}


def is_similar(a: str, b: str) -> bool:
    return len(words(a) & words(b)) >= 2


def main():
    spark = get_spark("09_fpgrowth")
    baskets = spark.sql(BASKET_SQL).cache()
    n_baskets = baskets.count()
    size = baskets.select(F.size("items").alias("n"))
    bstats = size.agg(F.avg("n"), F.expr("percentile_approx(n, 0.5)"), F.max("n"),
                      F.avg((F.col("n") >= 2).cast("int"))).first()
    desc = to_pandas(spark.sql(f"SELECT stock_code, description FROM {DB}.dim_product")) \
        .set_index("stock_code")["description"].to_dict()

    # ---------- sensitivity of the support threshold ----------
    sens = []
    for s in SENSITIVITY:
        t0 = time.time()
        m = FPGrowth(itemsCol="items", minSupport=s, minConfidence=MIN_CONFIDENCE).fit(baskets)
        fi = m.freqItemsets
        sizes = {r["k"]: r["count"] for r in
                 fi.groupBy(F.size("items").alias("k")).count().collect()}
        n_rules = m.associationRules.count()
        sens.append({"min_support": s, "min_invoices": int(np.ceil(s * n_baskets)),
                     "itemsets": int(sum(sizes.values())), "itemsets_size1": sizes.get(1, 0),
                     "itemsets_size2": sizes.get(2, 0), "itemsets_size3plus":
                     int(sum(v for k, v in sizes.items() if k >= 3)), "rules": n_rules,
                     "seconds": round(time.time() - t0, 1)})
        print(sens[-1], flush=True)
    sens = pd.DataFrame(sens)
    save_table(sens, "fpgrowth_support_sensitivity")

    # ---------- main model ----------
    model = FPGrowth(itemsCol="items", minSupport=MIN_SUPPORT, minConfidence=MIN_CONFIDENCE).fit(baskets)
    fi = to_pandas(model.freqItemsets)
    fi["support"] = fi["freq"] / n_baskets
    fi["size"] = fi["items"].apply(len)
    fi["descriptions"] = fi["items"].apply(lambda xs: " + ".join(desc.get(x, x) for x in sorted(xs)))
    fi["items"] = fi["items"].apply(lambda xs: ";".join(sorted(xs)))
    fi = fi.sort_values(["size", "freq"], ascending=[True, False])
    save_table(fi, "fpgrowth_frequent_itemsets")

    rules = to_pandas(model.associationRules)
    rules["antecedent_desc"] = rules["antecedent"].apply(lambda xs: " + ".join(desc.get(x, x) for x in sorted(xs)))
    rules["consequent_desc"] = rules["consequent"].apply(lambda xs: " + ".join(desc.get(x, x) for x in xs))
    rules["n_antecedent"] = rules["antecedent"].apply(len)
    rules["similar_name"] = [
        len(a) == 1 and is_similar(desc.get(a[0], ""), desc.get(c[0], ""))
        for a, c in zip(rules["antecedent"], rules["consequent"])]
    rules["antecedent"] = rules["antecedent"].apply(lambda xs: ";".join(sorted(xs)))
    rules["consequent"] = rules["consequent"].apply(lambda xs: ";".join(xs))
    rules["support_invoices"] = (rules["support"] * n_baskets).round().astype(int)
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    save_table(rules, "fpgrowth_rules")

    spark.sql(f"DROP TABLE IF EXISTS {DB}.assoc_rules")
    spark.sql(f"""CREATE TABLE {DB}.assoc_rules (antecedent STRING, consequent STRING,
                  antecedent_desc STRING, consequent_desc STRING, support DOUBLE,
                  confidence DOUBLE, lift DOUBLE, similar_name BOOLEAN) STORED AS PARQUET""")
    (model.associationRules
     .select(F.array_join(F.array_sort("antecedent"), ";").alias("antecedent"),
             F.array_join("consequent", ";").alias("consequent"), "support", "confidence", "lift")
     .createOrReplaceTempView("r"))
    # descriptions/flags computed in pandas: join back via a small Spark SQL VALUES list is
    # avoided (createDataFrame(pandas) crashes); store the pandas columns through a CSV round-trip.
    # NB: Spark/Hadoop silently ignore files whose name starts with "_" or "."
    tmp = TABLES_DIR / "assoc_rules_tmp.csv"
    rules[["antecedent", "consequent", "antecedent_desc", "consequent_desc", "support",
           "confidence", "lift", "similar_name"]].to_csv(tmp, index=False)
    (spark.read.option("header", True).option("escape", '"')
     .schema("antecedent STRING, consequent STRING, antecedent_desc STRING, consequent_desc STRING, "
             "support DOUBLE, confidence DOUBLE, lift DOUBLE, similar_name BOOLEAN")
     .csv(str(tmp)).write.insertInto(f"{DB}.assoc_rules", overwrite=True))
    n_hive = spark.table(f"{DB}.assoc_rules").count()
    tmp.unlink()

    # ---------- summaries ----------
    cross = rules[~rules["similar_name"]]
    pair = rules[rules["n_antecedent"] == 1]
    summary = {
        "baskets": n_baskets,
        "basket_size_mean": round(float(bstats[0]), 2),
        "basket_size_median": int(bstats[1]),
        "basket_size_max": int(bstats[2]),
        "baskets_with_2plus_items_pct": round(100 * float(bstats[3]), 2),
        "min_support": MIN_SUPPORT, "min_confidence": MIN_CONFIDENCE,
        "frequent_items": int((fi["size"] == 1).sum()),
        "frequent_itemsets": len(fi), "rules": len(rules), "rules_in_hive": n_hive,
        "rules_single_antecedent": len(pair),
        "rules_similar_name": int(rules["similar_name"].sum()),
        "rules_similar_name_pct_of_single_antecedent": round(100 * rules["similar_name"].sum() / max(len(pair), 1), 1),
        "lift_median": round(float(rules["lift"].median()), 2),
        "lift_max": round(float(rules["lift"].max()), 2),
        "confidence_median": round(float(rules["confidence"].median()), 3),
    }
    (TABLES_DIR / "fpgrowth_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- figures ----------
    def top_rules_plot(df, title, name):
        # A->B and B->A have the same lift; keep only the direction with higher confidence
        df = df.assign(pair_key=[tuple(sorted((a, c))) for a, c in
                                 zip(df["antecedent"], df["consequent"])])
        df = df.sort_values(["lift", "confidence"], ascending=False).drop_duplicates("pair_key")
        d = df.head(15).iloc[::-1]
        labels = [f"{capwords(a)[:38]}  →  {capwords(c)[:34]}" for a, c in
                  zip(d["antecedent_desc"], d["consequent_desc"])]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(labels, d["lift"], color=SERIES[0], height=0.6)
        for y, (l, cf, s) in enumerate(zip(d["lift"], d["confidence"], d["support_invoices"])):
            ax.text(l, y, f"  lift {l:.1f} · conf {cf:.0%} · {s:,} inv.", va="center",
                    fontsize=8, color=TEXT_2)
        ax.set_xlim(0, d["lift"].max() * 1.35)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("Lift")
        ax.set_title(title)
        save(fig, name)

    top_rules_plot(pair, "Top 15 product pairs by lift (stronger direction shown)",
                   "fpgrowth_top_rules_all")
    top_rules_plot(cross[cross["n_antecedent"] == 1],
                   "Top 15 product pairs by lift, products without a similar name",
                   "fpgrowth_top_rules_cross")

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="Blues",
                    s=14, edgecolor="none", vmin=1)
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.015, 0.02, 0.03])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, _p: ""))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Support (share of invoices, log scale)")
    ax.set_ylabel("Confidence")
    fig.colorbar(sc, ax=ax, label="Lift")
    ax.set_title(f"{len(rules):,} association rules (minSupport {MIN_SUPPORT:.0%}, "
                 f"minConfidence {MIN_CONFIDENCE:.0%})")
    save(fig, "fpgrowth_rules_scatter")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 45)
    print(json.dumps(summary, indent=2))
    print(sens.to_string(index=False))
    cols = ["antecedent_desc", "consequent_desc", "support_invoices", "confidence", "lift", "similar_name"]
    print(pair[cols].head(10).round(3).to_string(index=False))
    print(cross[cross["n_antecedent"] == 1][cols].head(15).round(3).to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
