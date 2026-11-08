"""Exploratory data analysis of Online Retail II (reads the Hive star schema).

Measure: EFFECTIVE product sales = txn_type 'sale' AND is_product AND NOT is_reversed
(see 03_etl_hive.py). Gross sales and cancellations are reported separately.
Outputs: outputs/tables/eda_*.csv|json and outputs/figures/eda_*.png (200 dpi).
"""
import json

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from common import HIVE_DB, TABLES_DIR, get_spark, save_table, to_pandas
from plotstyle import (BLUE_DARK, BLUE_LIGHT, NEUTRAL, SERIES, TEXT_2, WEEKDAYS, YEAR_COLOR,
                       plt, save)

DB = HIVE_DB
EFF = "f.txn_type = 'sale' AND f.is_product AND NOT f.is_reversed"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def gbp(x, _pos=None):
    if abs(x) >= 1e6:
        return f"£{x / 1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"£{x / 1e3:,.0f}k"
    return f"£{x:,.0f}"


def num(df):
    """Spark DECIMAL -> float."""
    for c in df.columns:
        if len(df) and type(df[c].dropna().iloc[0] if df[c].notna().any() else 0).__name__ == "Decimal":
            df[c] = df[c].astype(float)
    return df


def main():
    spark = get_spark("06_eda")
    q = lambda sql: num(to_pandas(spark.sql(sql)))  # noqa: E731
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW eff AS
        SELECT f.*, CAST(f.revenue AS DOUBLE) AS rev, d.year, d.month, d.weekday_num,
               d.weekday_name, p.description
        FROM {DB}.fact_sales f
        JOIN {DB}.dim_date d ON f.date_key = d.date_key
        JOIN {DB}.dim_product p ON f.stock_code = p.stock_code
        WHERE {EFF}""")
    spark.table("eff").cache().count()

    # ---------------- KPIs ----------------
    k = q("""SELECT SUM(rev) revenue, COUNT(*) lines, SUM(quantity) units,
                    COUNT(DISTINCT invoice) invoices, COUNT(DISTINCT customer_id) customers,
                    COUNT(DISTINCT stock_code) products, COUNT(DISTINCT country) countries,
                    SUM(CASE WHEN has_customer THEN rev ELSE 0 END) revenue_with_customer,
                    COUNT(DISTINCT CASE WHEN NOT has_customer THEN invoice END) guest_invoices
             FROM eff""").iloc[0].to_dict()
    g = q(f"""SELECT
          SUM(CASE WHEN f.txn_type='sale' THEN CAST(f.revenue AS DOUBLE) ELSE 0 END) gross_sales,
          SUM(CASE WHEN f.txn_type='sale' AND f.is_reversed THEN CAST(f.revenue AS DOUBLE) ELSE 0 END) reversed_sales,
          -SUM(CASE WHEN f.txn_type='cancellation' AND NOT f.is_reversed THEN CAST(f.revenue AS DOUBLE) ELSE 0 END)
              other_cancellations
          FROM {DB}.fact_sales f WHERE f.is_product""").iloc[0].to_dict()
    kpis = {key: (int(v) if float(v).is_integer() else round(float(v), 2))
            for key, v in {**k, **g}.items()}
    kpis["revenue_with_customer_pct"] = round(100 * k["revenue_with_customer"] / k["revenue"], 2)
    kpis["avg_lines_per_invoice"] = round(k["lines"] / k["invoices"], 2)
    kpis["avg_invoice_value"] = round(k["revenue"] / k["invoices"], 2)

    # ---------------- Monthly + year-over-year ----------------
    mon = q("""SELECT year, month, SUM(rev) revenue, COUNT(DISTINCT invoice) invoices,
                      COUNT(DISTINCT customer_id) customers, SUM(quantity) units
               FROM eff GROUP BY year, month ORDER BY year, month""")
    mon["year_month"] = mon["year"].astype(str) + "-" + mon["month"].astype(str).str.zfill(2)
    mon["partial_month"] = mon["year_month"] == "2011-12"
    save_table(mon, "eda_monthly")
    full = mon[~mon["partial_month"]]
    yoy = full.pivot(index="month", columns="year", values="revenue")
    yoy_growth = (yoy[2011] / yoy[2010] - 1) * 100
    save_table(yoy.assign(growth_2011_vs_2010_pct=yoy_growth).reset_index(), "eda_yoy_monthly")
    kpis["revenue_2010_full_year"] = round(float(mon.loc[mon["year"] == 2010, "revenue"].sum()), 2)
    jan_nov = lambda y: float(mon.loc[(mon["year"] == y) & (mon["month"] <= 11), "revenue"].sum())  # noqa
    kpis["revenue_jan_nov_2010"] = round(jan_nov(2010), 2)
    kpis["revenue_jan_nov_2011"] = round(jan_nov(2011), 2)
    kpis["growth_jan_nov_2011_vs_2010_pct"] = round(100 * (jan_nov(2011) / jan_nov(2010) - 1), 2)
    peak = full.loc[full["revenue"].idxmax()]
    kpis["peak_month"] = peak["year_month"]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for y in (2010, 2011):
        d = full[full["year"] == y]
        ax.plot(d["month"], d["revenue"], color=YEAR_COLOR[y], marker="o", ms=5, label=str(y))
    dec09 = mon[mon["year_month"] == "2009-12"]["revenue"].iloc[0]
    ax.scatter([12], [dec09], color=YEAR_COLOR[2009], s=40, zorder=3, label="Dec 2009")
    ax.set_xticks(range(1, 13), MONTHS)
    ax.yaxis.set_major_formatter(FuncFormatter(gbp))
    ax.set_title("Monthly effective product sales, year over year (Dec 2011 partial, not shown)")
    ax.legend(loc="upper left")
    save(fig, "eda_monthly_yoy")

    # ---------------- Weekday / hour ----------------
    wd = q(f"""SELECT e.weekday_num, e.weekday_name, SUM(e.rev) revenue,
                      COUNT(DISTINCT e.invoice) invoices, COUNT(DISTINCT e.date_key) trading_days
               FROM eff e GROUP BY e.weekday_num, e.weekday_name ORDER BY e.weekday_num""")
    wd["revenue_per_trading_day"] = wd["revenue"] / wd["trading_days"]
    save_table(wd, "eda_weekday")
    hr = q("""SELECT hour, COUNT(DISTINCT invoice) invoices, SUM(rev) revenue FROM eff
              GROUP BY hour ORDER BY hour""")
    save_table(hr, "eda_hour")
    wh = q("""SELECT weekday_num, hour, COUNT(DISTINCT invoice) invoices FROM eff
              GROUP BY weekday_num, hour""")
    save_table(wh, "eda_weekday_hour")
    pv = wh.pivot(index="weekday_num", columns="hour", values="invoices").fillna(0)
    pv = pv.reindex(range(1, 8)).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 2]})
    ax = axes[0]
    sat = wd[wd["weekday_num"] == 6]
    wd_plot = wd[wd["weekday_num"] != 6]          # one Saturday only in 2 years: not comparable
    ax.bar(wd_plot["weekday_name"].str[:3], wd_plot["revenue_per_trading_day"], color=SERIES[0],
           width=0.6)
    ax.yaxis.set_major_formatter(FuncFormatter(gbp))
    n_sat = int(sat["trading_days"].sum()) if len(sat) else 0
    ax.set_title(f"Revenue per trading day (Saturday excluded:\n{n_sat} trading day in 2 years)",
                 fontsize=10)
    ax = axes[1]
    im = ax.imshow(pv.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(7), [w[:3] for w in WEEKDAYS])
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_xlabel("Hour")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Invoices (2 years)")
    ax.set_title("Invoices by weekday and hour", fontsize=10)
    save(fig, "eda_weekday_hour")

    # ---------------- Countries ----------------
    cty = q("""SELECT country, SUM(rev) revenue, COUNT(DISTINCT customer_id) customers,
                      COUNT(DISTINCT invoice) invoices FROM eff GROUP BY country ORDER BY revenue DESC""")
    cty["revenue_share_pct"] = 100 * cty["revenue"] / cty["revenue"].sum()
    cty["revenue_per_customer"] = cty["revenue"] / cty["customers"].replace(0, np.nan)
    save_table(cty, "eda_country")
    kpis["uk_revenue_share_pct"] = round(float(cty.loc[cty["country"] == "United Kingdom",
                                                       "revenue_share_pct"].iloc[0]), 2)
    top = cty[cty["country"] != "United Kingdom"].head(10).iloc[::-1]   # UK share in the title
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(top["country"], top["revenue"], color=SERIES[0], height=0.6)
    for y, (v, p) in enumerate(zip(top["revenue"], top["revenue_share_pct"])):
        ax.text(v, y, f"  {gbp(v)} ({p:.1f}%)", va="center", fontsize=8, color=TEXT_2)
    ax.xaxis.set_major_formatter(FuncFormatter(gbp))
    ax.set_xlim(0, top["revenue"].max() * 1.3)
    ax.set_title(f"Top 10 non-UK countries (UK = {kpis['uk_revenue_share_pct']:.1f}% of revenue)")
    save(fig, "eda_countries")

    # ---------------- Products + Pareto ----------------
    prod = q("""SELECT stock_code, description, SUM(rev) revenue, SUM(quantity) units,
                       COUNT(DISTINCT invoice) invoices, COUNT(DISTINCT customer_id) customers
                FROM eff GROUP BY stock_code, description ORDER BY revenue DESC, stock_code""")
    prod["cum_share_pct"] = 100 * prod["revenue"].cumsum() / prod["revenue"].sum()
    save_table(prod, "eda_product")
    n80 = int((prod["cum_share_pct"] < 80).sum() + 1)
    kpis["products_for_80pct_revenue"] = n80
    kpis["products_for_80pct_revenue_pct_of_catalog"] = round(100 * n80 / len(prod), 1)
    t15 = prod.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(t15["description"].str.title(), t15["revenue"], color=SERIES[0], height=0.6)
    for y, (v, n) in enumerate(zip(t15["revenue"], t15["invoices"])):
        ax.text(v, y, f"  {gbp(v)} · {n:,} invoices", va="center", fontsize=8, color=TEXT_2)
    ax.xaxis.set_major_formatter(FuncFormatter(gbp))
    ax.set_xlim(0, t15["revenue"].max() * 1.35)
    ax.set_title("Top 15 products by effective sales")
    save(fig, "eda_top_products")

    # ---------------- Customers ----------------
    cust = q("""SELECT customer_id, SUM(rev) revenue, COUNT(DISTINCT invoice) invoices
                FROM eff WHERE has_customer GROUP BY customer_id ORDER BY revenue DESC""")
    cust["cum_share_pct"] = 100 * cust["revenue"].cumsum() / cust["revenue"].sum()
    n = len(cust)
    for p in (1, 10, 20):
        kpis[f"top{p}pct_customers_revenue_share_pct"] = round(
            float(cust["revenue"].head(int(np.ceil(n * p / 100))).sum() / cust["revenue"].sum() * 100), 2)
    kpis["customers_one_invoice_pct"] = round(100 * float((cust["invoices"] == 1).mean()), 2)
    kpis["customer_invoices_median"] = float(cust["invoices"].median())
    kpis["customer_revenue_median"] = round(float(cust["revenue"].median()), 2)
    freq = cust["invoices"].clip(upper=20).value_counts().sort_index()
    save_table(freq.rename_axis("invoices_capped_20").rename("customers").reset_index(),
               "eda_customer_frequency")
    curve = pd.DataFrame({"customer_pct": 100 * np.arange(1, n + 1) / n,
                          "revenue_cum_pct": cust["cum_share_pct"].values})
    save_table(curve.iloc[::10], "eda_customer_pareto")
    prod_curve = pd.DataFrame({"product_pct": 100 * np.arange(1, len(prod) + 1) / len(prod),
                               "revenue_cum_pct": prod["cum_share_pct"].values})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(curve["customer_pct"], curve["revenue_cum_pct"], color=SERIES[0], label="Customers")
    ax.plot(prod_curve["product_pct"], prod_curve["revenue_cum_pct"], color=SERIES[1], label="Products")
    ax.plot([0, 100], [0, 100], color=NEUTRAL, lw=1, ls="--")
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Share of customers / products (ranked by revenue)")
    ax.set_title("Concentration of revenue (Lorenz-style)", fontsize=10)
    ax.legend(loc="lower right")
    ax.annotate(f"top 10% customers = {kpis['top10pct_customers_revenue_share_pct']:.0f}%",
                (10, kpis["top10pct_customers_revenue_share_pct"]), xytext=(12, -4),
                textcoords="offset points", fontsize=8, color=TEXT_2)
    ax = axes[1]
    ax.bar(freq.index, 100 * freq.values / n, color=SERIES[0], width=0.7)
    ax.set_xticks([1, 5, 10, 15, 20], ["1", "5", "10", "15", "20+"])
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Invoices per customer (2 years)")
    ax.set_title(f"Purchase frequency ({kpis['customers_one_invoice_pct']:.0f}% bought once)",
                 fontsize=10)
    save(fig, "eda_customers")

    # ---------------- Invoices ----------------
    inv = q("""SELECT invoice, has_customer, COUNT(*) lines, SUM(rev) value FROM eff
               GROUP BY invoice, has_customer""")
    summ = inv.groupby("has_customer").agg(
        invoices=("invoice", "size"), lines_median=("lines", "median"), lines_mean=("lines", "mean"),
        value_median=("value", "median"), value_mean=("value", "mean"),
        value_p99=("value", lambda v: v.quantile(0.99))).reset_index()
    save_table(summ, "eda_invoice_summary")
    kpis["invoice_lines_median"] = float(inv["lines"].median())
    kpis["invoice_value_median"] = round(float(inv["value"].median()), 2)
    kpis["invoices_with_2plus_lines_pct"] = round(100 * float((inv["lines"] >= 2).mean()), 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in [(axes[0], "lines", "Lines per invoice"),
                           (axes[1], "value", "Invoice value (£)")]:
        bins = np.logspace(0, np.log10(inv[col].clip(lower=1).max()), 40)
        for hc, color, label in [(True, SERIES[0], "Registered customer"),
                                 (False, SERIES[1], "No customer id")]:
            v = inv.loc[inv["has_customer"] == hc, col].clip(lower=1)
            ax.hist(v, bins=bins, color=color, alpha=0.75, label=label, edgecolor="white",
                    linewidth=0.4)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:,.0f}"))
        ax.set_xlabel(f"{title} (log scale)")
        ax.set_ylabel("Invoices")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right")
    save(fig, "eda_invoices")

    # ---------------- Cancellations ----------------
    canc = q(f"""SELECT f.year_month,
            SUM(CASE WHEN f.txn_type='sale' THEN CAST(f.revenue AS DOUBLE) ELSE 0 END) gross_sales,
            SUM(CASE WHEN f.txn_type='sale' AND f.is_reversed THEN CAST(f.revenue AS DOUBLE) ELSE 0 END) reversed_sales,
            -SUM(CASE WHEN f.txn_type='cancellation' AND NOT f.is_reversed THEN CAST(f.revenue AS DOUBLE) ELSE 0 END) partial_cancellations
            FROM {DB}.fact_sales f WHERE f.is_product GROUP BY f.year_month ORDER BY f.year_month""")
    canc["reversed_pct_of_gross"] = 100 * canc["reversed_sales"] / canc["gross_sales"]
    canc["partial_cancel_pct_of_gross"] = 100 * canc["partial_cancellations"] / canc["gross_sales"]
    save_table(canc, "eda_cancellations_monthly")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(canc))
    ax.bar(x, canc["reversed_pct_of_gross"], color=SERIES[0], width=0.7, label="Fully reversed lines")
    ax.bar(x, canc["partial_cancel_pct_of_gross"], bottom=canc["reversed_pct_of_gross"],
           color=SERIES[1], width=0.7, label="Other cancellations")
    ax.set_xticks(x[::2], canc["year_month"].iloc[::2], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Cancelled value as % of gross product sales")
    ax.legend(loc="upper left")
    save(fig, "eda_cancellations")

    # ---------------- New vs returning customers ----------------
    nr = q("""WITH firsts AS (SELECT customer_id, MIN(year_month) first_ym FROM eff
                              WHERE has_customer GROUP BY customer_id)
              SELECT e.year_month,
                     COUNT(DISTINCT CASE WHEN e.year_month = f.first_ym THEN e.customer_id END) new_customers,
                     COUNT(DISTINCT CASE WHEN e.year_month > f.first_ym THEN e.customer_id END) returning_customers
              FROM eff e JOIN firsts f ON e.customer_id = f.customer_id
              GROUP BY e.year_month ORDER BY e.year_month""")
    save_table(nr, "eda_new_vs_returning")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(nr))
    ax.bar(x, nr["returning_customers"], color=SERIES[0], width=0.7, label="Returning")
    ax.bar(x, nr["new_customers"], bottom=nr["returning_customers"], color=SERIES[1], width=0.7,
           label="New (first purchase)")
    ax.set_xticks(x[::2], nr["year_month"].iloc[::2], rotation=45, ha="right")
    ax.set_ylabel("Active customers")
    ax.set_title("Active customers per month: new vs returning (Dec 2009 = start of data, all 'new')")
    ax.legend(loc="upper left")
    save(fig, "eda_new_vs_returning")

    (TABLES_DIR / "eda_kpis.json").write_text(json.dumps(kpis, indent=2, default=str))
    pd.set_option("display.width", 200)
    print(json.dumps(kpis, indent=2, default=str))
    print(yoy.assign(growth=yoy_growth).round(1).to_string())
    print(wd.round(1).to_string(index=False))
    print(summ.round(2).to_string(index=False))
    print(prod.head(10)[["description", "revenue", "invoices"]].round(0).to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
