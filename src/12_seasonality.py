"""Seasonality and product growth / decline (reads the Hive star schema).

Measure: effective product sales. Comparable windows: Jan-Nov 2010 vs Jan-Nov 2011
(December 2011 is partial; December 2009 has no prior year).
  1. Monthly seasonal index = month revenue / mean monthly revenue of that year;
     correlation of the 2010 and 2011 profiles (Jan-Nov).
  2. Weekly revenue by ISO year/week (early-January days of week 52/53 belong to the
     previous ISO year), 2010 vs 2011 overlay; peak weeks.
  3. Christmas-themed share of revenue by month (description contains CHRISTMAS, XMAS or
     ADVENT - a keyword rule, it misses unnamed seasonal items).
  4. Product growth / decline Jan-Nov 2011 vs Jan-Nov 2010 for products with >= GBP 2,000 in
     either window; new and disappeared products. 'new_in_2011' means no sales in Jan-Nov 2010
     (the product may have launched in December 2010); 'gone_in_2011' means no sales in
     Jan-Nov 2011.
Two seasonal cycles are enough to show that the Q4 peak repeats; they are NOT enough to
separate trend from seasonality statistically (no decomposition is attempted).
"""
import json

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from common import HIVE_DB, TABLES_DIR, get_spark, save_table, to_pandas
from plotstyle import BLUE_DARK, SERIES, TEXT_2, YEAR_COLOR, plt, save

DB = HIVE_DB
MIN_REV = 2000.0
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
EFF = "f.txn_type = 'sale' AND f.is_product AND NOT f.is_reversed"


def gbp(x, _p=None):
    sign, a = ("-" if x < 0 else ""), abs(x)
    body = f"£{a / 1e6:,.1f}M" if a >= 1e6 else (f"£{a / 1e3:,.0f}k" if a >= 1e3 else f"£{a:,.0f}")
    return sign + body


def main():
    spark = get_spark("12_seasonality")
    q = lambda s: to_pandas(spark.sql(s))  # noqa: E731

    # ---------- 1. monthly seasonal index ----------
    mon = q(f"""SELECT d.year, d.month, CAST(SUM(f.revenue) AS DOUBLE) AS revenue
                FROM {DB}.fact_sales f JOIN {DB}.dim_date d ON f.date_key = d.date_key
                WHERE {EFF} GROUP BY d.year, d.month ORDER BY d.year, d.month""")
    comp = mon[(mon["year"].isin([2010, 2011])) & (mon["month"] <= 11)]
    piv = comp.pivot(index="month", columns="year", values="revenue")
    idx = piv / piv.mean()
    idx.columns = [f"index_{c}" for c in idx.columns]
    seas = pd.concat([piv, idx], axis=1).reset_index()
    seas["yoy_pct"] = 100 * (piv[2011] / piv[2010] - 1).values
    save_table(seas, "seasonality_monthly_index")
    corr_p = float(np.corrcoef(piv[2010], piv[2011])[0, 1])
    corr_s = float(piv[2010].rank().corr(piv[2011].rank()))
    q4 = {y: float(piv.loc[9:11, y].sum() / piv[y].sum()) for y in (2010, 2011)}

    # ---------- 2. weekly ----------
    wk = q(f"""WITH t AS (
                 SELECT f.invoice, f.customer_id, f.revenue, WEEKOFYEAR(f.invoice_ts) AS week,
                        CASE WHEN MONTH(f.invoice_ts) = 1 AND WEEKOFYEAR(f.invoice_ts) >= 52
                               THEN YEAR(f.invoice_ts) - 1
                             WHEN MONTH(f.invoice_ts) = 12 AND WEEKOFYEAR(f.invoice_ts) = 1
                               THEN YEAR(f.invoice_ts) + 1
                             ELSE YEAR(f.invoice_ts) END AS iso_year
                 FROM {DB}.fact_sales f WHERE {EFF})
               SELECT iso_year AS year, week, CAST(SUM(revenue) AS DOUBLE) AS revenue,
                      COUNT(DISTINCT invoice) AS invoices, COUNT(DISTINCT customer_id) AS customers
               FROM t GROUP BY iso_year, week ORDER BY year, week""")
    save_table(wk, "seasonality_weekly")
    peaks = {}
    for y in (2010, 2011):
        d = wk[wk["year"] == y]
        r = d.loc[d["revenue"].idxmax()]
        peaks[y] = {"iso_week": int(r["week"]), "revenue": round(float(r["revenue"]), 2)}

    # ---------- 3. Christmas-themed share ----------
    xmas = q(f"""SELECT d.year_month,
                   CAST(SUM(CASE WHEN UPPER(p.description) RLIKE 'CHRISTMAS|XMAS|ADVENT'
                                 THEN f.revenue ELSE 0 END) AS DOUBLE) AS xmas_revenue,
                   CAST(SUM(f.revenue) AS DOUBLE) AS revenue
                 FROM {DB}.fact_sales f
                 JOIN {DB}.dim_date d ON f.date_key = d.date_key
                 JOIN {DB}.dim_product p ON f.stock_code = p.stock_code
                 WHERE {EFF} GROUP BY d.year_month ORDER BY d.year_month""")
    xmas["xmas_share_pct"] = 100 * xmas["xmas_revenue"] / xmas["revenue"]
    save_table(xmas, "seasonality_christmas_share")
    n_xmas = q(f"""SELECT COUNT(*) n FROM {DB}.dim_product
                   WHERE is_product AND UPPER(description) RLIKE 'CHRISTMAS|XMAS|ADVENT'""")["n"].iloc[0]

    # ---------- 4. product growth / decline ----------
    prod = q(f"""SELECT f.stock_code, p.description,
                   CAST(SUM(CASE WHEN YEAR(f.invoice_ts) = 2010 AND MONTH(f.invoice_ts) <= 11
                                 THEN f.revenue ELSE 0 END) AS DOUBLE) AS rev_2010,
                   CAST(SUM(CASE WHEN YEAR(f.invoice_ts) = 2011 AND MONTH(f.invoice_ts) <= 11
                                 THEN f.revenue ELSE 0 END) AS DOUBLE) AS rev_2011,
                   MAX(f.invoice_ts) AS last_sale_ts
                 FROM {DB}.fact_sales f JOIN {DB}.dim_product p ON f.stock_code = p.stock_code
                 WHERE {EFF} GROUP BY f.stock_code, p.description""")
    prod["change"] = prod["rev_2011"] - prod["rev_2010"]
    prod["change_pct"] = np.where(prod["rev_2010"] > 0, 100 * prod["change"] / prod["rev_2010"], np.nan)
    big = prod[(prod["rev_2010"] >= MIN_REV) | (prod["rev_2011"] >= MIN_REV)].copy()
    big["status"] = np.select(
        [big["rev_2010"] == 0, big["rev_2011"] == 0, big["change_pct"] <= -50, big["change_pct"] >= 50],
        ["new_in_2011", "gone_in_2011", "declining", "growing"], "stable")
    save_table(big.sort_values("change"), "product_growth_decline")
    status = big.groupby("status").agg(products=("stock_code", "size"), rev_2010=("rev_2010", "sum"),
                                       rev_2011=("rev_2011", "sum")).reset_index()
    save_table(status, "product_growth_decline_summary")

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    for y in (2010, 2011):
        d = wk[wk["year"] == y]
        ax.plot(d["week"], d["revenue"], color=YEAR_COLOR[y], lw=1.6, label=str(y))
    ax.yaxis.set_major_formatter(FuncFormatter(gbp))
    ax.set_xlabel("ISO week of year")
    ax.set_title("Weekly effective sales (2011 ends in week 49)", fontsize=10)
    ax.legend(loc="upper left")
    ax = axes[1]
    for y in (2010, 2011):
        ax.plot(range(1, 12), idx[f"index_{y}"], color=YEAR_COLOR[y], marker="o", ms=4, label=str(y))
    ax.axhline(1, color=TEXT_2, lw=0.8, ls="--")
    ax.set_xticks(range(1, 12), MONTHS[:11])
    ax.set_title(f"Monthly seasonal index Jan-Nov (r = {corr_p:.2f})", fontsize=10)
    ax.legend(loc="upper left")
    fig.suptitle("Seasonality: the Q4 peak repeats in both years", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    save(fig, "seasonality_weekly_index")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = np.arange(len(xmas))
    ax.bar(x, xmas["xmas_share_pct"], color=SERIES[0], width=0.7)
    ax.set_xticks(x[::2], xmas["year_month"].iloc[::2], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title(f"Share of revenue from Christmas-named products ({n_xmas} products; keyword rule)")
    save(fig, "seasonality_christmas_share")

    dec = big[big["status"].isin(["declining", "gone_in_2011"])].nsmallest(10, "change")
    gro = big[big["status"].isin(["growing", "new_in_2011"])].nlargest(10, "change")
    d = pd.concat([dec, gro]).sort_values("change")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = [BLUE_DARK if v < 0 else "#b83232" for v in d["change"]]
    ax.barh([s.title()[:40] for s in d["description"]], d["change"], color=colors, height=0.6)
    for y, (v, st) in enumerate(zip(d["change"], d["status"])):
        ax.text(v, y, f"  {gbp(v)} ({st.replace('_', ' ')})  " if v > 0 else f"({st.replace('_', ' ')}) {gbp(v)}  ",
                va="center", ha="left" if v > 0 else "right", fontsize=7.5, color=TEXT_2)
    ax.axvline(0, color=TEXT_2, lw=0.8)
    lim = d["change"].abs().max() * 1.6
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(FuncFormatter(gbp))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Biggest product revenue changes, Jan-Nov 2011 vs Jan-Nov 2010")
    save(fig, "product_growth_decline")

    summary = {
        "corr_monthly_profile_2010_2011_pearson": round(corr_p, 3),
        "corr_monthly_profile_2010_2011_spearman": round(corr_s, 3),
        "sep_nov_share_of_jan_nov_2010_pct": round(100 * q4[2010], 1),
        "sep_nov_share_of_jan_nov_2011_pct": round(100 * q4[2011], 1),
        "peak_month_index_2010": round(float(idx["index_2010"].max()), 2),
        "peak_month_index_2011": round(float(idx["index_2011"].max()), 2),
        "peak_week_2010": peaks[2010], "peak_week_2011": peaks[2011],
        "christmas_named_products": int(n_xmas),
        "christmas_share_max_pct": round(float(xmas["xmas_share_pct"].max()), 1),
        "christmas_share_max_month": xmas.loc[xmas["xmas_share_pct"].idxmax(), "year_month"],
        "products_analysed_min_rev": MIN_REV, "products_analysed": len(big),
        "status_counts": big["status"].value_counts().to_dict(),
    }
    (TABLES_DIR / "seasonality_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option("display.width", 220)
    print(json.dumps(summary, indent=2, default=str))
    print(seas.round(2).to_string(index=False))
    print(status.round(0).to_string(index=False))
    print(d[["description", "rev_2010", "rev_2011", "change", "status"]].round(0).to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
