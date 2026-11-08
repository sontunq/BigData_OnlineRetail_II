-- revenue_by_month
SELECT d.year_month,
               COUNT(DISTINCT f.invoice)                 AS invoices,
               COUNT(DISTINCT f.customer_id)             AS customers,
               SUM(f.quantity)                           AS units,
               ROUND(CAST(SUM(f.revenue) AS DOUBLE), 2)  AS revenue
        FROM retail.fact_sales f
        JOIN retail.dim_date d ON f.date_key = d.date_key
        WHERE f.txn_type = 'sale' AND f.is_product
        GROUP BY d.year_month
        ORDER BY d.year_month;

-- revenue_by_country_top15
SELECT f.country, k.is_uk,
               COUNT(DISTINCT f.customer_id)            AS customers,
               COUNT(DISTINCT f.invoice)                AS invoices,
               ROUND(CAST(SUM(f.revenue) AS DOUBLE), 2) AS revenue,
               ROUND(100 * CAST(SUM(f.revenue) AS DOUBLE)
                     / SUM(CAST(SUM(f.revenue) AS DOUBLE)) OVER (), 2) AS revenue_share_pct
        FROM retail.fact_sales f
        JOIN retail.dim_country k ON f.country = k.country
        WHERE f.txn_type = 'sale' AND f.is_product
        GROUP BY f.country, k.is_uk
        ORDER BY revenue DESC
        LIMIT 15;

-- top_products_by_revenue
SELECT p.stock_code, p.description,
               SUM(f.quantity)                          AS units,
               COUNT(DISTINCT f.invoice)                AS invoices,
               ROUND(CAST(SUM(f.revenue) AS DOUBLE), 2) AS revenue
        FROM retail.fact_sales f
        JOIN retail.dim_product p ON f.stock_code = p.stock_code
        WHERE f.txn_type = 'sale' AND f.is_product
        GROUP BY p.stock_code, p.description
        ORDER BY revenue DESC
        LIMIT 15;

-- invoices_by_weekday_hour
SELECT d.weekday_num, d.weekday_name, f.hour,
               COUNT(DISTINCT f.invoice)                AS invoices,
               ROUND(CAST(SUM(f.revenue) AS DOUBLE), 2) AS revenue
        FROM retail.fact_sales f
        JOIN retail.dim_date d ON f.date_key = d.date_key
        WHERE f.txn_type = 'sale' AND f.is_product
        GROUP BY d.weekday_num, d.weekday_name, f.hour
        ORDER BY d.weekday_num, f.hour;

-- cancellation_rate_by_month
SELECT year_month,
               ROUND(-CAST(SUM(CASE WHEN txn_type = 'cancellation' THEN revenue ELSE 0 END) AS DOUBLE), 2)
                   AS cancelled_value,
               ROUND(CAST(SUM(CASE WHEN txn_type = 'sale' THEN revenue ELSE 0 END) AS DOUBLE), 2)
                   AS sales_value,
               ROUND(100 * -CAST(SUM(CASE WHEN txn_type = 'cancellation' THEN revenue ELSE 0 END) AS DOUBLE)
                     / CAST(SUM(CASE WHEN txn_type = 'sale' THEN revenue ELSE 0 END) AS DOUBLE), 2)
                   AS cancelled_pct_of_sales
        FROM retail.fact_sales
        WHERE is_product
        GROUP BY year_month
        ORDER BY year_month;

-- customers_by_year_and_repeat
SELECT YEAR(c.first_purchase_ts) AS first_purchase_year,
               COUNT(*) AS customers,
               SUM(CASE WHEN n.invoices >= 2 THEN 1 ELSE 0 END) AS repeat_customers
        FROM retail.dim_customer c
        JOIN (SELECT customer_id, COUNT(DISTINCT invoice) AS invoices
              FROM retail.fact_sales
              WHERE txn_type = 'sale' AND is_product AND has_customer
              GROUP BY customer_id) n
          ON c.customer_id = n.customer_id
        GROUP BY YEAR(c.first_purchase_ts)
        ORDER BY first_purchase_year;

-- partitions_fact
SHOW PARTITIONS retail.fact_sales;
