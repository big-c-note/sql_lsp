-- Complex query to test LSP parsing & metadata resolution
WITH recent_orders AS (
  SELECT o.order_id, o.user_id, o.order_date
from catalog2.schema2.orders o
  WHERE o.order_date > current_date() - INTERVAL 7 DAYS
),

high_value_customers AS (
  SELECT ro.user_id, SUM(oi.quantity * p.price) AS total_spent
  FROM recent_orders ro
  JOIN catalog3.schema3.order_items oi ON ro.order_id = oi.order_id
  JOIN catalog1.schema1.products p ON oi.product_id = p.product_id
  GROUP BY ro.user_id
  HAVING total_spent > 100
),

review_stats AS (
  SELECT
    r.product_id,
    COUNT(*) AS review_count,
    AVG(r.rating) AS avg_rating,
    MAX(r.review_time) AS last_review
  FROM catalog2.schema2.reviews r
  GROUP BY r.product_id
)

SELECT
  u.user_id,
  u.name,
  u.email,
  hvc.total_spent,
  p.name AS product_name,
  p.attributes.color,
  rs.review_count,
  rs.avg_rating,
  ARRAY_CONTAINS(p.attributes.tags, 'electronics') AS is_electronic,
  ROUND(oi.quantity * p.price * (1 - oi.discount_percent / 100), 2) AS net_price
FROM high_value_customers
JOIN catalog.schema1
-- FROM high_value_customers
-- JOIN catalog1.schema1.users u ON u.user_id = hvc.user_id
-- JOIN catalog2.schema2.orders o ON o.user_id = u.user_id
-- JOIN catalog3.schema3.order_items oi ON o.order_id = oi.order_id
-- JOIN catalog1.schema1.products p ON p.product_id = oi.product_id
-- LEFT JOIN review_stats rs ON rs.product_id = p.product_id
WHERE u.preferences['theme'] = 'dark'
  AND p.price > 50
ORDER BY net_price DESC;
