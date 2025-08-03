import pytest
from sqlglot import parse_one, exp
from sqlglot.errors import ErrorLevel


@pytest.fixture
def sql():
    return "-- Complex query to test LSP parsing & metadata resolution\nWITH recent_orders AS (\n  SELECT o.order_id, o.user_id, o.order_date\nF\n-- from catalog2.schema2.orders o\n  WHERE o.order_date > current_date() - INTERVAL 7 DAYS\n),\n\nhigh_value_customers AS (\n  SELECT ro.user_id, SUM(oi.quantity * p.price) AS total_spent\n  FROM recent_orders ro\n  JOIN catalog3.schema3.order_items oi ON ro.order_id = oi.order_id\n  JOIN catalog1.schema1.products p ON oi.product_id = p.product_id\n  GROUP BY ro.user_id\n  HAVING total_spent > 100\n),\n\nreview_stats AS (\n  SELECT\n    r.product_id,\n    COUNT(*) AS review_count,\n    AVG(r.rating) AS avg_rating,\n    MAX(r.review_time) AS last_review\n  FROM catalog2.schema2.reviews r\n  GROUP BY r.product_id\n)\n\nSELECT\n  u.user_id,\n  u.name,\n  u.email,\n  hvc.total_spent,\n  p.name AS product_name,\n  p.attributes.color,\n  rs.review_count,\n  rs.avg_rating,\n  ARRAY_CONTAINS(p.attributes.tags, 'electronics') AS is_electronic,\n  ROUND(oi.quantity * p.price * (1 - oi.discount_percent / 100), 2) AS net_price\nFROM high_value_customers hvc\nJOIN catalog1.schema1.users u ON u.user_id = hvc.user_id\nJOIN catalog2.schema2.orders o ON o.user_id = u.user_id\nJOIN catalog3.schema3.order_items oi ON o.order_id = oi.order_id\nJOIN catalog1.schema1.products p ON p.product_id = oi.product_id\nLEFT JOIN review_stats rs ON rs.product_id = p.product_id\nWHERE u.preferences['theme'] = 'dark'\n  AND p.price > 50\nORDER BY net_price DESC;\n"


@pytest.fixture
def sql2():
    return "-- Complex query to test LSP parsing & metadata resolution\nWITH recent_orders AS (\n  SELECT o2.order_id, o2.user_id, o2.order_date\nfrom (select order_id, user_id, order_date from catalog2.schema2.orders) o2\n  WHERE o2.order_date > current_date() - INTERVAL 7 DAYS\n),\n\nhigh_value_customers_0 as (\n  SELECT ro.user_id, oi.quantity, p.price\n  FROM recent_orders ro\n  JOIN catalog3.schema3.order_items oi ON ro.order_id = oi.order_id\n  JOIN catalog1.schema1.products p ON oi.product_id = p.product_id\n), \n\nhigh_value_customers AS (\n  SELECT hvc.user_id, SUM(hvc.quantity * hvc.price) AS total_spent\nFROM high_value_customers_0 as hvc\n  GROUP BY hvc.user_id\n  HAVING total_spent > 100\n),\n\nreview_stats AS (\n  SELECT\n    r.product_id,\n    COUNT(*) AS review_count,\n    AVG(r.rating) AS avg_rating,\n    MAX(r.review_time) AS last_review\n  FROM catalog2.schema2.reviews r\n  GROUP BY r.product_id\n)\n\nSELECT\n  u.user_id,\n  u.name,\n  u.email,\n  hvc.total_spent,\n  p.name AS product_name,\n  p.attributes.color,\n  rs.review_count,\n  rs.avg_rating,\n  ARRAY_CONTAINS(p.attributes.tags, 'electronics') AS is_electronic,\n  ROUND(oi.quantity * p.price * (1 - oi.discount_percent / 100), 2) AS net_price\n FROM high_value_customers hvc \n JOIN catalog1.schema1.users u ON u.user_id = hvc.user_id\n JOIN catalog2.schema2.orders o ON o.user_id = u.user_id\n JOIN catalog3.schema3.order_items oi ON o.order_id = oi.order_id\n JOIN catalog1.schema1.products p ON p.product_id = oi.product_id\n LEFT JOIN review_stats rs ON rs.product_id = p.product_id\nWHERE u.preferences['theme'] = 'dark'\n  AND p.price > 50\nORDER BY net_price DESC;\n"


def test_get_candidate_ctes(sql):
    from databricks_sql_lsp import get_candidate_cte_names

    tree = parse_one(sql, dialect="spark", error_level=ErrorLevel.IGNORE)
    candidate_ctes = get_candidate_cte_names(tree)
    assert candidate_ctes == ["recent_orders", "high_value_customers", "review_stats"]


def test_get_table_nodes_from_select(sql2):
    from databricks_sql_lsp import get_table_nodes_from_select

    select_node = parse_one(sql2, dialect="spark", error_level=ErrorLevel.IGNORE)
    table_nodes = get_table_nodes_from_select(select_node)
    table_aliases = [node.alias_or_name for node in table_nodes]
    for node in table_nodes:
        assert isinstance(node, exp.Table)
    assert table_aliases == ["hvc", "u", "o", "oi", "p", "rs"]


def test_get_table_nodes_from_select_2(sql2):
    from databricks_sql_lsp import get_table_nodes_from_select

    tree = parse_one(sql2, dialect="spark", error_level=ErrorLevel.IGNORE)
    column_node = [
        node
        for node in tree.find_all(exp.Column)
        if node.this.this == "order_id" and node.table == "o2"
    ][0]
    select_node = column_node.parent_select
    table_nodes = get_table_nodes_from_select(select_node)
    table_aliases = [node.alias_or_name for node in table_nodes]
    for node in table_nodes:
        assert isinstance(node, exp.Subquery)
    assert table_aliases == ["o2"]


def test_get_candidate_column_names(sql2):
    from databricks_sql_lsp import get_candidate_column_names

    tree = parse_one(sql2, dialect="spark", error_level=ErrorLevel.IGNORE)
    column_node = [
        node
        for node in tree.find_all(exp.Column)
        if node.this.this == "total_spent" and node.table == "hvc"
    ][0]
    candidate_column_names = get_candidate_column_names(column_node, tree)
    expected_column_names = set(
        [
            "hvc.user_id",
            "hvc.total_spent",
            "u.user_id",
            "u.email",
            "u.name",
            "u.preferences",
            "u.created_at",
            "o.order_id",
            "o.user_id",
            "o.amount",
            "o.order_date",
            "o.is_gift",
            "oi.order_id",
            "oi.order_item_id",
            "oi.product_id",
            "oi.discount_percent",
            "oi.quantity",
            "p.price",
            "p.attributes",
            "p.product_id",
            "p.name",
            "rs.product_id",
            "rs.review_count",
            "rs.avg_rating",
            "rs.last_review",
        ]
    )
    assert set(candidate_column_names) == expected_column_names


def test_get_candidate_column_names_2(sql2):
    from databricks_sql_lsp import get_candidate_column_names

    tree = parse_one(sql2, dialect="spark", error_level=ErrorLevel.IGNORE)
    column_node = [
        node
        for node in tree.find_all(exp.Column)
        if node.this.this == "order_id" and node.table == "o2"
    ][0]
    candidate_column_names = get_candidate_column_names(column_node, tree)
    expected_column_names = set(["o2.order_id", "o2.user_id", "o2.order_date"])
    assert set(candidate_column_names) == expected_column_names


def test_annotate_types(sql2):
    from databricks_sql_lsp import get_all_table_names, get_columns_for_table

    from sqlglot.optimizer.annotate_types import annotate_types
    from sqlglot.expressions import DataType
    from sqlglot.optimizer.qualify import qualify

    """
    type_map = {
        "STRING": "VARCHAR",
        "VARCHAR": "VARCHAR",
        "CHAR": "CHAR",
        "BOOLEAN": "BOOLEAN",
        "BOOL": "BOOLEAN",
        "BINARY": "BINARY",
        "DATE": "DATE",
        "TIMESTAMP": "TIMESTAMP",
        "TIMESTAMP_NTZ": "TIMESTAMP",
        "TIMESTAMP_LTZ": "TIMESTAMPLTZ",
        "DECIMAL": "DECIMAL",
        "NUMERIC": "DECIMAL",
        "FLOAT": "FLOAT",
        "DOUBLE": "DOUBLE",
        "DOUBLE PRECISION": "DOUBLE",
        "REAL": "FLOAT",
        "BYTE": "TINYINT",
        "TINYINT": "TINYINT",
        "SMALLINT": "SMALLINT",
        "SHORT": "SMALLINT",
        "INT": "INT",
        "INTEGER": "INT",
        "BIGINT": "BIGINT",
        "LONG": "BIGINT",
        "ARRAY": "ARRAY",
        "MAP": "MAP",
        "STRUCT": "STRUCT",
    }
    """
    type_map = {
        "STRING": "VARCHAR",
        "DOUBLE": "DOUBLE",
        "FLOAT": "FLOAT",
        "ARRAY": "ARRAY",
        "MAP": "MAP",
        "BYTE": "SMALLINT",
        "DECIMAL": "DECIMAL",
        "LONG": "INT",
        "INT": "INT",
        "DATE": "DATE",
        "BOOLEAN": "BOOLEAN",
        "STRUCT": "STRUCT",
        "TIMESTAMP": "TIMESTAMP",
    }

    table_names = get_all_table_names()
    from collections import defaultdict

    schema = defaultdict(lambda: defaultdict(dict))

    types = set()
    for table_name in table_names:
        parts = table_name.split(".")
        if len(parts) != 3:
            continue
        catalog, db, table = parts
        try:
            columns = get_columns_for_table(table_name)
            types = types.union(set([col["type_name"] for col in columns]))
            column_schema = {}
            for col in columns:
                if "type_name" in col.keys():
                    column_schema[col["name"]] = type_map[col["type_name"]]
                elif "type" in col.keys():
                    column_schema[col["name"]] = type_map[col["type"]]
                else:
                    pass
            schema[catalog][db][table] = column_schema
        except KeyError:
            pass
    tree = parse_one(sql2, dialect="spark")
    qual = qualify(tree, dialect="spark", schema=schema)
    with_types = annotate_types(qual, dialect="spark", schema=schema)
    column_nodes = {
        node.this.this: node._type.this.value
        for node in with_types.find_all(exp.Column)
    }
    # Note: there is a column net price that is supposed to be unknown because
    # it is incorrectly used in the order by statement.
    expected_nodes = {
        "user_id": "INT",
        "name": "VARCHAR",
        "email": "VARCHAR",
        "total_spent": "DOUBLE",
        "review_count": "BIGINT",
        "avg_rating": "DOUBLE",
        "attributes": "STRUCT",
        "order_id": "INT",
        "product_id": "INT",
        "net_price": "UNKNOWN",
        "price": "DOUBLE",
        "quantity": "INT",
        "preferences": "MAP",
        "order_date": "DATE",
        "rating": "INT",
        "review_time": "TIMESTAMPTZ",
        "discount_percent": "FLOAT",
    }
    import pdb

    pdb.set_trace()
