from collections import defaultdict, deque
import json
import logging
from pathlib import Path
import typing as t

from lsprotocol.types import (
    CompletionItem,
    CompletionList,
    CompletionParams,
    CompletionItemKind,
    Diagnostic,
    DiagnosticSeverity,
    DidChangeTextDocumentParams,
    Hover,
    HoverParams,
    MarkupContent,
    MarkupKind,
    Position,
    Range,
)
from pygls.server import LanguageServer
from sqlglot import parse_one, exp
from sqlglot.errors import ErrorLevel, OptimizeError
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.qualify import qualify

# TODO: how to set debug?
logging.basicConfig(filename="sql_lsp.log", filemode="w", level=logging.DEBUG)


server = LanguageServer("sql_lsp", "v0.1")

# TODO: make this variable via config.
# TODO readme on adding other dialects
# TODO get this a lighter weight pull in a dialect specific dir
# Try using update_positions
# TODO add tests
# TODO try using sql glot for the bfs
# TODO try docstrings

DIALECT = "spark"
CATALOG_SCHEMA_PATH: str = "~/.nvim-databricks/databricks_schema.json"


def _get_databricks_schema(
    catalog_schema_path: str | Path,
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    catalog_schema_path = Path(catalog_schema_path)
    with open(catalog_schema_path) as f:
        # TODO: validate catalog db table schema with response and column /
        # type infos
        catalog_schema = json.load(f)

    # TODO: finish types
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
    # Note: this is the schema as per sqlglot schema. However, we could add
    # arbitrary metadata and change the getter funcs to extend hover
    # capabilities.
    schema: dict[str, dict[str, dict[str, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for catalog in catalog_schema["catalogs"].keys():
        for db in catalog_schema["catalogs"][catalog]["schemas"].keys():
            for table in catalog_schema["catalogs"][catalog]["schemas"][db][
                "tables"
            ].keys():
                column_schema = {}
                columns = (
                    catalog_schema.get("catalogs", {})
                    .get(catalog, {})
                    .get("schemas", {})
                    .get(db, {})
                    .get("tables", {})
                    .get(table, {})
                    .get("response", {})
                    .get("columns", {})
                )
                for col in columns:
                    if "type_name" in col.keys():
                        column_schema[col["name"]] = type_map[col["type_name"]]
                    elif "type" in col.keys():
                        column_schema[col["name"]] = type_map[col["type"]]
                    else:
                        logging.warning(
                            f"There is no type information under keys `type` and `type_name` for column {col['name']}"
                        )
                # If there are no columns, then sqlglot will error on it.
                if column_schema:
                    schema[catalog][db][table] = column_schema
    return schema


def get_schema(
    catalog_schema_path: str | Path, dialect: str = "databicks"
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    catalog_schema_path = Path(catalog_schema_path).expanduser()
    if dialect == "databricks" or dialect == "spark":
        # TODO change to spark or have additional databricks dialect that
        # converts to spark sqlglot dialect
        schema = _get_databricks_schema(catalog_schema_path)
    else:
        raise NotImplementedError("Only databricks sql is currently supported")
    return schema


SCHEMA = get_schema(CATALOG_SCHEMA_PATH, DIALECT)


# Sqlglot helpers
def bfs_limited(
    root: exp.Expression,
    max_depth: int = 1,
    match_types: t.Tuple[t.Type[exp.Expression]] = (),
) -> t.Iterator[exp.Expression]:
    # TODO: push this as helper to sqlglot or see if it can be done with walk
    # and depth.
    queue = deque([(root, 0)])  # node and its depth

    while queue:
        node, depth = queue.popleft()

        if depth > max_depth:
            continue

        if depth > 0:  # skip root node itself
            if not match_types or isinstance(node, match_types):
                yield node

        # only add children if we haven't reached max depth yet
        if depth < max_depth:
            queue.extend((child, depth + 1) for child in node.iter_expressions())


# Interop between sqlglot and pygls
# TODO may need to change if end index changes
def lsp_position_to_cursor_offset(lines: list[str], line: int, character: int) -> int:
    return sum(len(ln) + 1 for ln in lines[:line]) + character


def offset_to_lsp_position(lines: list[str], offset: int) -> Position:
    total = 0
    for i, line in enumerate(lines):
        if total + len(line) + 1 > offset:
            return Position(line=i, character=offset - total)
        total += len(line) + 1
    return Position(line=len(lines) - 1, character=len(lines[-1]))


def find_node_before_cursor(tree: exp.Select, cursor_offset: int):
    """
    Find the deepest node in the AST that starts before or contains the cursor position.
    """
    best_node = None
    best_span = -1

    for node in tree.walk():
        meta = node.meta or {}
        # TODO change this if we do update_pos
        start = meta.get("start")
        end = meta.get("end")

        if start is not None and end is not None:
            if start <= cursor_offset <= end:
                # Prefer deeper (smaller span) matches
                span = end - start
                if best_node is None or span <= best_span:
                    best_node = node
                    best_span = span

    return best_node


def get_columns_from_database_table(
    full_table_name: str, table_alias: str = "", schema=SCHEMA
) -> dict[str, str]:
    # TODO: consider allowing for non full tables or assert the
    # schema structure
    catalog, db, table = full_table_name.split(".")
    columns: dict[str, str] = schema[catalog][db][table]
    columns_with_alias: dict[str, str] = {
        table_alias + col: _type for col, _type in columns.items()
    }
    return columns_with_alias


def get_table_nodes_from_select(
    select_node: exp.Select,
) -> list[exp.Table | exp.Subquery]:
    """
    Get all table or subquery nodes from (FROM, JOIN, CTEs) from a Select node.
    Note: `exp.Subquery` is handled differently than `exp.Table`. Consider
    creating a type for both.
    """
    table_nodes = []

    # TODO: type these
    from_clause = select_node.args.get("from")
    if from_clause:
        table_node = from_clause.this
        table_nodes.append(table_node)

    for join in select_node.args.get("joins", []):
        table_node = join.this
        table_nodes.append(table_node)

    return table_nodes


def get_ctes(tree: exp.Select) -> dict[str, exp.CTE]:
    ctes = {}
    # Full name of a cte is always == alias.
    if "with" in tree.args.keys():
        for cte in tree.args["with"].args["expressions"]:
            ctes[cte.alias] = cte
    return ctes


def _get_columns_from_cte(
    table_node: exp.Table, table_alias: str
) -> dict[str, str | None]:
    columns = {
        table_alias
        + column_node.this.this: getattr(
            getattr(column_node, "_type", None), "this", None
        )
        for column_node in bfs_limited(table_node.args["this"], 1, (exp.Column))
    }
    aliases = {
        table_alias + alias_node.alias: getattr(alias_node.args["alias"], "_type", None)
        for alias_node in bfs_limited(table_node.args["this"], 1, (exp.Alias))
    }
    return dict(columns, **aliases)


def _get_columns_from_subquery(
    subquery_node: exp.Subquery, table_alias: str
) -> dict[str, str | None]:
    next_select_node = subquery_node.find(exp.Select)
    columns = {
        table_alias
        + column_node.this.this: getattr(
            getattr(column_node, "_type", None), "this", None
        )
        for column_node in bfs_limited(next_select_node, 1, (exp.Column))
    }
    aliases = {
        table_alias + alias_node.alias: getattr(alias_node.args["alias"], "_type", None)
        for alias_node in bfs_limited(next_select_node, 1, (exp.Alias))
    }
    return dict(columns, **aliases)


def get_columns_from_table_node(
    table_node: exp.Table, tree: exp.Select
) -> dict[str, str | None]:
    ctes = get_ctes(tree)

    table_alias = ""
    # Note: CTE, Subquery and Table have an alias when available
    if table_node.alias:
        table_alias += table_node.alias + "."
    columns: dict[str, str | None] = {}
    if isinstance(table_node, exp.Table):
        full_table_name = get_full_table_name(table_node)
        # Note: if there is a cte with the same name, that is what we are
        # referring to. In that case I'll handle the CTE node.
        node = ctes.get(full_table_name, table_node)
        if isinstance(node, exp.CTE):
            columns.update(_get_columns_from_cte(node, table_alias))
        else:
            columns.update(
                get_columns_from_database_table(full_table_name, table_alias)
            )

    elif isinstance(table_node, exp.Subquery):
        columns.update(_get_columns_from_subquery(table_node, table_alias))

    return columns


def get_candidate_column_names(column_node: exp.Column, tree: exp.Select):
    """tree is the top level node"""
    parent_select_node = column_node.parent_select
    table_nodes = get_table_nodes_from_select(parent_select_node)
    # TODO: There are situations where we can't parse a from. In these cases,
    # we can return all columns available.
    column_names = []
    for table_node in table_nodes:
        column_names += list(get_columns_from_table_node(table_node, tree).keys())

    return list(set(column_names))


def get_full_table_name(table_node: exp.Table) -> str:
    full_table_name = ""
    if table_node.catalog:
        full_table_name += table_node.catalog + "."
    if table_node.db:
        full_table_name += table_node.db + "."
    full_table_name += table_node.this.name
    return full_table_name


def get_database_table_names(schema=SCHEMA) -> list[str]:
    tables = []
    for catalog in schema.keys():
        for db in schema[catalog].keys():
            for table in schema[catalog][db].keys():
                # TODO: consider allowing for non full tables or assert the
                # schema structure
                tables.append(catalog + "." + db + "." + table)
    return tables


def get_candidate_cte_names(tree) -> list[str]:
    candidate_cte_names = [node.alias for node in tree.find_all(exp.CTE)]
    return candidate_cte_names


def get_candidate_table_names(prefix: str, tree: exp.Select) -> list[str]:
    database_table_names: list[str] = get_database_table_names()
    cte_table_names: list[str] = get_candidate_cte_names(tree)
    all_table_names: list[str] = database_table_names + cte_table_names
    # Note: prefix will never contain a dot at the end. TODO test this, nvim
    # not letting me.
    candidate_table_names = [
        table_name
        for table_name in all_table_names
        if table_name.startswith(prefix) or table_name.startswith(prefix + ".")
    ]
    return candidate_table_names


def generate_diagnostics(tree: exp.Select, lines: list[str]):
    # TODO type
    diagnostics: list[Diagnostic] = []
    all_tables: set[str] = set(get_database_table_names()).union(
        set(get_candidate_cte_names(tree))
    )
    for node in tree.find_all(exp.Table):
        table_name: str = get_full_table_name(node)
        # TODO
        if table_name not in all_tables:
            if node.meta.get("start") or node.meta.get("end"):
                # TODO start and end update if pos
                start_offset: int = node.meta.get("start")
                end_offset: int = node.meta.get("end") + 1
                diagnostics.append(
                    Diagnostic(
                        range=Range(
                            start=offset_to_lsp_position(lines, start_offset),
                            end=offset_to_lsp_position(lines, end_offset),
                        ),
                        message=f"Unknown table: '{table_name}'",
                        severity=DiagnosticSeverity.Error,
                        source="sql_lsp",
                    )
                )

    for node in tree.find_all(exp.Column):
        parent_select = node.parent_select
        if parent_select:
            available_column_names = set(get_candidate_column_names(node, tree))
            # Get column name helper
            column_name = ""
            if node.table:
                column_name += node.table + "."
            column_name += node.this.this
            if column_name not in available_column_names:
                # TODO
                if node.meta.get("start") or node.meta.get("end"):
                    start_offset = node.meta.get("start")
                    end_offset = node.meta.get("end") + 1
                    diagnostics.append(
                        Diagnostic(
                            range=Range(
                                start=offset_to_lsp_position(lines, start_offset),
                                end=offset_to_lsp_position(lines, end_offset),
                            ),
                            message=f"Unknown column: '{column_name}'",
                            severity=DiagnosticSeverity.Error,
                            source="sql_lsp",
                        )
                    )

    for node in tree.find_all(exp.Column):
        if node._type.this == exp.DataType.Type.UNKNOWN:
            # TODO
            if node.meta.get("start") or node.meta.get("end"):
                start_offset = node.meta.get("start")
                end_offset = node.meta.get("end") + 1
                diagnostics.append(
                    Diagnostic(
                        range=Range(
                            start=offset_to_lsp_position(lines, start_offset),
                            end=offset_to_lsp_position(lines, end_offset),
                        ),
                        message=f"Unknown column type for '{node.this.this}'",
                        severity=DiagnosticSeverity.Warning,
                        source="sql_lsp",
                    )
                )

    return diagnostics


@server.feature("textDocument/didChange")
def did_change(params: DidChangeTextDocumentParams):
    doc = server.workspace.get_text_document(params.text_document.uri)
    lines = doc.source.splitlines()
    """
    tree = parse_one(doc.source, dialect=DIALECT, error_level=ErrorLevel.IGNORE)
    tree_has_types = False
    try:
        # Note: qual can only be used when all the columns are known
        qual = qualify(tree, dialect=DIALECT, schema=SCHEMA)
        tree = annotate_types(qual, dialect=DIALECT, schema=SCHEMA)
        tree_has_types = True
    except OptimizeError:
        pass
    diagnostics = generate_diagnostics(tree, lines, tree_has_types)
    server.publish_diagnostics(doc.uri, diagnostics)
    """
    tree_0 = parse_one(doc.source, dialect=DIALECT, error_level=ErrorLevel.IGNORE)
    # Note: qual can only be used when all the columns are known
    qual = qualify(
        tree_0, dialect=DIALECT, schema=SCHEMA, validate_qualify_columns=False
    )
    tree = annotate_types(qual, dialect=DIALECT, schema=SCHEMA)
    diagnostics = generate_diagnostics(tree, lines)
    server.publish_diagnostics(doc.uri, diagnostics)


@server.feature("textDocument/completion")
def completions(params: CompletionParams):
    doc = server.workspace.get_text_document(params.text_document.uri)
    lines = doc.source.splitlines()
    cursor_position = params.position.character
    line_position = params.position.line

    tree = parse_one(doc.source, dialect=DIALECT, error_level=ErrorLevel.IGNORE)
    cursor_offset = lsp_position_to_cursor_offset(lines, line_position, cursor_position)
    node = find_node_before_cursor(tree, cursor_offset)
    if node and isinstance(node, exp.Table):
        prefix = ""
        if node.catalog:
            prefix += node.catalog + "."
        if node.db:
            prefix += node.db + "."
        prefix += node.this.name
        candidate_table_names = get_candidate_table_names(prefix, tree)
        items = [
            CompletionItem(label=t, kind=CompletionItemKind.Class)
            for t in candidate_table_names
        ]
        return CompletionList(is_incomplete=False, items=items)

    if node and isinstance(node, exp.Column):
        candidate_column_names = get_candidate_column_names(node, tree)
        items = [
            CompletionItem(label=t, kind=CompletionItemKind.Class)
            for t in candidate_column_names
        ]
        return CompletionList(is_incomplete=False, items=items)

    return CompletionList(is_incomplete=False, items=[])


# Hover support
@server.feature("textDocument/hover")
def hover(params: HoverParams):
    doc = server.workspace.get_text_document(params.text_document.uri)
    lines = doc.source.splitlines()
    cursor_position = params.position.character
    line_position = params.position.line
    line = doc.source.splitlines()[line_position]

    tree = parse_one(doc.source, dialect=DIALECT, error_level=ErrorLevel.IGNORE)
    qual = qualify(tree, dialect="spark", schema=SCHEMA, validate_qualify_columns=False)
    tree = annotate_types(qual, dialect="spark", schema=SCHEMA)
    cursor_offset = lsp_position_to_cursor_offset(lines, line_position, cursor_position)
    node = find_node_before_cursor(tree, cursor_offset)
    if node and isinstance(node.parent, exp.Column):
        content = MarkupContent(
            kind=MarkupKind.Markdown,
            value=f"**Column**: `{node.parent.this.this}`\n\n**Type:**\n{node.parent._type.this.value}",
        )
        return Hover(contents=content)

    if node and isinstance(node.parent, exp.Table):
        columns = get_columns_from_table_node(node.parent, tree)
        column_list = "\n".join([f"- {col}: {_type}" for col, _type in columns.items()])
        content = MarkupContent(
            kind=MarkupKind.Markdown,
            value=f"**Columns:**\n{column_list}",
        )
        return Hover(contents=content)
    return Hover(contents="")


if __name__ == "__main__":
    # server.start_io()
    server.start_tcp("127.0.0.1", 2087)
