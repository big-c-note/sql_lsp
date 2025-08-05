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
from sqlglot.errors import ErrorLevel
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.qualify import qualify

# TODO tests
# TODO mention future of funcs and auto complete of types and things, maybe
# that is another thing


# TODO check how are debugs configed?
logging.basicConfig(filename="sql_lsp.log", filemode="w", level=logging.DEBUG)


server = LanguageServer("sql_lsp", "v0.1")

# Load metadata (your cached Databricks metadata)
# TODO: make this variable via config.
# TODO what other configs
# TODO readme on adding other dialects
# TODO get this a lighter weight pull in a dialect specific dir
# TODO option to config ther refresh time

# Globals
DIALECT = "spark"
CATALOG_SCHEMA_PATH: str = "~/.nvim-databricks/databricks_schema.json"


def _get_databricks_schema(
    catalog_schema_path: str | Path,
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
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
    # arbitrary metadata and chage the getter funcs to extend hover
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
                # TODO
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
                schema[catalog][db][table] = column_schema
    return schema


def get_schema(
    catalog_schema_path: str | Path, dialect: str = "databicks"
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    # TODO fix
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
def lsp_position_to_cursor_offset(lines: list[str], line: int, character: int) -> int:
    return sum(len(ln) + 1 for ln in lines[:line]) + character


def offset_to_lsp_position(lines: list[str], offset: int) -> Position:
    total = 0
    for i, line in enumerate(lines):
        if total + len(line) + 1 > offset:
            return Position(line=i, character=offset - total)
        total += len(line) + 1
    return Position(line=len(lines) - 1, character=len(lines[-1]))


def find_node_before_cursor(tree, cursor_offset: int):
    """
    Find the deepest node in the AST that starts before or contains the cursor position.
    """
    best_node = None
    best_span = -1

    for node in tree.walk():
        meta = node.meta or {}
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


# TODO test all these
# TODO does make sense to pass tablenode because then i can do it like the
# other
def get_candidate_table_names(prefix: str, tree) -> list[str]:
    # TODO change func name
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


def get_candidate_cte_names(tree) -> list[str]:
    candidate_cte_names = [node.alias for node in tree.find_all(exp.CTE)]
    return candidate_cte_names


# TODO get types as well. check if the args this is best way
def _get_column_names_from_cte(cte_node: exp.Table) -> list[str]:
    column_names = [
        column_node.this.this
        for column_node in bfs_limited(cte_node.args["this"], 1, (exp.Column))
    ]
    alias_names = [
        alias_node.alias
        for alias_node in bfs_limited(cte_node.args["this"], 1, (exp.Alias))
    ]
    return column_names + alias_names


# TODO may make sense to combine with above call them columns
def _get_column_names_from_subquery(subquery_node: exp.Subquery) -> list[str]:
    # TODO: combine with above.
    next_select_node = subquery_node.find(exp.Select)
    column_names = [
        column_node.this.this
        for column_node in bfs_limited(next_select_node, 1, (exp.Column))
    ]
    alias_names = [
        alias_node.alias for alias_node in bfs_limited(next_select_node, 1, (exp.Alias))
    ]
    return column_names + alias_names


# TODO should retun a column object then there can be a simpe function to get
# the column names
def get_candidate_column_names(column_node, select):
    parent_select_node = column_node.parent_select
    table_nodes = get_table_nodes_from_select(parent_select_node)
    # TODO: There are situations where we can't parse a from. In these cases,
    # we can return all columnd available.
    ctes = get_ctes(select)

    # TODO: also need aliases from the tables! only alias if table.
    column_names = []
    for table_node in table_nodes:
        column_names += get_columns_from_table_node(table_node, ctes)
    return list(set(column_names))


# TODO make columns not just name
# TODO type table node for cte as well
# TODO types
def get_columns_from_table_node(table_node, ctes):
    table_alias = ""
    # Note: both CTE, Subquery and Table have an alias when available
    column_names = []
    if table_node.alias:
        table_alias += table_node.alias + "."
    if isinstance(table_node, exp.Table):
        full_table_name = get_full_table_name(table_node)
        # Note: if there is a cte with the same name, that is what we are
        # referring to.
        table_node = ctes.get(full_table_name, table_node)
        if isinstance(table_node, exp.CTE):
            columns = _get_column_names_from_cte(table_node)
            column_names += [table_alias + col for col in columns]
        else:
            columns = get_columns_from_database_table(full_table_name).keys()
            column_names += [table_alias + col for col in columns]

    elif isinstance(table_node, exp.Subquery):
        columns = _get_column_names_from_subquery(table_node)
        column_names += [table_alias + col for col in columns]

    return column_names


# Note look up columns by table name. look up ctes by table name. not sure if i
# need to look up column by col name.
# TODO check if qualifymakes this any easier


# TODO could just check if the cte in by traversing the tree
def get_ctes(select_node):
    ctes = {}
    # Full name of a cte is always == alias.
    if "with" in select_node.args.keys():
        for cte in select_node.args["with"].args["expressions"]:
            ctes[cte.alias] = cte
    return ctes


def get_full_table_name(table_node: exp.Table) -> str:
    full_table_name = ""
    if table_node.catalog:
        full_table_name += table_node.catalog + "."
    if table_node.db:
        full_table_name += table_node.db + "."
    full_table_name += table_node.this.name
    return full_table_name


def get_table_nodes_from_select(select_node) -> list[exp.Table | exp.Subquery]:
    """
    Get all table definitions (FROM, JOIN, CTEs) from a Select node.
    Returns a dictionary mapping table aliases (or names) to their definitions (Select or Table nodes).
    """
    table_nodes = []

    # FROM clause

    from_clause = select_node.args.get("from")
    if from_clause:
        # Note: Could also be a Subquery/CTE, but CTEs are alsp exp.Table
        table_node = from_clause.this
        table_nodes.append(table_node)

    # JOIN clauses
    for join in select_node.args.get("joins", []):
        # Note: Could also be a Subquery/CTE, but CTEs are alsp exp.Table
        table_node = join.this
        table_nodes.append(table_node)

    return table_nodes


# TODO make this column nodes
def get_columns_from_select(select_node):
    """
    Get columns directly selected in a select node.
    Returns a list of column names.
    """
    columns = []
    for exp_node in select_node.expressions:
        if isinstance(exp_node, exp.Column):
            columns.append(exp_node.name)
    return columns


def get_database_table_names(schema=SCHEMA) -> list[str]:
    tables = []
    for catalog in schema.keys():
        for db in schema[catalog].keys():
            for table in schema[catalog][db].keys():
                # TODO: consider allowing for non full tables or assert the
                # schema structure
                tables.append(catalog + "." + db + "." + table)
    return tables


def get_columns_from_database_table(full_table_name: str, schema=SCHEMA):
    # TODO: consider allowing for non full tables or assert the
    # schema structure
    catalog, db, table = full_table_name.split(".")
    return schema[catalog][db][table]


# TODO change candidae table names
def generate_diagnostics(tree, lines):
    diagnostics = []
    all_tables = set(get_database_table_names())
    cte_names = set(get_candidate_cte_names(tree))

    for node in tree.find_all(exp.Table):
        table_name = get_full_table_name(node)
        if table_name not in all_tables and node.this.name not in cte_names:
            start_offset = node.meta.get("start")
            end_offset = node.meta.get("end") + 1
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
        if node._type.this == exp.DataType.Type.UNKNOWN:
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

    for node in tree.find_all(exp.Column):
        parent_select = node.parent_select
        if parent_select:
            available_columns = set(get_candidate_column_names(node, parent_select))
            # Get column name helper
            column_name = ""
            if node.table:
                column_name += node.table + "."
            column_name += node.this.this
            if column_name not in available_columns:
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
                        # TODO change source
                        source="sql_lsp",
                    )
                )

    return diagnostics


@server.feature("textDocument/didChange")
def did_change(params: DidChangeTextDocumentParams):
    doc = server.workspace.get_text_document(params.text_document.uri)
    lines = doc.source.splitlines()

    tree_0 = parse_one(doc.source, dialect=DIALECT, error_level=ErrorLevel.IGNORE)
    # TODO see if qualify should be used always
    qual = qualify(tree_0, dialect="spark", schema=SCHEMA)
    tree = annotate_types(qual, dialect="spark", schema=SCHEMA)

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
        # TODO pass node?
        candidate_table_names = get_candidate_table_names(prefix, tree)
        # TODO candidate table names is like helper for getting the names form
        # a list of table nodes? candidate has to do with autocomplete I think?
        items = [
            CompletionItem(label=t, kind=CompletionItemKind.Class)
            for t in candidate_table_names
        ]
        return CompletionList(is_incomplete=False, items=items)

    if node and isinstance(node, exp.Column):
        # Again helper over types?
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

    tree_0 = parse_one(doc.source, dialect=DIALECT, error_level=ErrorLevel.IGNORE)
    qual = qualify(tree_0, dialect="spark", schema=SCHEMA)
    tree = annotate_types(qual, dialect="spark", schema=SCHEMA)
    cursor_offset = lsp_position_to_cursor_offset(lines, line_position, cursor_position)
    node = find_node_before_cursor(tree, cursor_offset)
    if node and isinstance(node.parent, exp.Column):
        content = MarkupContent(
            kind=MarkupKind.Markdown,
            value=f"**Column**: `{node.parent.this.this}`\n\n**Type:**\n{node.parent._type.this.value}",
        )
        return Hover(contents=content)

    # TODO: add types.
    if node and isinstance(node.parent, exp.Table):
        ctes = get_ctes(tree)
        columns = get_columns_from_table_node(node.parent, ctes)
        column_list = "\n".join([f"- {col}" for col in columns])
        content = MarkupContent(
            kind=MarkupKind.Markdown,
            value=f"**Columns:**\n{column_list}",
        )
        return Hover(contents=content)
    return Hover(contents="")


if __name__ == "__main__":
    # server.start_io()
    server.start_tcp("127.0.0.1", 2087)
