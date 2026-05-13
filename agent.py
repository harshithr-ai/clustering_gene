"""
Text2SQL agent — Claude tool-use loop with sqlglot validation and retry.

Core flow:
  1. Caller submits a natural language question.
  2. We give Claude 3 tools: list_tables, get_table_schema, get_sample_values.
  3. Claude iterates (capped) until it emits a final SQL query.
  4. We validate via sqlglot (SELECT-only, single-statement, auto-LIMIT).
  5. We execute against SQLite. On failure, feed the error back and retry (capped).
  6. Yield typed events along the way so the UI can stream progress.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import pandas as pd
import sqlglot
from sqlglot import exp
from anthropic import Anthropic


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------
def list_tables(db_path: str) -> list[dict[str, Any]]:
    """Return a brief summary of every user table in the database."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        out = []
        for (name,) in rows:
            cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
            count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            out.append(
                {
                    "table": name,
                    "row_count": count,
                    "columns": [c[1] for c in cols],
                }
            )
        return out
    finally:
        conn.close()


def get_table_schema(db_path: str, table: str) -> dict[str, Any]:
    """Return columns, types, primary key, and foreign keys for one table."""
    conn = sqlite3.connect(db_path)
    try:
        # Validate the table exists (defends against the model hallucinating a name)
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            return {"error": f"Table '{table}' does not exist."}

        cols_raw = conn.execute(f"PRAGMA table_info({table})").fetchall()
        columns = [
            {
                "name": c[1],
                "type": c[2],
                "not_null": bool(c[3]),
                "default": c[4],
                "pk": bool(c[5]),
            }
            for c in cols_raw
        ]
        fks_raw = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        foreign_keys = [
            {"column": fk[3], "references_table": fk[2], "references_column": fk[4]}
            for fk in fks_raw
        ]
        return {
            "table": table,
            "columns": columns,
            "foreign_keys": foreign_keys,
        }
    finally:
        conn.close()


def get_sample_values(
    db_path: str, table: str, column: str, limit: int = 10
) -> dict[str, Any]:
    """Return up to `limit` distinct sample values for a column.

    Useful for the model to disambiguate enum-like columns (status, category, etc.).
    """
    # Only allow alphanumeric/underscore identifiers — last-line defense, since
    # we already control the tool-call surface.
    if not (table.replace("_", "").isalnum() and column.replace("_", "").isalnum()):
        return {"error": "Invalid identifier"}

    conn = sqlite3.connect(db_path)
    try:
        # Confirm column exists in table first
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in cols:
            return {
                "error": f"Column '{column}' not found in table '{table}'. "
                f"Available: {cols}"
            }
        rows = conn.execute(
            f"SELECT DISTINCT {column} FROM {table} "
            f"WHERE {column} IS NOT NULL LIMIT ?",
            (limit,),
        ).fetchall()
        return {
            "table": table,
            "column": column,
            "values": [r[0] for r in rows],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# sqlglot validator
# ---------------------------------------------------------------------------
class SqlValidationError(Exception):
    pass


def validate_sql(sql: str, dialect: str = "sqlite", default_limit: int = 1000) -> str:
    """Parse + check the SQL is a single, read-only SELECT. Auto-injects LIMIT.

    Returns the (possibly rewritten) SQL on success. Raises SqlValidationError
    with a human-readable message on failure — that message is fed back into
    the model on retry.
    """
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise SqlValidationError("Empty SQL")

    # Reject multi-statement strings outright.
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.ParseError as e:
        raise SqlValidationError(f"SQL did not parse: {e}") from e

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SqlValidationError(
            f"Exactly one statement is allowed; got {len(statements)}."
        )

    tree = statements[0]

    # Walk every node and reject anything that isn't a SELECT.
    forbidden = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
        exp.Alter, exp.TruncateTable, exp.Merge,
    )
    for node in tree.walk():
        if isinstance(node, forbidden):
            raise SqlValidationError(
                f"Only SELECT queries are allowed. Found: {type(node).__name__}."
            )

    # The top-level node must be SELECT (or a UNION/CTE wrapping one).
    if not isinstance(tree, (exp.Select, exp.Union, exp.With, exp.Subquery)):
        raise SqlValidationError(
            f"Top-level statement must be SELECT; got {type(tree).__name__}."
        )

    # Auto-inject LIMIT if the model omitted one and the query has no aggregation.
    select_node = tree.find(exp.Select) if not isinstance(tree, exp.Select) else tree
    if select_node is not None and select_node.args.get("limit") is None:
        has_agg = any(
            isinstance(n, (exp.AggFunc, exp.Group)) for n in select_node.walk()
        )
        if not has_agg:
            tree = tree.limit(default_limit)

    return tree.sql(dialect=dialect)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
def execute_sql(db_path: str, sql: str, timeout_s: float = 10.0) -> pd.DataFrame:
    """Run a validated SELECT and return a DataFrame. Read-only connection."""
    # SQLite read-only URI mode
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_s)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------
TOOL_DEFS = [
    {
        "name": "list_tables",
        "description": (
            "List every table in the database, with row counts and column names. "
            "Call this first to discover the schema."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_table_schema",
        "description": (
            "Return detailed column info (name, type, primary key, foreign keys) "
            "for a single table. Call this after list_tables to learn the structure "
            "of tables you plan to query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "The exact table name as returned by list_tables.",
                },
            },
            "required": ["table"],
        },
    },
    {
        "name": "get_sample_values",
        "description": (
            "Return up to N distinct sample values from a column. Use this to "
            "disambiguate enum-like fields (e.g. order status, category) before "
            "writing the WHERE clause."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table":  {"type": "string"},
                "column": {"type": "string"},
                "limit":  {"type": "integer", "default": 10},
            },
            "required": ["table", "column"],
        },
    },
]


SYSTEM_PROMPT = """\
You are an expert SQL analyst. Your job is to translate a user's natural-language \
question into a single, correct, read-only SQLite SELECT query that answers it.

You have tools to inspect the database schema. Use them — never guess at table or \
column names. A typical flow is:

  1. Call list_tables to see what's available.
  2. Call get_table_schema for each relevant table.
  3. Call get_sample_values when a column's possible values matter (e.g. status fields).
  4. Emit the final SQL.

Rules:
- SQLite dialect.
- SELECT only. No INSERT, UPDATE, DELETE, DDL, or multiple statements.
- Quote identifiers only if necessary.
- Prefer explicit JOINs over implicit comma-joins.
- When the question is ambiguous, make a reasonable assumption and state it briefly.

When you are ready, return ONLY the SQL inside a single ```sql code block, with no \
prose before or after. Do not call any more tools after emitting the SQL.
"""


# ---------------------------------------------------------------------------
# Agent state + events
# ---------------------------------------------------------------------------
@dataclass
class AgentRun:
    question: str
    db_path: str
    model: str
    final_sql: Optional[str] = None
    result: Optional[pd.DataFrame] = None
    status: str = "running"  # running | succeeded | failed
    failure_reason: Optional[str] = None
    retry_count: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    events: list[dict] = field(default_factory=list)


def _dispatch_tool(name: str, args: dict, db_path: str) -> Any:
    if name == "list_tables":
        return list_tables(db_path)
    if name == "get_table_schema":
        return get_table_schema(db_path, args["table"])
    if name == "get_sample_values":
        return get_sample_values(
            db_path, args["table"], args["column"], int(args.get("limit", 10))
        )
    return {"error": f"Unknown tool: {name}"}


def _extract_sql_from_text(text: str) -> Optional[str]:
    """Pull SQL out of a fenced ```sql block, or return text as-is if it looks like SQL."""
    if "```sql" in text:
        start = text.index("```sql") + len("```sql")
        end = text.find("```", start)
        if end == -1:
            return None
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.find("```", start)
        if end == -1:
            return None
        return text[start:end].strip()
    stripped = text.strip()
    upper = stripped.upper()
    if upper.startswith(("SELECT", "WITH")):
        return stripped
    return None


def run_agent(
    question: str,
    db_path: str,
    api_key: str,
    model: str = "claude-sonnet-4-5",
    max_retries: int = 3,
    max_tool_calls: int = 6,
    default_limit: int = 1000,
) -> Iterator[dict]:
    """Run the agent and yield events. Final event has type='complete' or 'error'.

    Each yielded dict has at minimum {'type': str, 'ts': float, ...}. The caller
    (Streamlit UI) consumes these and renders progress.
    """
    client = Anthropic(api_key=api_key)
    run = AgentRun(question=question, db_path=db_path, model=model)

    def emit(ev_type: str, **payload):
        ev = {"type": ev_type, "ts": time.time(), **payload}
        run.events.append(ev)
        return ev

    yield emit("run_started", question=question)

    # Outer retry loop. Each iteration is a complete tool-use conversation.
    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        run.retry_count = attempt
        if attempt > 0:
            yield emit("retry", attempt=attempt, reason=last_error)

        # Build the conversation. On retry, append the failure feedback so the
        # model can self-correct.
        user_content = question
        if last_error:
            user_content = (
                f"{question}\n\n"
                f"PREVIOUS ATTEMPT FAILED:\n{last_error}\n"
                "Please fix the SQL and try again."
            )
        messages: list[dict] = [{"role": "user", "content": user_content}]

        sql_candidate: Optional[str] = None
        tool_calls_used = 0

        # Inner tool-use loop.
        while True:
            yield emit("llm_call_start", attempt=attempt)
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFS,
                    messages=messages,
                )
            except Exception as e:
                run.status = "failed"
                run.failure_reason = f"LLM call failed: {e}"
                run.completed_at = time.time()
                yield emit("error", message=run.failure_reason)
                return

            run.tokens_in += response.usage.input_tokens
            run.tokens_out += response.usage.output_tokens
            yield emit(
                "llm_call_end",
                stop_reason=response.stop_reason,
                tokens_in=response.usage.input_tokens,
                tokens_out=response.usage.output_tokens,
            )

            # Surface any text the model emitted this turn.
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    yield emit("llm_text", text=block.text)

            # If the model wants to call tools, run them and loop.
            if response.stop_reason == "tool_use":
                # Append the assistant turn verbatim so the next call sees it.
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    tool_calls_used += 1
                    if tool_calls_used > max_tool_calls:
                        # Force-stop: tell the model it's out of budget.
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(
                                    {"error": "Tool call budget exceeded. "
                                     "Emit the SQL now based on what you know."}
                                ),
                            }
                        )
                        continue

                    yield emit(
                        "tool_call",
                        name=block.name,
                        input=block.input,
                        call_index=tool_calls_used,
                    )
                    try:
                        result = _dispatch_tool(block.name, block.input, db_path)
                    except Exception as e:
                        result = {"error": f"Tool execution failed: {e}"}

                    run.tool_calls.append(
                        {"name": block.name, "input": block.input, "output": result}
                    )
                    yield emit(
                        "tool_result",
                        name=block.name,
                        output=result,
                        call_index=tool_calls_used,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        }
                    )

                messages.append({"role": "user", "content": tool_results})
                continue  # loop back for another LLM call

            # Otherwise: model is done. Find the SQL.
            full_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            sql_candidate = _extract_sql_from_text(full_text)
            break

        if not sql_candidate:
            last_error = (
                "No SQL was emitted. Please respond with a single ```sql ... ``` "
                "code block containing the query."
            )
            continue

        yield emit("sql_drafted", sql=sql_candidate)

        # Validate
        try:
            safe_sql = validate_sql(sql_candidate, default_limit=default_limit)
        except SqlValidationError as e:
            last_error = f"Validation error: {e}"
            yield emit("validation_failed", error=str(e), sql=sql_candidate)
            continue

        if safe_sql != sql_candidate:
            yield emit("sql_rewritten", original=sql_candidate, rewritten=safe_sql)

        # Execute
        yield emit("execution_started", sql=safe_sql)
        try:
            df = execute_sql(db_path, safe_sql)
        except Exception as e:
            last_error = f"Execution error: {e}"
            yield emit("execution_failed", error=str(e), sql=safe_sql)
            continue

        run.final_sql = safe_sql
        run.result = df
        run.status = "succeeded"
        run.completed_at = time.time()
        yield emit(
            "complete",
            sql=safe_sql,
            row_count=len(df),
            columns=list(df.columns),
            latency_ms=int((run.completed_at - run.started_at) * 1000),
            tokens_in=run.tokens_in,
            tokens_out=run.tokens_out,
            retry_count=run.retry_count,
        )
        # Stash the run so the caller can grab `result` at the end.
        yield {"type": "_run_object", "run": run}
        return

    # Exhausted retries
    run.status = "failed"
    run.failure_reason = last_error or "Unknown failure"
    run.completed_at = time.time()
    yield emit(
        "error",
        message=f"Failed after {max_retries} retries: {run.failure_reason}",
    )
    yield {"type": "_run_object", "run": run}
