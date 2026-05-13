# Text2SQL — Streamlit Prototype

A working prototype of the agentic Text-to-SQL system from the design doc, built as a single Streamlit app so you can poke at it without spinning up the full FastAPI/Next.js stack.

**What it does:** you ask a question in English → Claude inspects the schema via tool use → emits SQL → we validate it (sqlglot, SELECT-only, auto-LIMIT) → execute it against a local SQLite database → show you the results, the trace, and the SQL.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Anthropic API key
#    Open .env and replace `your-anthropic-api-key-here` with a real key
#    (Get one at https://console.anthropic.com/settings/keys)

# 3. Seed the demo database (one-time; the UI will also offer this button)
python seed_db.py

# 4. Run the app
streamlit run app.py
```

Then open http://localhost:8501.

## What's in the box

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — input, agent trace, results table, history |
| `agent.py` | Schema-introspection tools, sqlglot validator, Claude tool-use loop with retries |
| `seed_db.py` | Creates `demo.db` — a small sales schema (~25 customers, 15 products, 300 orders) |
| `.env` | Configuration — API key goes here |
| `requirements.txt` | Python deps |

## Demo schema

The seeded SQLite database has 5 tables:

- `customers` (id, name, email, country, signup_date)
- `employees` (id, name, role, region, hired_on)
- `products` (id, name, category, unit_price, in_stock)
- `orders` (id, customer_id, employee_id, order_date, status, total)
- `order_items` (id, order_id, product_id, quantity, unit_price)

Try the sample questions in the UI to see the agent in action.

## How the agent works (short version)

1. **System prompt** tells Claude it's a SQL analyst with three tools.
2. **Tools exposed via Claude's tool-use API:**
   - `list_tables()` — every table with column names and row counts
   - `get_table_schema(table)` — columns, types, PKs, FKs
   - `get_sample_values(table, column, limit)` — distinct values; great for figuring out what `status` means
3. Claude iterates (capped at 6 tool calls) until it emits a SQL query in a fenced code block.
4. **`sqlglot` validates:** parse must succeed, single statement only, no INSERT/UPDATE/DELETE/DDL, and a `LIMIT 1000` is auto-injected if the query has no aggregation and no LIMIT.
5. **Executor** opens SQLite in **read-only URI mode** and runs the query.
6. **On validation or execution failure**, the error message is fed back into the next attempt; up to 3 retries (configurable in `.env`).

Every step streams an event into the UI so you can watch the agent think.

## Mapping back to the design doc

This is the **Phase 1–4 happy path collapsed into ~600 lines**. Differences from the full spec, on purpose:

| Spec | This prototype |
|---|---|
| FastAPI + Next.js + Postgres + Redis + Qdrant | Single Streamlit process + SQLite |
| LangGraph state machine | Plain Python loop (same node semantics, simpler) |
| Vector store RAG over schema | Schema is small enough that we hand the whole table list to Claude as a tool |
| Multi-tenant auth, JWT, encrypted DSNs | None — this is a single-user demo |
| Run history in Postgres | In-memory `st.session_state` |
| SSE streaming over HTTP | Streamlit's natural rerender-on-yield |

The agent contract (tools, retry loop, validator rules, event types) is the same as in the design doc, so when you graduate to the full FastAPI stack the `agent.py` module ports over almost as-is.

## Configuration knobs (`.env`)

| Var | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Set this before launching. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Any tool-use-capable Claude model works |
| `DATABASE_PATH` | `demo.db` | Local SQLite file |
| `MAX_RETRIES` | `3` | Hard cap on agent retries |
| `MAX_TOOL_CALLS` | `6` | Hard cap on tool calls per attempt |
| `DEFAULT_ROW_LIMIT` | `1000` | Auto-injected LIMIT |

## Troubleshooting

- **"Anthropic API key is missing"** → open `.env`, paste your key.
- **`ModuleNotFoundError`** → `pip install -r requirements.txt`. Python 3.10+.
- **Empty results / wrong table** → use the sidebar to verify the schema; click "(Re)seed demo database" to reset.
- **Validation failures** → the error appears inline in the trace and is fed back to the model for the next retry.
