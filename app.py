"""
Text2SQL — Streamlit prototype.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import list_tables, run_agent
from seed_db import seed as seed_db


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Text2SQL — Claude Agent",
    page_icon="🧮",
    layout="wide",
)

API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5").strip()
DB_PATH = os.getenv("DATABASE_PATH", "demo.db").strip()
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "6"))
DEFAULT_ROW_LIMIT = int(os.getenv("DEFAULT_ROW_LIMIT", "1000"))


def _api_key_looks_real(k: str) -> bool:
    return bool(k) and k != "your-anthropic-api-key-here" and len(k) > 20


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of completed run dicts
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------------------------------------------------------------------------
# Sidebar — config + schema browser
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    if _api_key_looks_real(API_KEY):
        st.success(f"API key loaded ({len(API_KEY)} chars)")
    else:
        st.error("Anthropic API key not set. Edit `.env` and set `ANTHROPIC_API_KEY`.")

    st.caption(f"**Model:** `{MODEL}`")
    st.caption(f"**Database:** `{DB_PATH}`")
    st.caption(f"**Retry budget:** {MAX_RETRIES}  |  **Tool budget:** {MAX_TOOL_CALLS}")

    st.divider()
    st.markdown("### 🗄️ Database")

    db_exists = Path(DB_PATH).exists()
    if not db_exists:
        st.warning("Demo database not found.")
    else:
        st.success("Demo database is ready.")

    if st.button("🔄 (Re)seed demo database", use_container_width=True):
        with st.spinner("Seeding..."):
            seed_db()
        st.rerun()

    if db_exists:
        st.divider()
        st.markdown("### 📋 Schema")
        try:
            tables = list_tables(DB_PATH)
            for t in tables:
                with st.expander(f"`{t['table']}` — {t['row_count']} rows"):
                    st.write(", ".join(f"`{c}`" for c in t["columns"]))
        except Exception as e:
            st.error(f"Could not read schema: {e}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧮 Text2SQL")
st.caption(
    "Ask a question in plain English. The agent inspects the schema, drafts SQL, "
    "validates it, runs it, and shows you the result."
)

# Hard gate: nothing useful happens without an API key and a DB.
if not _api_key_looks_real(API_KEY):
    st.error(
        "**Anthropic API key is missing.**  \n"
        "Open the `.env` file in the project root and set `ANTHROPIC_API_KEY`. "
        "Get a key at https://console.anthropic.com/settings/keys"
    )
    st.stop()

if not Path(DB_PATH).exists():
    st.warning("Demo database doesn't exist yet. Click **🔄 (Re)seed demo database** in the sidebar.")
    st.stop()


# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
SAMPLES = [
    "How many orders were placed in 2024?",
    "Top 5 customers by total spend.",
    "Which product category has the highest revenue?",
    "Show monthly revenue for delivered orders in 2024.",
    "Which employees made the most sales in Europe?",
    "List all cancelled orders with their total value.",
]

st.markdown("#### Ask a question")
cols = st.columns([5, 1])
question = cols[0].text_input(
    "question",
    label_visibility="collapsed",
    placeholder="e.g. Top 5 customers by total spend",
    value=st.session_state.pending_question or "",
    key="question_input",
)
ask_clicked = cols[1].button("Ask", type="primary", use_container_width=True)

with st.expander("💡 Try a sample"):
    sample_cols = st.columns(2)
    for i, s in enumerate(SAMPLES):
        if sample_cols[i % 2].button(s, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = s
            st.rerun()

# If a sample button was clicked previously, clear the pending flag so the
# input reflects it but doesn't keep firing.
if st.session_state.pending_question and st.session_state.pending_question == question:
    st.session_state.pending_question = None


# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------
def _render_event(ev: dict, container) -> None:
    """Render one streaming event into the trace container."""
    t = ev["type"]
    if t == "run_started":
        container.markdown(f"🟢 **Run started** — `{ev['question']}`")
    elif t == "retry":
        container.markdown(
            f"🔁 **Retry {ev['attempt']}** — previous failure: `{ev['reason']}`"
        )
    elif t == "llm_call_start":
        container.markdown(f"🧠 Calling Claude (attempt {ev['attempt']})…")
    elif t == "llm_call_end":
        container.caption(
            f"   ↳ stop_reason=`{ev['stop_reason']}` · "
            f"in={ev['tokens_in']} · out={ev['tokens_out']} tokens"
        )
    elif t == "llm_text":
        # Only show short reasoning blurbs to avoid clutter
        snippet = ev["text"].strip()
        if snippet and len(snippet) < 600:
            container.markdown(f"> {snippet}")
    elif t == "tool_call":
        container.markdown(
            f"🔧 **Tool call #{ev['call_index']}** `{ev['name']}` → "
            f"`{json.dumps(ev['input'])}`"
        )
    elif t == "tool_result":
        out = ev["output"]
        # Compact rendering: small dicts/lists inline, big ones in JSON expander
        out_str = json.dumps(out, default=str)
        if len(out_str) < 200:
            container.caption(f"   ↳ {out_str}")
        else:
            with container.expander(f"   ↳ result ({len(out_str)} chars)"):
                st.json(out)
    elif t == "sql_drafted":
        container.markdown("📝 **SQL drafted**")
        container.code(ev["sql"], language="sql")
    elif t == "sql_rewritten":
        container.caption("   ↳ auto-injected `LIMIT` for safety")
    elif t == "validation_failed":
        container.error(f"❌ Validation failed: {ev['error']}")
    elif t == "execution_started":
        container.markdown("⚡ Executing…")
    elif t == "execution_failed":
        container.error(f"❌ Execution failed: {ev['error']}")
    elif t == "complete":
        container.success(
            f"✅ **Complete** — {ev['row_count']} rows in "
            f"{ev['latency_ms']} ms · {ev['retry_count']} retries · "
            f"{ev['tokens_in']}+{ev['tokens_out']} tokens"
        )
    elif t == "error":
        container.error(f"💥 {ev['message']}")


if ask_clicked and question.strip():
    st.divider()
    st.markdown("### Agent trace")
    trace_box = st.container()

    final_run = None
    try:
        for ev in run_agent(
            question=question.strip(),
            db_path=DB_PATH,
            api_key=API_KEY,
            model=MODEL,
            max_retries=MAX_RETRIES,
            max_tool_calls=MAX_TOOL_CALLS,
            default_limit=DEFAULT_ROW_LIMIT,
        ):
            if ev["type"] == "_run_object":
                final_run = ev["run"]
                continue
            _render_event(ev, trace_box)
    except Exception as e:
        st.error(f"Agent crashed: {e}")
        st.exception(e)

    # Final result
    if final_run is not None:
        st.divider()
        if final_run.status == "succeeded" and final_run.result is not None:
            st.markdown("### 📊 Result")
            tab1, tab2, tab3 = st.tabs(["Table", "SQL", "Run summary"])
            with tab1:
                st.dataframe(
                    final_run.result, use_container_width=True, height=420
                )
                csv = final_run.result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name="result.csv",
                    mime="text/csv",
                )
            with tab2:
                st.code(final_run.final_sql, language="sql")
            with tab3:
                st.json(
                    {
                        "question": final_run.question,
                        "model": final_run.model,
                        "status": final_run.status,
                        "retries": final_run.retry_count,
                        "tool_calls": len(final_run.tool_calls),
                        "tokens_in": final_run.tokens_in,
                        "tokens_out": final_run.tokens_out,
                        "latency_ms": int(
                            (final_run.completed_at - final_run.started_at) * 1000
                        ),
                        "row_count": len(final_run.result),
                    }
                )

            # Persist into history
            st.session_state.history.append(
                {
                    "question": final_run.question,
                    "sql": final_run.final_sql,
                    "row_count": len(final_run.result),
                    "preview": final_run.result.head(50),
                    "retries": final_run.retry_count,
                    "tokens_in": final_run.tokens_in,
                    "tokens_out": final_run.tokens_out,
                }
            )
        else:
            st.error(
                f"Run failed after {final_run.retry_count} retries: "
                f"{final_run.failure_reason}"
            )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    st.markdown("### 🕘 This session's runs")
    for i, h in enumerate(reversed(st.session_state.history)):
        with st.expander(
            f"**{h['question']}** — {h['row_count']} rows, "
            f"{h['retries']} retries"
        ):
            st.code(h["sql"], language="sql")
            st.dataframe(h["preview"], use_container_width=True)
