\# Temporal Agent Memory System



A production-oriented memory system for an LLM agent that stores, retrieves,

and reasons over user facts with full temporal awareness — tracking not just

\*what\* is true, but \*when\* it was true.



\## Status



Phases 1–5 complete. Core memory storage, versioning, and temporal query

engine are implemented and tested (75 tests passing).



\## Architecture



\- \*\*`app/memory/models.py`\*\* — `Memory` Pydantic model: subject/attribute/value,

&#x20; temporal validity (`valid\_from`/`valid\_to`, half-open interval), provenance,

&#x20; confidence, and lifecycle status. Fully validated (timezone-aware datetimes

&#x20; normalized to UTC, interval consistency, status consistency).

\- \*\*`app/storage/database.py`\*\* — SQLite schema and connection handling,

&#x20; with CHECK constraints mirroring the model's invariants.

\- \*\*`app/storage/memory\_repository.py`\*\* — CRUD + temporal queries against

&#x20; SQLite: `insert`, `get`, `update`, `list\_by\_key`, `get\_at\_time`. Supports

&#x20; shared connections for multi-statement transactions.

\- \*\*`app/memory/manager.py`\*\* — the governance layer. LLMs (or any caller)

&#x20; never touch SQL directly.

&#x20; - `store()` — creates a new memory, atomically checking for temporal

&#x20;   overlap with existing memories under the same key.

&#x20; - `supersede()` — atomically closes the currently-open memory for a key

&#x20;   and opens a new one, preserving full history (no data is ever deleted).

&#x20; - `get\_at\_time()` / `get\_current()` / `get\_range()` / `get\_transition()` —

&#x20;   temporal query types (point-in-time, current, range, and transition

&#x20;   detection).



\## Key design decisions



\- \*\*Half-open intervals\*\* `\[valid\_from, valid\_to)` — a memory's `valid\_to`

&#x20; is exclusive, so adjacent memories (e.g. Node.js ending exactly when Go

&#x20; begins) never overlap.

\- \*\*Bitemporal-lite\*\*: `valid\_from`/`valid\_to` capture when something was

&#x20; true in the world; `recorded\_at` captures when the system learned it.

\- \*\*Atomicity via `BEGIN IMMEDIATE`\*\*: both `store()` and `supersede()`

&#x20; wrap their read-then-write logic in a single SQLite transaction, verified

&#x20; by deliberately breaking the commit ordering and confirming the test

&#x20; suite catches it.

\- \*\*Frozen models\*\*: a `Memory` is never mutated in place; changes create

&#x20; new versions, preserving full audit history.



\## Running tests



```powershell

pytest -q

```



\## Roadmap



\- Phase 6: memory consolidation / discard

\- Phase 7: embeddings

\- Phase 8: FAISS semantic retrieval

\- Phase 9: hybrid temporal + semantic retrieval

\- Phase 10+: LLM extraction, LangGraph orchestration, agent reasoning, FastAPI, evaluation

