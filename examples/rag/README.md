# rag examples — authorization-filtered retrieval

Every search reverse-queries ADS (CAQ) for the condition under which the subject may read, then
pushes that metadata filter into the vector store, so only permitted documents come back.

Each script assumes **ADS is already running** with `domain_rag_retrieval.yaml` loaded (see the
[Running ADS](../../docs/running-ads.md)) and reads `PDP_BASE_URL`, `LLM_MODEL`, and your provider key
from the environment / `.env`.

| Script | What it shows |
|---|---|
| `agentic_rag.py` | Retrieval-as-a-tool inside a `create_agent` agent — alice (engineering) and bob (finance) each retrieve only their permitted docs. |
| `deepagents_rag.py` | The same, driven by a Deep Agent. |
| `static_rag.py` | A non-agent pipeline: identity supplied via `authz_context(user)` around `similarity_search`. |

```bash
python examples/rag/agentic_rag.py
```

See the [rag module README](../../axiomatics_ai/rag/README.md) for the API.
