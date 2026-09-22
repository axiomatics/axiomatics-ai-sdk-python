# tool_decorator examples — per-tool post-invocation with `@protected_tool`

Assumes **ADS is already running** with `domain_tool_decorator_masking.yaml` loaded (see the
[Running ADS](../../docs/running-ads.md)); reads `PDP_BASE_URL`, `LLM_MODEL`, and your provider key from
the environment / `.env`.

| Script | What it shows |
|---|---|
| `masking_server.py` | A FastMCP server whose tools are decorated with `@protected_tool(post_processor=...)` — `ColumnMaskStrategy` masks PII fields on Deny; a custom strategy redacts free-text emails. |
| `masking_client.py` | A LangChain agent calling the masking server — alice sees full data, bob gets masked/redacted results. |

```bash
python examples/tool_decorator/masking_server.py    # terminal 1
python examples/tool_decorator/masking_client.py     # terminal 2
```

See the [tool_decorator module README](../../axiomatics_ai/tool_decorator/README.md) for the API,
including how to write a custom post-invocation strategy.
