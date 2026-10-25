# Contributing

Thanks for helping! Quick start:

```bash
git clone https://github.com/matheusbuniotto/pydantic-ai-durable
cd pydantic-ai-durable
uv sync --extra dev
uv run pytest                                  # tests
uv run --with ruff ruff check . && uv run --with ruff ruff format .   # lint + format
uv run python examples/crash_recovery.py       # try the demo
```

- Open an issue before large changes.
- Add or update tests for behaviour changes; CI runs Python 3.10–3.13.
- Keep functions small. Complexity limits live in `pyproject.toml` and are meant to ratchet down, never up.
- Use clear commit messages; PRs are squash-merged.
