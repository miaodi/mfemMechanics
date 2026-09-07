# Experiments

Lightweight Python scripts and notes for brainstorming and idea verification.
Contents are exploratory, not supported library capabilities, and are not part
of the CMake build or CTest suite. Record assumptions and distinguish tentative
ideas from verified conclusions.

## Python Environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once.
Run these commands from the repository root:

```bash
# Create or synchronize the shared environment.
uv sync --project experiments --locked

# Check the environment, creating it automatically if needed.
uv run --project experiments --locked python -c "import sys; print(sys.executable)"

# Add a dependency when an experiment needs it (example only).
uv add --project experiments sympy
```

Run a future script with `uv run --project experiments path/to/script.py`.
No manual environment activation is required. uv manages `experiments/.venv/`
and can download Python 3.14 if needed. Initial setup requires network access
unless the necessary artifacts are cached. The shared dependencies include the
numerical and Jupyter tools used by the sticky-rod notebook.

Commit `pyproject.toml`, `uv.lock`, and `.python-version` together when changing
the environment. The lockfile records dependency resolution; `--locked` checks
that it agrees with the project configuration rather than updating it.

## Organization

- Keep scripts, notes, and small reproducible inputs under a topic directory.
- Document each script's question, assumptions, run command, and conclusions.
- Notebooks may retain inline plots, tables, and animations as research notes.
  Virtual environments and Python caches are ignored. If a standalone experiment
  needs generated files, keep them in its ignored `output/` directory.
- Promote reusable implementations to the library, regression checks to
  `tests/`, supported examples to `examples/`, and established explanations to
  `docs/` only after appropriate verification.

## Projects

- [ ] [Cu-Cu hybrid bonding](cu-cu-hybrid-bonding/README.md)
