# Setup And Launch

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once. From
the repository root, synchronize the shared Python environment and open the
notebook:

```bash
uv sync --project experiments --locked
uv run --project experiments --locked jupyter lab experiments/cu-cu-hybrid-bonding/1d-stickiness-model/study.ipynb
```

uv creates `experiments/.venv/` and obtains Python 3.14 if needed. Initial setup
requires network access unless the required packages and interpreter are cached.
No manual environment activation or separate kernel installation is needed.

Select **Python 3 (ipykernel)**, then **Kernel > Restart Kernel and Run All Cells**.
The first cell reports the interpreter; it should be inside `experiments/.venv/`.
The committed notebook is output-free. Running all cells generates its controls,
plots, tables, and embedded animation; the model PDF is linked from the notebook.

For a remote/headless machine, use the same command with `--no-browser` and open
the server URL through your usual SSH port forwarding:

```bash
uv run --project experiments --locked jupyter lab --no-browser experiments/cu-cu-hybrid-bonding/1d-stickiness-model/study.ipynb
```

Keep the kernel running to use the controls. Saved notebook outputs can be viewed
without recomputing; trust this repository's notebook when prompted to enable
its embedded animation. Stop the server with `Ctrl-C` in its launch terminal.
