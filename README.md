# parabolic_mirror

Wave propagation experiments for reflection off an off-axis parabolic (OAP)
mirror, extracted from `SLM_code/development/parmir.py` into its own project.

## Structure

```text
parabolic_mirror/
|-- pyproject.toml
|-- README.md
|-- .gitignore
|-- parmir.py            # experiment script, run directly
`-- src/
    `-- optics_lib/
        |-- __init__.py   # re-exports everything from core.py
        `-- core.py       # optical_field class, propagation methods, etc.
```

`optics_lib` is installed into `.venv` via `uv sync`, so any script at the
project root can do:

```python
import optics_lib as opt
```

without `sys.path` hacks.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python parmir.py
```

## Daily work

Add another script at the project root; it can import `optics_lib` the same
way. Add dependencies with:

```bash
uv add <package>
```

then re-run `uv sync`.
