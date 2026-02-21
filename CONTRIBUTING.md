## Developing

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/dek3rr/thrml-boost.git
cd thrml-boost
```

Then install with development dependencies:

```bash
# Using pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[development,testing,examples]"

# Or using uv
uv venv venv
source venv/bin/activate
uv pip install -e ".[development,testing,examples]"
```

Install pre-commit hooks:

```bash
pre-commit install
```

The pre-commit hooks will automatically run `ruff` (formatting + linting) and `pyright` (type checking) on every commit.

To skip hooks for a WIP commit:

```bash
git commit --no-verify -m "wip: your message"
```

## Running tests

```bash
# All fast tests
pytest -v -m "not slow" tests/

# Everything including the slow MNIST test
pytest -v tests/
```

## Branching

- `main` — stable, always passing CI
- `dev` — active development, merge into main when ready
- Feature branches off `dev`: `feat/your-feature`, `fix/your-fix`

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new sampling strategy
fix: correct ragged block handling in hinton_init
perf: thread global state through scan carry
test: add vmap correctness tests for parallel tempering
docs: update README installation instructions
chore: bump ruff to v0.11.0
```
