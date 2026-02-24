**Agents Guide**

- Purpose: onboard autonomous coding agents (linters, CI bots, human-like assistants) to the SwimHPE repo; give actionable commands, conventions, and pointers so agents can make safe, consistent edits.
- Keep changes minimal and well-tested; prefer non-destructive edits and open a PR for larger changes.

Project quick pointers
- Repo goal: fine-tune a YOLO-Pose model for swimmer keypoint detection (see `CLAUDE.md`).
- Useful code locations: `train.py`, `val.py`, `predict_visualize.py`, `webcam_demo.py`, `GUI.py` (experimental — do not modify unless explicitly requested), and data utilities under `data_processing/`.
- Important dataset quirks and canonical mapping: `data_processing/keypoint_mapping.py` (7-column lower-body shift, Y-axis flip) and `data_processing/format_conversion.py`.

Build / environment / run commands
- Create and activate a virtualenv and install deps:
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
- Quick model commands (same as repo top-level scripts):
```bash
python train.py              # train / fine-tune on SwimXYZ
python val.py                # validate model on swim dataset
python predict_visualize.py  # run on a single image
python webcam_demo.py        # real-time webcam inference
python GUI.py                # launch PyQt6 GUI
```
- Run a one-off script (example):
```bash
python data_processing/format_conversion.py --help
```

Testing
- This repo does not include a full pytest config by default, but tests live under `GUI/tests.py` and any new tests should use `pytest`.
- Install pytest: `pip install pytest` (or add to `requirements.txt`).
- Run the full test suite:
```bash
pytest -q
```
- Run a single test file:
```bash
pytest GUI/tests.py -q
```
- Run a single test by node/class/function:
```bash
pytest path/to/test_file.py::TestClass::test_method -q
pytest -k "substring" -q    # run tests matching substring
```

Lint / format / static analysis
- Recommended tools: `black`, `isort`, `flake8`, `mypy` (optional), `ruff` (optional for speed).
- Quick install:
```bash
pip install black isort flake8 mypy ruff
```
- Format everything (stable baseline):
```bash
black .
isort .
```
- Lint (syntax / style):
```bash
flake8 .
ruff .
```
- Type-check (incremental):
```bash
mypy --namespace-packages --ignore-missing-imports .
```

Code style guidelines (for autonomous agents)
- Language and style surface
  - Primary language: Python 3.10+ (type hints encouraged). Keep code simple and explicit.
  - Use small, well-named functions (single responsibility). Break large functions (>200 LOC) into smaller helpers.

- Imports
  - Order imports as: standard library, third-party, local. Keep one empty line between groups.
  - Use absolute imports for package modules (e.g. `from data_processing.keypoint_mapping import SWIMXYZ_COL_TO_YOLO_IDX`).
  - Avoid wildcard imports (`from module import *`).

- Formatting
  - Use `black` defaults for formatting; do not fight `black` line-wrapping.
  - Run `isort` to keep imports deterministic. Use `isort` profiles compatible with `black`.
  - Keep max line length 88 (black default) unless a clear justification exists.

- Types and docstrings
  - Add type hints for public functions and complex internal functions. Use `-> None` when functions return nothing.
  - Prefer Google or NumPy style docstrings. Include Args, Returns, and Raises sections for non-trivial functions.
  - Small throwaway scripts (one-off CLIs) may omit full typing, but core modules should be typed incrementally.

- Naming conventions
  - Modules / packages: snake_case (e.g. `data_processing`).
  - Functions and variables: snake_case. Keep names descriptive (e.g. `calculate_bounding_box`).
  - Classes: PascalCase. Limit public API surface.
  - Constants: UPPER_SNAKE (e.g. `KEYPOINT_VISIBLE`).
  - Avoid single-letter names except for indices/iterators in small scopes.

- Error handling and logging
  - Prefer `logging` over `print` for library code. Use module-level logger: `logger = logging.getLogger(__name__)`.
  - Catch specific exceptions; avoid broad `except Exception:` unless wrapping for error-reporting and re-raising.
  - When re-raising, preserve context with `raise` or `raise from`.
  - Return `None` only when documented; prefer raising `ValueError`/`RuntimeError` for incorrect inputs.

- Tests and safety
  - New features must include tests. Keep tests small and deterministic.
  - Avoid committing large data files; use fixtures and mocks for I/O heavy operations.
  - For GPU/ML tests, mark heavy tests (e.g. `@pytest.mark.slow`) and exclude from default CI.

- Data handling (project-specific)
  - Refer to `data_processing/format_conversion.py` for canonical conversion flow (Y flip, visibility flags, bounding box calculation).
  - Always validate that keypoint indices map correctly using `data_processing/keypoint_mapping.py` when modifying conversion logic.
  - When writing converters, ensure coordinate normalization and padding use explicit image width/height parameters (no hidden globals).

Git / commit guidance for agents
- Do not create commits unless explicitly instructed. When creating commits follow the repository's conventional style: short imperative subject (one line) and a 1–2 sentence body describing the "why".
- Avoid changing unrelated files. If you must, explain reason in the commit message.

Cursor / Copilot rules
- Cursor rules: no `.cursor/` or `.cursorrules` were found in the repository — none to include.
- Copilot rules: no `.github/copilot-instructions.md` was found in the repository — nothing to include.

Useful file references
- Dataset & conversion: `data_processing/format_conversion.py`, `data_processing/keypoint_mapping.py`.
- Training & evaluation commands: top-level `train.py`, `val.py`, `predict_visualize.py`, `webcam_demo.py`, `GUI.py`.
- Requirements: `requirements.txt`.

Safety and change policy for agents
- Non-destructive first: prefer edits that add tests, docs, or small refactors.
- For breaking or large changes: open a draft PR and request human review.
- Never commit secrets (API keys, credentials). If you find secrets, stop and notify the human operator.

- GUI experimental policy: the GUI is an experimental feature. Agents MUST NOT modify or add changes under the `GUI/` folder or `GUI.py` unless the user explicitly asks for GUI work. Treat any GUI-related edits as optional, potentially fragile, and requiring human approval before committing.

Next steps agents may take (suggestions)
1. Add `pyproject.toml` with `black`, `isort`, and `mypy` config (if CI expects it).
2. Add a small `tox` or `nox` configuration to run lint/test matrix.
3. Add CI workflow that runs `black --check`, `isort --check-only`, `flake8` and `pytest -q`.

If you are blocked: ask exactly one targeted question and propose a default. Keep changes small and reversible.

Neck joint: how to undo
- To undo adding the Neck joint: revert the `Neck` entry in `data_processing/keypoint_mapping.py` (remove it from `COCO_KP_NAMES` and `SWIMXYZ_TO_COCO_NAME`), change any `39`-length YOLO keypoint arrays in `data_processing/format_conversion.py` back to `36` (12 keypoints × 3), and restore any keypoint-count constants in `train.py`/`val.py`/visualization scripts to their previous values.
