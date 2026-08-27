# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt      # setup
python run_pipeline.py               # generate data -> EDA -> train (~10s, must run before anything else)
streamlit run app.py                 # web UI at http://localhost:8501
python demo_tool.py                  # score five fixed profiles from the CLI
```

Individual pipeline stages, if you only need one:

```bash
python src/generate_data.py    # writes data/credit_risk_synthetic.csv
python src/eda.py              # writes outputs/figures/*.png + outputs/eda_summary.md
python src/train_model.py      # writes models/* + outputs/evaluation_report.md
```

There is no test suite, linter, or build step configured. `demo_tool.py` is the closest thing
to a regression check: it exercises the approve path, the reject path, and the validation
guard in one run.

On this Windows setup neither `streamlit` nor `adk` is on PATH. Use `python -m streamlit run
app.py` if the bare command fails. `.claude/launch.json` defines a `streamlit-ui` config on
port 8501.

## Architecture

Three stages, strictly ordered — each consumes the previous one's artifacts on disk:

```
generate_data.py -> data/*.csv -> train_model.py -> models/*.joblib -> predict.py -> app.py
                                                                                  -> demo_tool.py
```

`src/predict.py` is the single scoring implementation. `app.py` and `demo_tool.py` both call
`score_applicant`; nothing else should reimplement the scoring path.

**`src/` is not a package.** There is no `__init__.py` and imports inside it are flat
(`from config import ...`, not `from src.config import ...`). Two mechanisms keep that working,
and a new script must use one of them:

- `app.py` and `demo_tool.py` live at the root and import from `src/`, so they
  `sys.path.insert(0, PROJECT_ROOT / "src")` before their first `from predict import ...`.
- The pipeline scripts are *run* as `python src/train_model.py`, which puts `src/` on
  `sys.path` automatically. `run_pipeline.py` launches them as subprocesses rather than
  importing them, so it needs no path juggling of its own.

**`src/config.py` is the single source of truth** for paths, the seed, and — critically — the
two distinct feature lists:

- `RAW_FEATURES` — the 8 fields a user enters, including `existing_debt`
- `MODEL_FEATURES` — the 8 columns the model is fitted on, where `existing_debt` is replaced
  by the engineered `dti = existing_debt / income`

These are both length 8 but are **not the same set**. `existing_debt` is deliberately excluded
from the model because `dti` already carries it and two near-collinear columns would
destabilise the coefficients this project exists to read. Any array or DataFrame passed to the
scaler or model must be in `MODEL_FEATURES` order.

## Non-obvious constraints

**The data is synthetic with a known generating rule**, and the model is verified against it.
`generate_data.py` builds the target from an explicit weighted latent score; `train_model.py`
holds an `EXPECTED_SIGNS` dict asserting each recovered coefficient carries the sign that rule
implies, and reports mismatches in the evaluation report. **If you change the weights in
`generate_data.py`, revisit `EXPECTED_SIGNS`** — otherwise the report will flag a mismatch that
is actually just a stale expectation.

**`age` and `loan_term` are intentional controls.** They are absent from the generating rule,
so their near-zero coefficients (about −0.084 each) are the correct result and a deliberate
sanity check. Do not "fix" them.

**Results are hardcoded in the docs.** `README.md` and `docs/WRITEUP.md` quote specific numbers
(accuracy 0.8975, ROC-AUC 0.9787, the full coefficient table, 59.5/40.5 class balance) from a
seed-42 run. Changing the seed, the sample size, the generating weights, or the model
hyperparameters invalidates those numbers in three places — the two docs and
`outputs/evaluation_report.md`. The report regenerates itself; the docs do not.

**`load_artifacts()` and `load_metadata()` in `predict.py` are `lru_cache`d.** Streamlit reruns
the whole script on every interaction, so this matters for responsiveness — but it also means a
freshly retrained model is not picked up until the Streamlit process restarts.

**Model artifacts must exist before the UI starts.** `app.py` calls `load_metadata()` at import
time and `st.stop()`s with an instruction to run the pipeline if `models/` is empty.

## Project history

The spec in `credit_risk_solution_design.pptx` called for a conversational Google ADK agent as
the interface and explicitly ruled out Streamlit and Flask. That constraint was lifted during
implementation; the ADK agent (`loan_agent/`) was built, verified, and then removed at the
user's request in favour of the Streamlit UI. `docs/WRITEUP.md` records this deviation
deliberately — do not "correct" it to match the deck.
