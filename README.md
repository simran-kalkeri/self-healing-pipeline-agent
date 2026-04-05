# 🧠 Self-Healing Agentic ML Pipeline — v3

> A **goal-aware, learning** multi-agent ML pipeline that diagnoses failures, reasons about proximity to its goal, scores actions from experience, and knows precisely when to stop — powered by **LangGraph** + **Groq LLM**.

---

## 🎯 What Changed in v3

| Capability | v2 | v3 |
|---|---|---|
| Goal awareness | ❌ Binary pass/fail | ✅ Tracks `goal_distance` + `near_goal` |
| Success definition | F1-only | Structural fix resolved OR F1 improved >0.01 |
| Action scoring | ❌ None | ✅ **Weighted** scoring: structural=0.10, large gain=1.5×, small=raw |
| Action priority | ❌ No ranking | ✅ Ranked list passed to LLM (sorted by score) |
| Near-goal behaviour | ❌ Applies SMOTE blindly | ✅ **Forces** `TUNE_HYPERPARAMETERS` first, then CHANGE_MODEL |
| Hyperparameter tuning | ❌ None | ✅ GridSearchCV (LogReg: C/max_iter, RF: n_estimators/max_depth) |
| Ineffective tracking | ❌ None | ✅ Failed ≥2 → hard-avoid |
| Retry logic | ❌ Never repeat | ✅ Retry if action has positive score |
| LLM context | History only | History + **ranked scores** + goal distance + near_goal flag |
| Stop conditions | STOP / max / stagnation | + CLOSE_ENOUGH + GOAL_REACHED |
| Agent count | 6 nodes | 7 nodes (+ Evaluator) |
| API key | Hardcoded | ✅ `.env` via `python-dotenv` |

---

## 🏗️ Architecture

```
detector
  → trainer
    → evaluator         ← computes goal_distance, near_goal
      → decision        ← hard-rule near-goal force + ranked LLM context
        → fix_executor  ← TUNE_HYPERPARAMETERS (GridSearchCV)
          → memory_agent ← weighted scoring + retroactive credit + INEFFECTIVE
            → validator  ← STOP / GOAL_REACHED / CLOSE_ENOUGH / MAX / STAGNATION
              ↑__________|   (loop if not done)
```

---

## 🤖 Agent Responsibilities

| Agent | Responsibility |
|---|---|
| `detector` | Schema → type → null-rate → signal strength |
| `trainer` | Train LogReg or RandomForest with tuned hyperparams if available |
| `evaluator` | Compute `goal_distance = F1_THRESHOLD - f1`, set `near_goal` flag |
| `decision` | Hard-force TUNE_HYPERPARAMETERS when near goal; pass ranked actions to LLM |
| `fix_executor` | SMOTE / cast / schema restore / model switch / feature select / **GridSearchCV** |
| `memory_agent` | Weighted scoring + retroactive credit for structural fixes + INEFFECTIVE tracking |
| `validator` | 5 stop conditions: STOP / GOAL_REACHED / CLOSE_ENOUGH / MAX_ATTEMPTS / STAGNATION |

---

## 🧠 Intelligence Features

### 1. Hard Near-Goal Force
```python
# Bypass LLM entirely — deterministic, no token cost
if near_goal and "TUNE_HYPERPARAMETERS" not in tried:
    action = "TUNE_HYPERPARAMETERS"   # forced, confidence = 1.0
```
```
[DECISION] 🎯 Near goal → FORCING TUNE_HYPERPARAMETERS (precision rule)
```

### 2. Ranked Action Priority
```python
ranked_actions = sorted(available, key=lambda a: action_scores.get(a, 0), reverse=True)
# Passed to LLM as: "Recommended actions (ranked by past score): [...]"
```
LLM now **reasons with explicit guidance** — highest-score actions listed first.

### 3. Weighted Action Scoring
```python
if structural_fix:         delta = 0.10          # biggest reward — schema repair matters most
elif improvement > 0.05:   delta = improvement * 1.5   # amplify large gains
else:                      delta = max(improvement, 0.01)  # floor for small wins
```
Teaches the system: structural fix > large F1 gain > small F1 gain.

### 4. Retroactive Credit for Structural Fixes
```
CAST_DATATYPE fixes NaN in loop 1 → F1 only improves in loop 2
→ memory_agent detects the error cleared and credits CAST_DATATYPE retroactively
[MEMORY] 🌟 Retroactive credit → 'CAST_DATATYPE' resolved structural error
```

### 5. Hyperparameter Tuning (GridSearchCV)
```
LogReg  → tunes: C ∈ [0.01, 0.1, 1, 10, 100], max_iter ∈ [300, 500, 1000]
RF      → tunes: n_estimators ∈ [50, 100, 200], max_depth ∈ [None, 5, 10, 20]
cv=3, scoring=f1_weighted
```
Best params stored in state → used by trainer on the next loop.  
Model params are **cleared when switching model type** (no LogReg params bleeding into RF).

### 6. CLOSE_ENOUGH Stop
If `near_goal=True` AND stagnating → accept current F1 gracefully. Won't thrash endlessly trying to close a 1–2% gap.

---

## 📊 State Schema

```python
class PipelineState(TypedDict):
    # Core
    df, error, action, f1, best_f1, attempt, history, done
    model_type: str          # "logistic" | "random_forest"
    model_params: Dict       # best hyperparams from TUNE_HYPERPARAMETERS
    # Reasoning
    root_cause, reasoning, confidence
    # Goal-aware
    goal_distance: float     # F1_THRESHOLD - current_f1 (negative = exceeded)
    near_goal: bool          # within CLOSE_ENOUGH_DELTA (0.06) of threshold
    # Learning
    action_scores:        Dict[str, float]   # weighted cumulative score per action
    action_success_count: Dict[str, int]
    action_failure_count: Dict[str, int]
    failed_actions:       List[str]          # failed ≥2 times → hard avoid
    best_action:          str
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key
Create a `.env` file in the project root (already in `.gitignore` — **never commit it**):
```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

> `python-dotenv` is included in `requirements.txt`. The pipeline calls `load_dotenv()` on startup. Missing key raises a clear `EnvironmentError`.

---

## ▶️ Quick Start

```bash
# Compound failure (missing column + class imbalance) — multi-step fix
python pipeline.py --failure COMPOUND

# Partial data corruption (30% NaN in income) — null detection + CAST_DATATYPE
python pipeline.py --failure PARTIAL_CORRUPTION

# Near-goal performance — triggers TUNE_HYPERPARAMETERS hard rule
python pipeline.py --failure NEAR_GOAL

# Class imbalance — REBALANCE_DATA (SMOTE)
python pipeline.py --failure LOW_PERFORMANCE

# Missing column — ADD_MISSING_COLUMNS
python pipeline.py --failure MISSING_COLUMN

# Type mismatch — CAST_DATATYPE
python pipeline.py --failure TYPE_MISMATCH

# Clean run — should pass immediately with F1 ≥ 0.80
python pipeline.py --failure none
```

---

## 🧪 What Each Failure Tests

| Failure | Injected Problem | Expected Actions |
|---|---|---|
| `COMPOUND` | Missing column + 5% class imbalance | ADD_MISSING_COLUMNS → REBALANCE_DATA |
| `PARTIAL_CORRUPTION` | 30% NaN in `income` | CAST_DATATYPE (fill median) |
| `NEAR_GOAL` | 10% label noise → F1 ≈ 0.75 | **TUNE_HYPERPARAMETERS** (forced) |
| `LOW_PERFORMANCE` | 10% minority class | REBALANCE_DATA |
| `MISSING_COLUMN` | Drop `income` | ADD_MISSING_COLUMNS |
| `TYPE_MISMATCH` | String values in numeric col | CAST_DATATYPE |
| `none` | Clean data | STOP immediately |

---

## 📈 Sample Final Report — NEAR_GOAL

```
  Total attempts  : 7
  Final F1        : 0.7723
  Best F1         : 0.7723
  Goal            : 0.80
  Goal distance   : +0.0277        ← close but ceiling hit (label noise)
  Best action     : TUNE_HYPERPARAMETERS
  Action scores   : {'TUNE_HYPERPARAMETERS': 0.782, 'CHANGE_MODEL': -0.01, ...}

  [✅] #1: TUNE_HYPERPARAMETERS  f1=0.7500  Δ=+0.7500  cause='F1 is near goal threshold'  ← FORCED
  [❌] #2: CHANGE_MODEL          f1=0.7500  Δ=+0.0000
  [❌] #3: FEATURE_SELECTION     f1=0.7399  Δ=-0.0101
  [✅] #5: TUNE_HYPERPARAMETERS  f1=0.7522  Δ=+0.0123  ← RF tuned: max_depth=5
  [✅] #6: TUNE_HYPERPARAMETERS  f1=0.7723  Δ=+0.0201  ← RF params applied
  → STOP: MAX_ATTEMPTS (label noise ceiling ≈ 0.77)
```

---

## 🔒 Safety & Robustness

| Concern | Solution |
|---|---|
| Aggressive fix near goal | Hard-force TUNE_HYPERPARAMETERS; block SMOTE / FEATURE_SELECTION |
| Structural fix scored as failure | Retroactive credit when error clears next loop |
| Repeated failed action | Weighted scoring → INEFFECTIVE tracking (fail ≥2 → hard avoid) |
| LogReg params bleeding into RF | `model_params = {}` cleared on CHANGE_MODEL |
| LLM chooses blindly | Ranked actions passed to LLM — reasons with score guidance |
| LLM JSON truncated | `max_tokens=300`, 3-layer parse fallback |
| LLM API failure | Exception caught → STOP |
| Infinite loops | MAX_ATTEMPTS=7 + stagnation detection |
| No signal | Fast-path bypass: NO_SIGNAL → STOP immediately (no LLM call) |

---

## 📁 File Structure

```
self-healing-agent/
├── pipeline.py       # Complete v3 pipeline — all 7 agents
├── requirements.txt  # Dependencies (incl. python-dotenv)
├── .env              # API key (gitignored — never committed)
├── .gitignore
├── progress.md       # Technical notes
└── README.md         # This file
```

---

## ⚙️ Key Constants

| Constant | Value | Purpose |
|---|---|---|
| `F1_THRESHOLD` | 0.80 | Goal |
| `CLOSE_ENOUGH_DELTA` | 0.06 | Near-goal window (F1 ≥ 0.74) |
| `MAX_ATTEMPTS` | 7 | Hard ceiling |
| `CONFIDENCE_THRESHOLD` | 0.50 | Min LLM confidence to act |
| `NULL_RATE_THRESHOLD` | 10% | Flags partial corruption |
| `STAGNATION_WINDOW` | 3 | Attempts without improvement → stop |
