# 🧠 Self-Healing Agentic ML Pipeline — v3

> A **goal-aware, learning** multi-agent ML pipeline that diagnoses failures, reasons about proximity to its goal, scores actions from experience, and knows precisely when to stop — powered by **LangGraph** + **Groq LLM**.

---

## 🎯 What Changed in v3

| Capability | v2 | v3 |
|---|---|---|
| Goal awareness | ❌ Binary pass/fail | ✅ Tracks `goal_distance` + `near_goal` |
| Success definition | F1-only | Structural fix resolved OR F1 improved >0.01 |
| Action scoring | ❌ None | ✅ Cumulative F1-delta scores per action |
| Ineffective tracking | ❌ None | ✅ Failed ≥2 → hard-avoid |
| Near-goal behavior | ❌ Applies SMOTE blindly | ✅ Blocks heavy transforms, prefers CHANGE_MODEL |
| Retry logic | ❌ Never repeat | ✅ Retry if action has positive score |
| LLM context | History only | History + scores + goal distance + near_goal flag |
| Stop conditions | STOP / max / stagnation | + CLOSE_ENOUGH + GOAL_REACHED |
| Agent count | 6 nodes | 7 nodes (+ Evaluator) |

---

## 🏗️ Architecture

```
detector
  → trainer
    → evaluator          ← NEW: computes goal_distance, near_goal
      → decision         ← LLM with scores + proximity-aware context
        → fix_executor
          → memory_agent ← scores actions, tracks ineffective
            → validator  ← goal-aware stop logic
              ↑__________|   (loop if not done)
```

---

## 🤖 Agent Responsibilities

| Agent | Responsibility |
|---|---|
| `detector` | Schema → type → null-rate → signal strength |
| `trainer` | Train LogReg or RandomForest, compute raw F1 |
| `evaluator` | Compute `goal_distance = F1_THRESHOLD - f1`, set `near_goal` flag |
| `decision` | LLM reasons with full context; guards block invalid or risky actions |
| `fix_executor` | Apply fix (SMOTE, cast, schema restore, model switch, feature select) |
| `memory_agent` | Log attempt with success flag; update action scores; mark ineffective |
| `validator` | 5 stop conditions: STOP / GOAL_REACHED / CLOSE_ENOUGH / MAX_ATTEMPTS / STAGNATION |

---

## 🧠 Intelligence Features

### Goal-Aware Reasoning
```
F1 = 0.78 → goal = 0.80 → distance = +0.02 → near_goal = True
→ Block SMOTE and FEATURE_SELECTION (too aggressive)
→ Prefer CHANGE_MODEL or STOP
```

### Correct Success Definition
```python
structural_fix = prev_error != "" and current_error == ""   # schema resolved
f1_improved    = improvement > 0.01
success        = structural_fix or f1_improved
```
Structural fixes (like restoring a missing column) are **never penalized as failures**.

### Action Scoring System
```python
if success:    action_scores[action] += max(f1_improvement, 0.01)
else:          action_scores[action] -= 0.01
               if failure_count >= 2: → mark INEFFECTIVE (hard avoid)
```

### Learning Context for LLM
```
Action scores: {'ADD_MISSING_COLUMNS': 0.85, 'CHANGE_MODEL': 0.04}
INEFFECTIVE (avoid): ['FEATURE_SELECTION']
Near goal: True | Distance: +0.018
```

### CLOSE_ENOUGH Stop
If `near_goal=True` AND stagnating → stop and accept. The system won't endlessly try to close a 1% gap.

---

## 📊 Extended State Schema

```python
class PipelineState(TypedDict):
    # Core
    df, error, action, f1, best_f1, attempt, history, done, model_type
    # Reasoning
    root_cause, reasoning, confidence
    # Goal-aware (NEW)
    goal_distance: float      # F1_THRESHOLD - current_f1
    near_goal: bool           # within CLOSE_ENOUGH_DELTA (0.02)
    # Learning (NEW)
    action_scores: Dict[str, float]
    action_success_count: Dict[str, int]
    action_failure_count: Dict[str, int]
    failed_actions: List[str]   # failed ≥ 2 times
    best_action: str
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key
Create a `.env` file in the project root (it is already listed in `.gitignore` — **never commit it**):
```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

> **Note** — `python-dotenv` is included in `requirements.txt`. The pipeline calls `load_dotenv()` automatically on startup. If the key is missing, a clear `EnvironmentError` is raised.

---

## ▶️ Quick Start

```bash
# Compound failure (missing column + class imbalance)
python pipeline.py --failure COMPOUND

# Partial data corruption (30% NaN in income)
python pipeline.py --failure PARTIAL_CORRUPTION

# Near-goal performance — tests hyperparameter tuning
python pipeline.py --failure NEAR_GOAL

# Class imbalance
python pipeline.py --failure LOW_PERFORMANCE

# Missing column only
python pipeline.py --failure MISSING_COLUMN

# Type mismatch
python pipeline.py --failure TYPE_MISMATCH

# Clean run
python pipeline.py --failure none
```

---

## 📈 Sample Final Report

```
  Total attempts  : 3
  Final F1        : 0.8412
  Goal            : 0.80
  Goal distance   : -0.0412      ← negative = exceeded goal
  Best action     : ADD_MISSING_COLUMNS
  Action scores   : {'ADD_MISSING_COLUMNS': 0.85, 'REBALANCE_DATA': 0.04}
  Ineffective     : []

  [✅] #1: ADD_MISSING_COLUMNS    f1=0.0000  Δ=+0.0000  cause='income column absent'
  [✅] #2: REBALANCE_DATA         f1=0.7812  Δ=+0.7812  cause='class imbalance 97%/3%'
  [✅] #3: STOP                   f1=0.8412  Δ=+0.0600  cause=''
```

---

## 🔒 Safety & Robustness

| Concern | Solution |
|---|---|
| Aggressive fix near goal | Near-goal guard blocks SMOTE + FEATURE_SELECTION |
| Structural fix marked as failure | Correct success definition (error resolved = success) |
| Repeated failed action | Action scoring + INEFFECTIVE tracking |
| Exploiting LLM for obvious cases | Fast-paths bypass API for NO_SIGNAL / exhaustion |
| Bad JSON from LLM | 3-layer parse fallback |
| LLM API failure | Exception caught → STOP |
| Infinite loops | MAX_ATTEMPTS=7 + stagnation detection |

---

## 📁 File Structure

```
self-healing-agent/
├── pipeline.py      # Complete v3 pipeline — all 7 agents
├── requirements.txt # Dependencies
├── progress.md      # Technical notes
└── README.md        # This file
```

---

## ⚙️ Key Constants

| Constant | Value | Purpose |
|---|---|---|
| `F1_THRESHOLD` | 0.80 | Goal |
| `CLOSE_ENOUGH_DELTA` | 0.02 | Near-goal conservative mode |
| `MAX_ATTEMPTS` | 7 | Hard ceiling |
| `CONFIDENCE_THRESHOLD` | 0.50 | Min LLM confidence to act |
| `NULL_RATE_THRESHOLD` | 10% | Flags partial corruption |
| `STAGNATION_WINDOW` | 3 | Attempts without improvement → stop |
