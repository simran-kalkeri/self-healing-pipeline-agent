# Self-Healing Agentic ML Pipeline — Progress Report

**Date:** 2026-04-05  
**Status:** ✅ COMPLETE

---

## 🎯 What Was Built

A production-ready, end-to-end **Self-Healing Agentic ML Pipeline** using:
- **LangGraph** — multi-node directed graph orchestration
- **Groq LLM** (`llama3-70b-8192`) — autonomous remediation decisions
- **scikit-learn** — ML training (Logistic Regression, Random Forest)
- **imbalanced-learn** — SMOTE rebalancing fix
- **pandas / numpy** — data manipulation and synthetic generation

---

## 📁 Files Delivered

| File | Description |
|------|-------------|
| `pipeline.py` | Main codebase — all agents, graph, data, runner |
| `requirements.txt` | All Python dependencies |
| `progress.md` | This file |

---

## 🧠 Architecture

```
detector → train_and_evaluate → llm_decision → apply_fix → update_memory → validator
    ↑____________________________________________________|  (if not done)
```

### State (TypedDict)
| Field | Type | Purpose |
|-------|------|---------|
| `df` | DataFrame | Working dataset |
| `error` | str | Current detected error |
| `action` | str | LLM-chosen remediation |
| `f1` | float | Current F1 score |
| `best_f1` | float | Best F1 across all attempts |
| `attempt` | int | Loop counter |
| `history` | list[dict] | Memory of all past attempts |
| `done` | bool | Halt flag |
| `model_type` | str | "logistic" or "random_forest" |

---

## 🤖 Nodes (Agents) Implemented

### 1. `detector`
- Checks for missing required columns (`age`, `gender`, `income`, `target`)
- Checks for type mismatches in numeric columns
- Sets `state["error"]` with a structured error string

### 2. `train_and_evaluate`
- Encodes categorical columns
- Trains Logistic Regression or Random Forest
- Computes weighted F1 score
- Updates `best_f1`; sets error if F1 < 0.70 threshold

### 3. `llm_decision`
- Calls Groq (`llama3-70b-8192`) with strict, token-minimal prompt
- Sends: error, dataset summary, last 5 history entries
- Parses response with 3-layer fallback (`json.loads` → regex → keyword scan)
- Falls back to `STOP` on any failure — **never crashes**

### 4. `apply_fix`
| Action | Fix Applied |
|--------|------------|
| `ADD_MISSING_COLUMNS` | Reconstructs missing columns with synthetic values |
| `CAST_DATATYPE` | `pd.to_numeric(..., errors="coerce")` + median fill |
| `REBALANCE_DATA` | SMOTE with adaptive `k_neighbors` |
| `CHANGE_MODEL` | Toggle Logistic Regression ↔ Random Forest |
| `FEATURE_SELECTION` | `SelectKBest(f_classif, k=3)` |
| `STOP` | No-op |

### 5. `update_memory`
- Appends `{attempt, error, action, f1}` to `state["history"]`

### 6. `validator`
- Stops if: `action == STOP`, `attempt >= 5`, or `error == ""`
- Otherwise loops back to `detector`

---

## 🔁 Failure Injection (Testing Only)

| Failure Type | What It Does |
|---|---|
| `MISSING_COLUMN` | Drops `income` column |
| `TYPE_MISMATCH` | Corrupts `age` with "N/A" strings |
| `LOW_PERFORMANCE` | Shuffles target to destroy signal |

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with a simulated failure
```bash
# Default: LOW_PERFORMANCE failure
python pipeline.py

# Missing column failure
python pipeline.py --failure MISSING_COLUMN

# Type mismatch failure
python pipeline.py --failure TYPE_MISMATCH

# Clean run (no failure)
python pipeline.py --failure none
```

---

## 🔒 Safety & Robustness

- ✅ 3-layer JSON parsing fallback — never crashes on bad LLM output
- ✅ SMOTE uses adaptive `k_neighbors` to handle small class sizes
- ✅ All node exceptions are caught and converted to error states
- ✅ LLM prompt is minimal (max_tokens=64) to conserve Groq API credits
- ✅ MAX_ATTEMPTS = 5 prevents infinite loops

---

## ✅ Completion Status

| Component | Status |
|---|---|
| State TypedDict | ✅ Done |
| Detector node | ✅ Done |
| Train & Evaluate node | ✅ Done |
| LLM Decision node (Groq) | ✅ Done |
| Apply Fix node (5 fix types) | ✅ Done |
| Update Memory node | ✅ Done |
| Validator node | ✅ Done |
| LangGraph wiring + conditional edges | ✅ Done |
| Synthetic data generation | ✅ Done |
| Failure injection (3 types) | ✅ Done |
| Safe JSON parsing (3-layer) | ✅ Done |
| CLI argument parser | ✅ Done |
| requirements.txt | ✅ Done |
| Progress documentation | ✅ Done |

*
