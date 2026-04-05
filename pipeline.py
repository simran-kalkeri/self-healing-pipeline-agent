"""
Self-Healing Agentic ML Pipeline — v3 (Goal-Aware + Learning)
=============================================================
A goal-aware, learning multi-agent system using LangGraph + Groq LLM.

Agents:
  1. detector      — schema / type / null / signal checks
  2. trainer       — model training, raw F1
  3. evaluator     — goal distance, proximity flags
  4. decision      — Groq LLM with action scores + learning context
  5. fix_executor  — validated fix application
  6. memory_agent  — history + action scoring + ineffective tracking
  7. validator     — intelligent goal-aware stop/continue logic

NOTE: Failure injection is for testing purposes only.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv

load_dotenv()   # loads .env into os.environ

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not set. Add it to a .env file:\n"
        "  GROQ_API_KEY=your_key_here"
    )
GROQ_MODEL = "llama-3.3-70b-versatile"

F1_THRESHOLD       = 0.80   # Goal
CLOSE_ENOUGH_DELTA = 0.06   # Within this of goal → conservative mode (covers 0.74–0.79)
MAX_ATTEMPTS       = 7      # Raised for multi-step compound failures
TARGET_COLUMN      = "target"
REQUIRED_COLS      = ["age", "gender", "income", TARGET_COLUMN]

MIN_SIGNAL_THRESHOLD = 0.001
STAGNATION_WINDOW    = 3
STAGNATION_DELTA     = 0.01
CONFIDENCE_THRESHOLD = 0.50
NULL_RATE_THRESHOLD  = 0.10   # ≥ 10% NaN in numeric col → flagged

ALL_FIX_ACTIONS = frozenset({
    "REBALANCE_DATA", "CHANGE_MODEL", "FEATURE_SELECTION",
    "ADD_MISSING_COLUMNS", "CAST_DATATYPE", "TUNE_HYPERPARAMETERS",
})

SYSTEM_PROMPT = """You are an expert autonomous ML pipeline agent.

You reason about root causes, goal proximity, and historical performance to choose the best action.

Critical rules (follow in order):
1.  NO_SIGNAL → STOP immediately. No fix can recover missing signal.
2.  Fix schema/null/type issues FIRST: ADD_MISSING_COLUMNS, CAST_DATATYPE before any model actions.
3.  If near_goal=True (F1 within 0.06 of target, i.e. 0.74–0.79):
    → ALWAYS prefer TUNE_HYPERPARAMETERS first (precision GridSearchCV tuning)
    → Then CHANGE_MODEL as second choice
    → Do NOT apply SMOTE or FEATURE_SELECTION — they may hurt near the goal.
4.  NEVER use an action listed as INEFFECTIVE (failed ≥ 2 times).
5.  Prefer actions with high action_scores — these proved effective before.
6.  If F1 is stagnating across attempts → STOP.
7.  If confidence < 0.5 → STOP.
8.  REBALANCE_DATA helps class imbalance only — not signal or type issues.
9.  NULL_VALUES error → CAST_DATATYPE fills NaN with column median.
10. You may retry an action only if it has a POSITIVE action_score.

Available actions:
REBALANCE_DATA        - SMOTE to fix class imbalance
CHANGE_MODEL          - switch LogisticRegression ↔ RandomForest
FEATURE_SELECTION     - keep top-k most informative features
ADD_MISSING_COLUMNS   - restore absent required columns
CAST_DATATYPE         - fix non-numeric values OR fill NaN in numeric columns
TUNE_HYPERPARAMETERS  - GridSearchCV to find best model parameters (use when near goal)
STOP                  - halt pipeline

Return ONLY this exact JSON (no markdown, no explanation):
{
  "root_cause": "one-sentence diagnosis",
  "reasoning": "why this action will help given goal proximity, error, and history",
  "confidence": 0.0,
  "action": "ACTION_NAME"
}"""

# ─────────────────────────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    df:           pd.DataFrame
    error:        str
    action:       str
    f1:           float
    best_f1:      float
    attempt:      int
    history:      List[Dict[str, Any]]
    done:         bool
    model_type:   str           # "logistic" | "random_forest"
    model_params: Dict[str, Any]  # best hyperparams from TUNE_HYPERPARAMETERS
    # ── Reasoning fields ────────────────────────────────────────────────────
    root_cause:   str
    reasoning:    str
    confidence:   float
    # ── Goal-aware fields ───────────────────────────────────────────────────
    goal_distance: float        # F1_THRESHOLD - current_f1  (negative = passed)
    near_goal:    bool          # within CLOSE_ENOUGH_DELTA of threshold
    # ── Learning fields ─────────────────────────────────────────────────────
    action_scores:        Dict[str, float]   # cumulative F1 impact per action
    action_success_count: Dict[str, int]
    action_failure_count: Dict[str, int]
    failed_actions:       List[str]          # failed ≥ 2 times → hard avoid
    best_action:          str                # highest-scoring action so far

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA + FAILURE INJECTION  (testing only)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate data where features genuinely predict target via logistic rule."""
    rng    = np.random.default_rng(seed)
    age    = rng.integers(18, 70, size=n).astype(float)
    income = rng.integers(20_000, 120_000, size=n).astype(float)
    gender = rng.choice(["M", "F"], size=n)

    age_norm    = (age - 18) / (70 - 18)
    income_norm = (income - 20_000) / 100_000
    score       = 0.6 * income_norm + 0.4 * age_norm
    noise       = rng.normal(0, 0.15, size=n)
    prob        = 1 / (1 + np.exp(-(score - 0.5 + noise) * 6))
    target      = (prob > 0.5).astype(int)

    return pd.DataFrame({
        "age":         age,
        "gender":      gender,
        "income":      income,
        TARGET_COLUMN: target,
    })


def inject_failures(df: pd.DataFrame, failure_type: str) -> pd.DataFrame:
    """
    Inject controlled failures silently — agent must self-diagnose.
    Print statements intentionally suppressed.
    """
    df  = df.copy()
    rng = np.random.default_rng(99)

    if failure_type == "MISSING_COLUMN":
        df = df.drop(columns=["income"], errors="ignore")

    elif failure_type == "TYPE_MISMATCH":
        df["age"] = df["age"].astype(str).apply(
            lambda x: "N/A" if rng.random() < 0.3 else x
        )

    elif failure_type == "LOW_PERFORMANCE":
        n            = len(df)
        n_minority   = max(int(n * 0.10), 5)
        labels       = np.zeros(n, dtype=int)
        labels[rng.choice(n, size=n_minority, replace=False)] = 1
        df[TARGET_COLUMN] = labels

    elif failure_type == "COMPOUND":
        # Missing column + severe class imbalance (5% minority).
        # Keep TOP-SCORING class-1 rows (by age, the surviving correlated feature)
        # so MI estimator reliably detects signal after schema is fixed.
        # Random sampling causes MI=0 due to k-NN estimator variance at small N.
        df = df.drop(columns=["income"], errors="ignore")
        class1_idx = df.index[df[TARGET_COLUMN] == 1].tolist()
        n_keep     = max(int(len(df) * 0.05), 5)     # 5% minority, top by age
        top_class1 = df.loc[class1_idx, "age"].sort_values(ascending=False).index[:n_keep]
        flip_mask  = df.index.isin(class1_idx) & ~df.index.isin(top_class1)
        df.loc[flip_mask, TARGET_COLUMN] = 0
        # ← no print: agent must detect schema error first, then class imbalance


    elif failure_type == "NEAR_GOAL":
        # Mild label noise: flip ~10% of labels randomly.
        # Brings F1 from ~0.87 down to ~0.77-0.79 → near_goal=True
        # Agent must recognise it's close and prefer TUNE_HYPERPARAMETERS.
        rng  = np.random.default_rng(7)
        mask = rng.random(len(df)) < 0.10
        df.loc[mask, TARGET_COLUMN] = 1 - df.loc[mask, TARGET_COLUMN]
        # ← no print: agent must infer from low-but-close F1

    return df

# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col != TARGET_COLUMN:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def dataset_summary(df: pd.DataFrame) -> str:
    lines = [
        f"Rows: {len(df)}, Columns: {list(df.columns)}",
        f"Dtypes: { {c: str(t) for c, t in df.dtypes.items()} }",
        f"Nulls:  { df.isnull().sum().to_dict() }",
    ]
    if TARGET_COLUMN in df.columns:
        lines.append(f"Target dist: {df[TARGET_COLUMN].value_counts().to_dict()}")
    return "\n".join(lines)


def compute_signal_strength(df: pd.DataFrame) -> float:
    try:
        enc = encode_dataframe(df)
        if TARGET_COLUMN not in enc.columns:
            return 0.0
        X = enc.drop(columns=[TARGET_COLUMN])
        y = enc[TARGET_COLUMN]
        X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1)
        if X.empty or len(y.unique()) < 2:
            return 0.0
        mi = mutual_info_classif(X, y, random_state=42)
        return float(np.max(mi)) if len(mi) > 0 else 0.0
    except Exception as exc:
        print(f"[SIGNAL]    ⚠  Signal computation failed: {exc}")
        return 0.0


def build_llm_context(state: PipelineState) -> str:
    """Rich context: history log + action scores + goal state."""
    history        = state["history"]
    action_scores  = state["action_scores"]
    failed_actions = state["failed_actions"]
    near_goal      = state["near_goal"]
    goal_distance  = state["goal_distance"]
    f1             = state["f1"]

    lines = [
        f"Current F1: {f1:.4f}  |  Goal: {F1_THRESHOLD}  |"
        f"  Distance: {goal_distance:+.4f}  |  Near goal: {near_goal}",
        "",
    ]

    if history:
        lines.append("Recent attempts:")
        for h in history[-5:]:
            tag = "✅" if h.get("success") else "❌"
            lines.append(
                f"  [{tag}] #{h['attempt']}: action={h['action']}  "
                f"f1={h['f1']}  Δ={h.get('improvement', 0):+.4f}  "
                f"cause='{h.get('root_cause', '')[:50]}'"
            )
    else:
        lines.append("No previous attempts.")

    if action_scores:
        sorted_scores = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
        lines.append(f"\nAction scores (↑ = more effective): {dict(sorted_scores)}")

    if failed_actions:
        lines.append(f"INEFFECTIVE (failed ≥2): {failed_actions}  ← avoid these")

    return "\n".join(lines)


def safe_parse_llm_response(raw: str) -> Dict[str, Any]:
    """3-layer fallback to extract {root_cause, reasoning, confidence, action}."""
    VALID   = ALL_FIX_ACTIONS | {"STOP"}
    default = {"root_cause": "unknown", "reasoning": "parse failed",
               "confidence": 0.0, "action": "STOP"}

    cleaned = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()

    # 1. Direct JSON parse
    try:
        obj    = json.loads(cleaned)
        action = obj.get("action", "").strip().upper()
        return {
            "root_cause": str(obj.get("root_cause", "unknown")),
            "reasoning":  str(obj.get("reasoning",  "no reasoning")),
            "confidence": float(obj.get("confidence", 0.0)),
            "action":     action if action in VALID else "STOP",
        }
    except Exception:
        pass

    # 2. Regex field extraction
    try:
        am = re.search(r'"action"\s*:\s*"([^"]+)"',     raw)
        cm = re.search(r'"root_cause"\s*:\s*"([^"]+)"', raw)
        rm = re.search(r'"reasoning"\s*:\s*"([^"]+)"',  raw)
        fm = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
        action = (am.group(1).strip().upper() if am else "STOP")
        return {
            "root_cause": cm.group(1)         if cm else "unknown",
            "reasoning":  rm.group(1)         if rm else "no reasoning",
            "confidence": float(fm.group(1))  if fm else 0.0,
            "action":     action if action in VALID else "STOP",
        }
    except Exception:
        pass

    # 3. Keyword scan
    for act in VALID:
        if act in raw.upper():
            default["action"]    = act
            default["reasoning"] = "extracted via keyword scan"
            return default

    print(f"[WARN] Cannot parse LLM response: {raw!r}")
    return default

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def detector(state: PipelineState) -> PipelineState:
    """Schema → type → null-rate → signal strength checks."""
    df    = state["df"]
    error = ""

    # 1. Missing required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        error = f"MISSING_COLUMNS:{','.join(missing)}"
        print(f"[DETECTOR]  ❌ Missing columns: {missing}")
        return {**state, "error": error}

    # 2. Type mismatch in numeric columns
    for col in ["age", "income"]:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                error = f"TYPE_MISMATCH:{col}"
                print(f"[DETECTOR]  ❌ Type mismatch in '{col}'")
                return {**state, "error": error}

    # 3. High null rate (partial corruption)
    for col in ["age", "income"]:
        if col in df.columns:
            null_rate = df[col].isnull().mean()
            if null_rate > NULL_RATE_THRESHOLD:
                error = f"NULL_VALUES:{col}:{null_rate:.0%}"
                print(f"[DETECTOR]  ❌ High null rate in '{col}': {null_rate:.0%}"
                      f"  (threshold={NULL_RATE_THRESHOLD:.0%})")
                return {**state, "error": error}

    # 4. Signal strength
    signal = compute_signal_strength(df)
    print(f"[DETECTOR]  📡 Signal strength: {signal:.4f}  threshold={MIN_SIGNAL_THRESHOLD}")
    if signal < MIN_SIGNAL_THRESHOLD:
        error = f"NO_SIGNAL:max_mi={signal:.4f}"
        print("[DETECTOR]  ❌ No learnable signal — marking NO_SIGNAL")
        return {**state, "error": error}

    print("[DETECTOR]  ✅ All checks passed")
    return {**state, "error": ""}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — TRAINER
# ─────────────────────────────────────────────────────────────────────────────

def trainer(state: PipelineState) -> PipelineState:
    """Train model and compute raw F1. Skips if structural error exists."""
    if state["error"]:
        print(f"[TRAINER]   ⏭  Skipping — error: {state['error']}")
        return state

    df = encode_dataframe(state["df"])
    if TARGET_COLUMN not in df.columns:
        return {**state, "error": "MISSING_COLUMNS:target"}

    X = df.drop(columns=[TARGET_COLUMN]).apply(pd.to_numeric, errors="coerce").dropna(axis=1)
    y = df[TARGET_COLUMN]

    if X.empty or len(y.unique()) < 2:
        return {**state, "error": "INVALID_DATA:cannot_train", "f1": 0.0}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_type   = state.get("model_type", "logistic")
    model_params = state.get("model_params", {})

    # Build model — merge tuned hyperparams if available
    if model_type == "random_forest":
        rf_cfg = {"n_estimators": 50, "random_state": 42, "n_jobs": -1}
        rf_cfg.update(model_params)
        model = RandomForestClassifier(**rf_cfg)
    else:
        lr_cfg = {"max_iter": 500, "random_state": 42}
        lr_cfg.update(model_params)
        model = LogisticRegression(**lr_cfg)

    tuned_note = f" (tuned: {model_params})" if model_params else ""
    try:
        model.fit(X_train, y_train)
        score = f1_score(y_test, model.predict(X_test), average="weighted", zero_division=0)
    except Exception as exc:
        return {**state, "error": f"TRAINING_ERROR:{exc}", "f1": 0.0}

    best_f1 = max(state.get("best_f1", 0.0), score)
    error   = "" if score >= F1_THRESHOLD else f"LOW_PERFORMANCE:f1={score:.4f}"
    print(f"[TRAINER]   📊 Model={model_type}{tuned_note}  F1={score:.4f}  "
          f"Best={best_f1:.4f}  Goal={F1_THRESHOLD}")

    return {**state, "f1": score, "best_f1": best_f1, "error": error}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluator(state: PipelineState) -> PipelineState:
    """Compute goal distance and proximity flag for goal-aware reasoning."""
    f1            = state["f1"]
    goal_distance = F1_THRESHOLD - f1
    near_goal     = 0.0 < goal_distance <= CLOSE_ENOUGH_DELTA

    if goal_distance <= 0:
        print(f"[EVALUATOR] ✅ GOAL REACHED — F1={f1:.4f} ≥ {F1_THRESHOLD}")
    elif near_goal:
        print(f"[EVALUATOR] 🎯 NEAR GOAL — F1={f1:.4f}  distance={goal_distance:+.4f}"
              f"  (conservative mode active)")
    else:
        print(f"[EVALUATOR] 📏 Goal distance: {goal_distance:+.4f}  F1={f1:.4f}")

    return {**state, "goal_distance": goal_distance, "near_goal": near_goal}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — DECISION  (LLM)
# ─────────────────────────────────────────────────────────────────────────────

def decision(state: PipelineState) -> PipelineState:
    """Goal-aware, learning-driven LLM decision agent."""
    error          = state["error"]
    history        = state["history"]
    near_goal      = state["near_goal"]
    goal_distance  = state["goal_distance"]
    failed_actions = state["failed_actions"]
    action_scores  = state["action_scores"]

    _stop_defaults = {"root_cause": "", "reasoning": "", "confidence": 1.0}

    # ── Fast-path: goal reached ─────────────────────────────────────────────
    if not error:
        print("[DECISION]  ✅ Goal reached — action=STOP")
        return {**state, "action": "STOP", **_stop_defaults}

    # ── Fast-path: no signal → unsolvable ──────────────────────────────────
    if "NO_SIGNAL" in error:
        print("[DECISION]  🛑 NO_SIGNAL — action=STOP")
        return {**state, "action": "STOP",
                "root_cause": "dataset has no learnable signal",
                "reasoning":  "no preprocessing can recover information that doesn't exist",
                "confidence": 1.0}

    tried_actions = {h["action"] for h in history}

    # Available = all fix actions not in failed_actions, not tried
    available = ALL_FIX_ACTIONS - set(failed_actions) - tried_actions

    # Retryable = previously tried AND has positive score (proved useful)
    retryable = {
        a for a in tried_actions
        if a in action_scores and action_scores[a] > 0
        and a not in failed_actions
    }
    available = available | retryable

    if not available:
        print("[DECISION]  🛑 No viable actions remain — action=STOP")
        return {**state, "action": "STOP",
                "root_cause": "all fix strategies exhausted or ineffective",
                "reasoning":  "no new or retryable actions available",
                "confidence": 1.0}

    # ── Build context and call LLM ──────────────────────────────────────────
    ctx = build_llm_context(state)
    user_message = (
        f"Pipeline error: {error}\n"
        f"Near goal: {near_goal}  |  Goal distance: {goal_distance:+.4f}\n"
        f"Tried actions: {sorted(tried_actions)}\n"
        f"Available actions: {sorted(available)}\n"
        f"Ineffective (avoid): {sorted(failed_actions)}\n"
        f"Dataset:\n{dataset_summary(state['df'])}\n\n"
        f"Context:\n{ctx}"
    )

    print(f"[DECISION]  🤖 Querying Groq ({GROQ_MODEL})...")
    root_cause = reasoning = ""
    confidence = 0.0
    action     = "STOP"

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY, model_name=GROQ_MODEL,
            temperature=0, max_tokens=300,   # 300 to avoid JSON truncation
        )
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        raw = response.content.strip()
        print(f"[DECISION]  📨 Raw: {raw}")

        parsed     = safe_parse_llm_response(raw)
        root_cause = parsed["root_cause"]
        reasoning  = parsed["reasoning"]
        confidence = max(0.0, min(1.0, parsed["confidence"]))
        action     = parsed["action"]

        print(f"[DECISION]  🔍 Root cause : {root_cause}")
        print(f"[DECISION]  💭 Reasoning  : {reasoning}")
        print(f"[DECISION]  📊 Confidence : {confidence:.2f}")

        # Guard 1: low confidence
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"[DECISION]  ⚠  Confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD} → STOP")
            action = "STOP"

        # Guard 2: ineffective action
        elif action in failed_actions:
            print(f"[DECISION]  ⚠  '{action}' INEFFECTIVE (failed ≥2) → STOP")
            action = "STOP"

        # Guard 3: near-goal → block heavy transforms, prefer precision actions
        elif near_goal and action in {"REBALANCE_DATA", "FEATURE_SELECTION"}:
            print(f"[DECISION]  ⚠  Near goal → blocking '{action}' (too aggressive)")
            if "TUNE_HYPERPARAMETERS" not in tried_actions and "TUNE_HYPERPARAMETERS" not in failed_actions:
                action    = "TUNE_HYPERPARAMETERS"
                reasoning = "(near-goal override: hyperparameter tuning is safer than data transforms)"
            elif "CHANGE_MODEL" not in tried_actions and "CHANGE_MODEL" not in failed_actions:
                action    = "CHANGE_MODEL"
                reasoning = "(near-goal override: model switch preferred over data transform)"
            else:
                action = "STOP"

        # Guard 4: structurally irrelevant actions
        elif action == "ADD_MISSING_COLUMNS" and "MISSING" not in error.upper():
            print(f"[DECISION]  ⚠  ADD_MISSING_COLUMNS irrelevant to '{error}' → STOP")
            action = "STOP"
        elif (action == "CAST_DATATYPE"
              and "TYPE_MISMATCH" not in error.upper()
              and "NULL_VALUES"   not in error.upper()):
            print(f"[DECISION]  ⚠  CAST_DATATYPE irrelevant to '{error}' → STOP")
            action = "STOP"
        elif action == "REBALANCE_DATA" and "NO_SIGNAL" in error.upper():
            print(f"[DECISION]  ⚠  REBALANCE_DATA cannot fix NO_SIGNAL → STOP")
            action = "STOP"

    except Exception as exc:
        print(f"[DECISION]  ⚠  LLM failed: {exc} → STOP")
        root_cause = "LLM error"
        reasoning  = str(exc)
        confidence = 0.0
        action     = "STOP"

    print(f"[DECISION]  ➡  Final action: {action}")
    return {**state,
            "action":     action,
            "root_cause": root_cause,
            "reasoning":  reasoning,
            "confidence": confidence}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — FIX EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

def fix_executor(state: PipelineState) -> PipelineState:
    """Apply the chosen fix with validation guards."""
    action = state["action"]
    df     = state["df"].copy()
    update: Dict[str, Any] = {}

    print(f"[FIX]       🔧 Applying: {action}")

    if action == "ADD_MISSING_COLUMNS":
        truly_missing = [c for c in REQUIRED_COLS if c not in df.columns and c != TARGET_COLUMN]
        if not truly_missing:
            print("[FIX]       ℹ  All columns present — no-op")
        else:
            rng = np.random.default_rng(42)
            generators = {
                "age":    lambda: rng.integers(18, 70, size=len(df)).astype(float),
                "income": lambda: rng.integers(20_000, 120_000, size=len(df)).astype(float),
                "gender": lambda: rng.choice(["M", "F"], size=len(df)),
            }
            for col in truly_missing:
                if col in generators:
                    df[col] = generators[col]()
                    print(f"[FIX]       ➕ Restored column: '{col}'")
        update["df"] = df

    elif action == "CAST_DATATYPE":
        for col in ["age", "income"]:
            if col in df.columns:
                df[col]  = pd.to_numeric(df[col], errors="coerce")
                median   = df[col].median()
                n_filled = int(df[col].isnull().sum())
                df[col].fillna(median, inplace=True)
                print(f"[FIX]       🔢 '{col}' → numeric, "
                      f"filled {n_filled} NaN(s) with median={median:.1f}")
        update["df"] = df

    elif action == "REBALANCE_DATA":
        try:
            enc = encode_dataframe(df)
            X   = enc.drop(columns=[TARGET_COLUMN]).apply(pd.to_numeric, errors="coerce").dropna(axis=1)
            y   = enc[TARGET_COLUMN]
            k   = max(1, min(5, int(y.value_counts().min()) - 1))
            X_r, y_r = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
            res = pd.DataFrame(X_r, columns=X.columns)
            res[TARGET_COLUMN] = y_r
            update["df"] = res
            print(f"[FIX]       ⚖  SMOTE applied — new size: {len(res)}")
        except Exception as exc:
            print(f"[FIX]       ⚠  SMOTE failed: {exc}")
            update["df"] = df

    elif action == "CHANGE_MODEL":
        new_type = (
            "random_forest"
            if state.get("model_type", "logistic") == "logistic"
            else "logistic"
        )
        update["model_type"]   = new_type
        update["model_params"] = {}   # clear: old params are model-type-specific
        print(f"[FIX]       🔀 Model switched to: {new_type} (hyperparams reset)")

    elif action == "FEATURE_SELECTION":
        try:
            enc  = encode_dataframe(df)
            X    = enc.drop(columns=[TARGET_COLUMN]).apply(pd.to_numeric, errors="coerce").dropna(axis=1)
            y    = enc[TARGET_COLUMN]
            k    = max(1, min(3, X.shape[1]))
            sel  = SelectKBest(score_func=f_classif, k=k)
            X_s  = sel.fit_transform(X, y)
            cols = X.columns[sel.get_support()].tolist()
            df_s = pd.DataFrame(X_s, columns=cols)
            df_s[TARGET_COLUMN] = y.values
            update["df"] = df_s
            print(f"[FIX]       🎯 Feature selection — kept: {cols}")
        except Exception as exc:
            print(f"[FIX]       ⚠  Feature selection failed: {exc}")
            update["df"] = df

    elif action == "TUNE_HYPERPARAMETERS":
        try:
            enc = encode_dataframe(df)
            X   = enc.drop(columns=[TARGET_COLUMN]).apply(pd.to_numeric, errors="coerce").dropna(axis=1)
            y   = enc[TARGET_COLUMN]

            model_type = state.get("model_type", "logistic")
            if model_type == "random_forest":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth":    [None, 5, 10, 20],
                }
                base = RandomForestClassifier(random_state=42, n_jobs=-1)
            else:
                param_grid = {
                    "C":        [0.01, 0.1, 1, 10, 100],
                    "max_iter": [300, 500, 1000],
                }
                base = LogisticRegression(random_state=42)

            gs = GridSearchCV(
                base, param_grid, cv=3,
                scoring="f1_weighted", n_jobs=-1, refit=False,
            )
            gs.fit(X, y)
            best_params = gs.best_params_
            update["model_params"] = best_params
            print(f"[FIX]       🔬 Hyperparameter tuning complete")
            print(f"[FIX]           Model    : {model_type}")
            print(f"[FIX]           Best params : {best_params}")
            print(f"[FIX]           CV F1    : {gs.best_score_:.4f}")
        except Exception as exc:
            print(f"[FIX]       ⚠  Hyperparameter tuning failed: {exc}")

    else:
        print(f"[FIX]       🛑 '{action}' — no data transformation")

    return {**state, **update}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 6 — MEMORY AGENT
# ─────────────────────────────────────────────────────────────────────────────

def memory_agent(state: PipelineState) -> PipelineState:
    """
    Update history + action scoring system.

    Success = structural error resolved  OR  F1 improved > 0.01
    Score   = cumulative F1 delta per action
    Ineffective = action failed ≥ 2 times → added to failed_actions
    """
    history        = state["history"]
    action         = state["action"]
    f1             = state["f1"]
    error          = state["error"]
    action_scores  = dict(state["action_scores"])
    success_count  = dict(state["action_success_count"])
    failure_count  = dict(state["action_failure_count"])
    failed_actions = list(state["failed_actions"])

    prev_f1    = history[-1]["f1"]    if history else 0.0
    prev_error = history[-1]["error"] if history else ""
    improvement = f1 - prev_f1

    # ── Retroactive credit for structural fix ─────────────────────────────────────
    # If a structural error was cleared this loop, the PREVIOUS action caused it.
    # Credit that action retroactively and undo any penalty applied to it.
    if len(history) >= 1 and bool(prev_error) and not bool(error):
        prev_action = history[-1]["action"]
        if prev_action not in ("STOP", ""):
            retro_delta = 0.05   # fixed structural-fix credit
            action_scores[prev_action]  = action_scores.get(prev_action, 0.0) + retro_delta
            success_count[prev_action]  = success_count.get(prev_action, 0) + 1
            # Undo any failure penalty applied in the previous loop
            if failure_count.get(prev_action, 0) > 0:
                failure_count[prev_action] -= 1
                action_scores[prev_action] += 0.01   # reverse the -0.01 penalty
                # Un-mark as ineffective if failure count now < 2
                if prev_action in failed_actions and failure_count[prev_action] < 2:
                    failed_actions.remove(prev_action)
            print(f"[MEMORY]    🌟 Retroactive credit → '{prev_action}' resolved structural error")

    # ── Current action success ─────────────────────────────────────────────────
    structural_fix = bool(prev_error) and not bool(error)   # error resolved
    f1_improved    = improvement > 0.01
    success        = structural_fix or f1_improved

    # ── Action scoring ─────────────────────────────────────────────────────────
    if action not in ("STOP", ""):
        if success:
            delta = max(improvement, 0.01)          # credit structural fix too
            action_scores[action]  = action_scores.get(action, 0.0) + delta
            success_count[action]  = success_count.get(action, 0) + 1
        else:
            action_scores[action]  = action_scores.get(action, 0.0) - 0.01
            failure_count[action]  = failure_count.get(action, 0) + 1
            if failure_count[action] >= 2 and action not in failed_actions:
                failed_actions.append(action)
                print(f"[MEMORY]    🚫 '{action}' marked INEFFECTIVE (failed ≥2)")

    best_action = (max(action_scores, key=action_scores.get)
                   if action_scores else "")

    entry = {
        "attempt":     state["attempt"],
        "error":       error,
        "action":      action,
        "f1":          round(f1, 4),
        "improvement": round(improvement, 4),
        "success":     success,
        "confidence":  round(state.get("confidence", 0.0), 2),
        "root_cause":  state.get("root_cause", ""),
    }
    new_history = history + [entry]

    status = "✅ success" if success else "❌ no success"
    print(f"[MEMORY]    📝 Attempt {state['attempt']} — "
          f"action={action}  f1={f1:.4f}  Δ={improvement:+.4f}  {status}")
    if action_scores:
        print(f"[MEMORY]    📈 Scores: {action_scores}")

    return {**state,
            "history":              new_history,
            "attempt":              state["attempt"] + 1,
            "action_scores":        action_scores,
            "action_success_count": success_count,
            "action_failure_count": failure_count,
            "failed_actions":       failed_actions,
            "best_action":          best_action}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 7 — VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def _is_stagnating(history: List[Dict[str, Any]]) -> bool:
    if len(history) < STAGNATION_WINDOW:
        return False
    recent = [h["f1"] for h in history[-STAGNATION_WINDOW:]]
    deltas = [abs(recent[i+1] - recent[i]) for i in range(len(recent) - 1)]
    stagnating = all(d < STAGNATION_DELTA for d in deltas)
    if stagnating:
        print(f"[VALIDATOR] 📉 Stagnation — last {STAGNATION_WINDOW} F1s: "
              f"{[round(f, 4) for f in recent]}")
    return stagnating


def validator(state: PipelineState) -> PipelineState:
    """Intelligent, goal-aware stop/continue logic."""
    action    = state["action"]
    attempt   = state["attempt"]
    error     = state["error"]
    history   = state["history"]
    near_goal = state["near_goal"]
    f1        = state["f1"]
    dist      = state["goal_distance"]

    print(f"\n[VALIDATOR] — Attempt #{attempt}  action={action}  "
          f"F1={f1:.4f}  dist={dist:+.4f}  near_goal={near_goal}")

    # 1. Explicit STOP
    if action == "STOP":
        if "NO_SIGNAL" in error:
            reason = "NO_SIGNAL"
        elif not error:
            reason = "SUCCESS"
        else:
            reason = "LLM_DECISION"
        print(f"[VALIDATOR] 🛑 STOP — reason: {reason}")
        return {**state, "done": True}

    # 2. Goal reached during current loop
    if not error:
        print("[VALIDATOR] ✅ STOP — reason: GOAL_REACHED")
        return {**state, "done": True}

    # 3. Near-goal + stagnating → close-enough to accept
    if near_goal and _is_stagnating(history):
        print(f"[VALIDATOR] 🎯 STOP — reason: CLOSE_ENOUGH "
              f"(F1={f1:.4f}, distance={dist:+.4f}, stagnating)")
        return {**state, "done": True}

    # 4. Max attempts
    if attempt > MAX_ATTEMPTS:
        print(f"[VALIDATOR] 🛑 STOP — reason: MAX_ATTEMPTS ({MAX_ATTEMPTS})")
        return {**state, "done": True}

    # 5. Full stagnation
    if _is_stagnating(history):
        print("[VALIDATOR] 🛑 STOP — reason: STAGNATION")
        return {**state, "done": True}

    print(f"[VALIDATOR] 🔄 Continuing — attempt {attempt}/{MAX_ATTEMPTS}")
    return {**state, "done": False}

# ─────────────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def route_after_validator(state: PipelineState) -> str:
    return END if state["done"] else "detector"

# ─────────────────────────────────────────────────────────────────────────────
# BUILD LANGGRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """Construct and compile the 7-node LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    for name, fn in [
        ("detector",     detector),
        ("trainer",      trainer),
        ("evaluator",    evaluator),
        ("decision",     decision),
        ("fix_executor", fix_executor),
        ("memory_agent", memory_agent),
        ("validator",    validator),
    ]:
        graph.add_node(name, fn)

    graph.set_entry_point("detector")
    graph.add_edge("detector",     "trainer")
    graph.add_edge("trainer",      "evaluator")
    graph.add_edge("evaluator",    "decision")
    graph.add_edge("decision",     "fix_executor")
    graph.add_edge("fix_executor", "memory_agent")
    graph.add_edge("memory_agent", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {END: END, "detector": "detector"},
    )

    return graph.compile()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(failure_type: Optional[str] = None) -> Dict[str, Any]:
    print("=" * 65)
    print("  GOAL-AWARE SELF-HEALING ML PIPELINE  —  v3")
    print(f"  Failure injection: *** HIDDEN — agent must diagnose ***")
    print("=" * 65)

    df = generate_synthetic_data(n=500)
    if failure_type:
        df = inject_failures(df, failure_type)

    initial_state: PipelineState = {
        "df":                   df,
        "error":                "",
        "action":               "",
        "f1":                   0.0,
        "best_f1":              0.0,
        "attempt":              1,
        "history":              [],
        "done":                 False,
        "model_type":           "logistic",
        "model_params":         {},
        "root_cause":           "",
        "reasoning":            "",
        "confidence":           0.0,
        "goal_distance":        F1_THRESHOLD,
        "near_goal":            False,
        "action_scores":        {},
        "action_success_count": {},
        "action_failure_count": {},
        "failed_actions":       [],
        "best_action":          "",
    }

    result  = build_pipeline().invoke(initial_state)
    history = result["history"]

    print("\n" + "=" * 65)
    print("  GOAL-AWARE PIPELINE — FINAL REPORT")
    print("=" * 65)
    print(f"  Total attempts  : {result['attempt'] - 1}")
    print(f"  Final F1        : {result['f1']:.4f}")
    print(f"  Best F1         : {result['best_f1']:.4f}")
    print(f"  Goal            : {F1_THRESHOLD}")
    print(f"  Goal distance   : {result['goal_distance']:+.4f}")
    print(f"  Last root cause : {result.get('root_cause', 'N/A')}")
    print(f"  Best action     : {result['best_action']}")
    print(f"  Action scores   : {result['action_scores']}")
    print(f"  Ineffective     : {result['failed_actions']}")

    successful = [h for h in history if h.get("success")]
    failed_h   = [h for h in history if not h.get("success")]
    print(f"  Successful fixes: {[h['action'] for h in successful]}")
    print(f"  Failed fixes    : {[h['action'] for h in failed_h]}")

    print("\n  Detailed History:")
    for h in history:
        tag = "✅" if h.get("success") else "❌"
        print(f"    [{tag}] #{h['attempt']}: action={h['action']:22s} "
              f"f1={h['f1']:.4f}  Δ={h.get('improvement', 0):+.4f}  "
              f"conf={h.get('confidence', 0.0):.2f}  "
              f"cause='{h.get('root_cause', '')[:45]}'")
    print("=" * 65)

    return result

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Goal-Aware Self-Healing ML Pipeline v3 — LangGraph + Groq"
    )
    parser.add_argument(
        "--failure",
        choices=[
            "MISSING_COLUMN", "TYPE_MISMATCH", "LOW_PERFORMANCE",
            "COMPOUND", "PARTIAL_CORRUPTION", "NEAR_GOAL", "none",
        ],
        default="LOW_PERFORMANCE",
        help="Failure type to inject (hidden from agent — it must self-diagnose)",
    )
    args    = parser.parse_args()
    failure = None if args.failure == "none" else args.failure
    run_pipeline(failure_type=failure)