#This code is divided in three stages. 
#Front end --> HTML AND JAVASCRIPT FUNCTIONS
#BACK END --> PYTHON THAT COMUNUCATES WITH FRONT END
#AUXILIARY FUNCTIONS THAT HELP BACK END


#AFTER RUNNING CODE THE INTERFACE WILL OPEN IN BROWSER IN PAGE http://127.0.0.1:500



import os
import re
import csv
import time
import hashlib
import threading
import webbrowser
from io import BytesIO, StringIO
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from fpdf import FPDF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Import controls for RDKIT and PubChemy
RDKIT_AVAILABLE = False
RDKIT_IMPORT_ERROR = None
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Draw
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError as e:
    RDKIT_IMPORT_ERROR = f"ImportError: {e}"
except Exception as e:
    RDKIT_IMPORT_ERROR = f"{type(e).__name__}: {e}"

try:
    import rdkit as _rdkit_mod
    RDKIT_VERSION = getattr(_rdkit_mod, "__version__", "unknown")
except Exception:
    RDKIT_VERSION = None

try:
    import pubchempy as pcp
    PUBCHEMPY_AVAILABLE = True
except ImportError:
    PUBCHEMPY_AVAILABLE = False
except Exception:
    PUBCHEMPY_AVAILABLE = False


load_dotenv()

# Api Keys Load functions
def _load_api_keys() -> List[str]:
    keys: List[str] = []
    for i in range(1, 11):
        k = os.environ.get(f"GEMINI_API_KEY_{i}") or os.environ.get(f"API_KEY_{i}")
        if k and k.strip():
            keys.append(k.strip())
    if not keys:
        single = os.environ.get("API_KEY") or os.environ.get("GEMINI_API_KEY")
        if single and single.strip():
            keys.append(single.strip())
    return keys


API_KEYS: List[str] = _load_api_keys()

#Api keys control
if not API_KEYS:
    print("⚠️  WARNING: no API key found in .env")
else:
    print(f"✅ {len(API_KEYS)} API key(s) loaded for automatic rotation.")

if RDKIT_AVAILABLE:
    print(f"✅ RDKit cargado correctamente (versión {RDKIT_VERSION}).")
else:
    print(f"❌ RDKit NO disponible en este intérprete. Motivo: {RDKIT_IMPORT_ERROR}")
    import sys as _sys
    print(f"   Intérprete en uso: {_sys.executable}")
    print("   Comprueba que instalaste rdkit con ESE mismo python:")
    print("     python -m pip install rdkit")
    print("   Y verifica: python -c \"import rdkit; print(rdkit.__version__)\"")

if not PUBCHEMPY_AVAILABLE:
    print("ℹ️  pubchempy not available. Scenario H will not be able to resolve "
          "compound NAMES to structures (SMILES/InChI columns still work).")

# Configuration values
MODEL_NAME = "gemini-3.6-flash"

SEED = 42
TEST_FRAC = 0.20
MAX_CONTEXT_ROWS = 80
MAX_CONTEXT_SMILES_LEN = 160
MAX_QUERY_SMILES_LEN = 500
MAX_TRIES = 2
KEY_COOLDOWN_SECONDS = 60.0
MIN_INTERVAL_PER_KEY = float(os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", "8.0"))

FP_RADIUS = 2
FP_N_BITS = 1024

SCENARIOS = {
    "A": "No context - only SMILES from the 20%",
    "B": "80% context (RT + SMILES) → predicts the 20%",
    "C": "80% context (RT + SMILES) + optional paper → predicts the 20%",
    "D": "No RT context, only SMILES from the 20% + REQUIRED paper",
    "E": "Linear Regression - trained on 80%, predicts the 20% (local, no LLM)",
    "F": "Random Forest - trained on 80%, predicts the 20% (local, no LLM)",
    "G": "Manual SMILES - type any molecule, Gemini predicts its RT (optionally with 80% context from a sheet)",
    "H": "External / recent-study dataset - auto-detects the compound (SMILES, InChI or name) and RT columns, then runs an 80% context / 20% prediction evaluation (local, no LLM)",
}
LLM_SCENARIOS = ("A", "B", "C", "D", "G")
LOCAL_ML_SCENARIOS = ("E", "F")
MANUAL_SCENARIOS = ("G",)
EXTERNAL_SCENARIOS = ("H",)

EXTRA_COLS = [
    "column.name", "column.usp.code", "column.length", "column.particle.size",
    "column.temperature", "column.flowrate",
    "eluent.A.h2o", "eluent.A.pH",
    "eluent.B.meoh", "eluent.B.pH",
    "gradient.start.A", "gradient.end.B",
]

EXTERNAL_RT_CANDIDATES = [
    "rt", "rt_min", "retention_time", "retention_time_corrected",
    "correction_retention_time", "corrected_retention_time", "rt_corrected",
    "rt_sec", "rt_seconds",
]
EXTERNAL_SMILES_CANDIDATES = [
    "smiles", "pubchem.smiles.canonical", "canonical_smiles", "isomeric_smiles",
]
EXTERNAL_INCHI_CANDIDATES = ["inchi", "std_inchi", "standard_inchi"]
EXTERNAL_NAME_CANDIDATES = ["compound", "compound_name", "name", "molecule", "analyte"]
EXTERNAL_NUMERIC_CONTEXT_CANDIDATES = [
    "gradient_time", "ph_target", "ph_measured", "flow_rate", "temperature",
]
MAX_EXTERNAL_LLM_TEST = 500

state: Dict[str, Any] = {
    "xls_bytes": None,
    "paper_path": None,
    "paper_name": None,
    "scenario": None,
    "key_idx": 0,
    "exhausted_keys": {},
    "scenario_data": {s: [] for s in SCENARIOS},
    "processed_sheets": {s: set() for s in SCENARIOS},
    "key_last_call": {},
    "requests_made": 0,
    "last_gemini_error": None,
    "manual_predictions": [],
    "external_bytes": None,
    "external_filename": None,
    "external_is_csv": False,
    "trained_model_info": {},
    "manual_context_sheet": None,
    "manual_context_table": None,
    "manual_context_n_rows": 0,
    "manual_context_important_cols": [],
}


# Request count functions for each Api key
def _count_request():
    state["requests_made"] += 1


_throttle_lock = threading.Lock()

# Function that manages the turn of each key in the .env
def _wait_turn_for_key(idx: int):
    with _throttle_lock:
        last = state["key_last_call"].get(idx, 0.0)
        now = time.time()
        wait = MIN_INTERVAL_PER_KEY - (now - last)
        if wait > 0:
            time.sleep(wait)
        state["key_last_call"][idx] = time.time()

# Find column in excel or context functions
def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower().strip(): c for c in df.columns}
    for p in candidates:
        if p.lower().strip() in cols:
            return cols[p.lower().strip()]
    for c in df.columns:
        for p in candidates:
            if p.lower() in c.lower():
                return c
    return None

# Functions that prepares the dataframe
def prepare_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, List[str]]:
    col_rt = find_column(df, ["rt", "rt_min", "retention_time"])
    col_smiles = find_column(df, ["pubchem.smiles.canonical", "smiles"])
    if col_rt is None or col_smiles is None:
        raise RuntimeError("Could not find RT or SMILES columns in this sheet.")

    df = df.copy()
    df[col_rt] = pd.to_numeric(df[col_rt], errors="coerce")
    df[col_smiles] = df[col_smiles].astype(str)
    df = df.dropna(subset=[col_rt, col_smiles])
    df = df[df[col_smiles].str.strip().ne("")]
    df = df.reset_index(drop=True)

    important_cols = [c for c in EXTRA_COLS if find_column(df, [c]) is not None]
    important_cols = [find_column(df, [c]) for c in important_cols]
    important_cols = [c for c in important_cols if c is not None]

    return df, col_rt, col_smiles, important_cols

# External dataframe function for selection or visualization
def prepare_external_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, List[str], str, int]:
    col_rt = find_column(df, EXTERNAL_RT_CANDIDATES)
    if col_rt is None:
        raise RuntimeError("Could not find a retention time column in this dataset.")

    col_struct = find_column(df, EXTERNAL_SMILES_CANDIDATES)
    source_type = "smiles"

    if col_struct is None:
        col_inchi = find_column(df, EXTERNAL_INCHI_CANDIDATES)
        if col_inchi is not None:
            if not RDKIT_AVAILABLE:
                raise RuntimeError(
                    "This dataset stores structures as InChI. Converting them "
                    "to SMILES requires RDKit, which is not installed."
                )
            source_type = "inchi"
            col_struct = col_inchi

    if col_struct is None:
        col_name = find_column(df, EXTERNAL_NAME_CANDIDATES)
        if col_name is not None:
            source_type = "name"
            col_struct = col_name

    if col_struct is None:
        raise RuntimeError(
            "Could not find a SMILES, InChI or compound name column in this dataset."
        )

    work = df.copy()
    n_input = len(work)
    work[col_rt] = pd.to_numeric(work[col_rt], errors="coerce")

    if source_type == "smiles":
        work["_smiles_"] = work[col_struct].astype(str).str.strip()

    elif source_type == "inchi":
        cache: Dict[str, Optional[str]] = {}
        # chemical conversion function
        def _inchi_to_smiles(value: Any) -> Optional[str]:
            text = str(value).strip()
            if not text:
                return None
            if text in cache:
                return cache[text]
            try:
                mol = Chem.MolFromInchi(text)
                smi = Chem.MolToSmiles(mol) if mol is not None else None
            except Exception:
                smi = None
            cache[text] = smi
            return smi

        work["_smiles_"] = work[col_struct].apply(_inchi_to_smiles)

    else:
        if not PUBCHEMPY_AVAILABLE:
            raise RuntimeError(
                "This dataset only provides compound NAMES, not SMILES or "
                "InChI. Install pubchempy (pip install pubchempy) and make "
                "sure you have an internet connection so names can be "
                "resolved to structures via PubChem."
            )
        cache = {}
        #Chemical conversion function
        def _name_to_smiles(value: Any) -> Optional[str]:
            text = str(value).strip()
            if not text:
                return None
            if text in cache:
                return cache[text]
            smi = None
            try:
                hits = pcp.get_compounds(text, "name")
                if hits:
                    smi = hits[0].canonical_smiles
            except Exception:
                smi = None
            cache[text] = smi
            return smi

        work["_smiles_"] = work[col_struct].apply(_name_to_smiles)

    work = work.dropna(subset=[col_rt, "_smiles_"])
    work = work[work["_smiles_"].astype(str).str.strip().ne("")]
    work = work.reset_index(drop=True)

    if len(work) == 0:
        raise RuntimeError(
            "No valid rows remained after detecting RT and compound "
            "structure. Check that the compound column can be resolved to "
            "structures and that the RT column has numeric values."
        )

    important_cols = [c for c in EXTERNAL_NUMERIC_CONTEXT_CANDIDATES
                       if find_column(work, [c]) is not None]
    important_cols = [find_column(work, [c]) for c in important_cols]
    important_cols = [c for c in important_cols if c is not None]

    n_dropped = n_input - len(work)

    return work, col_rt, "_smiles_", important_cols, source_type, n_dropped

# Function that splits the page in 80% context, 20% test. Min 3 compounds needed
def split_page(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 2:
        raise RuntimeError(f"This sheet only has {n} valid compound(s) (at least 2 needed).")
    df_shuf = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_test = max(1, round(n * TEST_FRAC))
    n_test = min(n_test, n - 1)
    ctx = df_shuf.iloc[: n - n_test].reset_index(drop=True)
    test = df_shuf.iloc[n - n_test:].reset_index(drop=True)
    return ctx, test

# Function that builds the context table for Gemini
def build_context_table(ctx: pd.DataFrame, col_rt: str, col_smiles: str,
                         important_cols: List[str]) -> str:
    lines = []
    for _, r in ctx.head(MAX_CONTEXT_ROWS).iterrows():
        smi = str(r[col_smiles]).strip()[:MAX_CONTEXT_SMILES_LEN]
        rt = float(r[col_rt])
        extras = []
        for c in important_cols:
            val = r[c]
            if pd.notna(val):
                if isinstance(val, float):
                    extras.append(f"{c}={val:.3f}" if val % 1 else f"{c}={int(val)}")
                else:
                    extras.append(f"{c}={val}")
        suffix = (" | " + " | ".join(extras)) if extras else ""
        lines.append(f"{rt:.3f}\t{smi}{suffix}")
    return "\n".join(lines)

#Chemical conversion for linear regresion and random forest.
def smiles_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_N_BITS)
    arr = np.zeros((FP_N_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Fallback function when RDKKIT not installed
def _simple_smiles_descriptors(smiles: str) -> np.ndarray:
    s = smiles.strip()
    n_C = s.count("C") + s.count("c")
    n_N = s.count("N") + s.count("n")
    n_O = s.count("O") + s.count("o")
    n_S = s.count("S") + s.count("s")
    n_halogen = s.count("F") + s.count("Cl") + s.count("Br") + s.count("I")
    n_double = s.count("=")
    n_triple = s.count("#")
    n_ring = sum(s.count(str(d)) for d in range(1, 10))
    n_branch = s.count("(")
    n_charge = s.count("+") + s.count("-")
    n_stereo = s.count("@")
    return np.array(
        [len(s), n_C, n_N, n_O, n_S, n_halogen, n_double, n_triple,
         n_ring, n_branch, n_charge, n_stereo],
        dtype=np.float32,
    )

#Fallback when no RDKIT
def _smiles_hash_fingerprint(smiles: str, n_bits: int) -> np.ndarray:
    vec = np.zeros(n_bits, dtype=np.float32)
    s = smiles.strip()
    for n in (1, 2, 3):
        for i in range(len(s) - n + 1):
            token = s[i:i + n]
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % n_bits] += 1.0
    return vec

# Smiles to number for linear regresion and random forest.
def molecular_features(smiles: str) -> np.ndarray:
    if RDKIT_AVAILABLE:
        fp = smiles_to_fingerprint(smiles)
        if fp is not None:
            return fp
    descriptors = _simple_smiles_descriptors(smiles)
    hash_bits = FP_N_BITS - len(descriptors)
    hash_fp = _smiles_hash_fingerprint(smiles, n_bits=hash_bits)
    return np.concatenate([hash_fp, descriptors])

#Convert dataframe in numeral matriz for training
def build_feature_matrix(rows: pd.DataFrame, col_smiles: str,
                          important_cols: List[str]) -> np.ndarray:
    n = len(rows)
    n_extra = len(important_cols)
    X = np.zeros((n, FP_N_BITS + n_extra), dtype=np.float32)
    for i, (_, row) in enumerate(rows.iterrows()):
        smi = str(row[col_smiles]).strip()
        X[i, :FP_N_BITS] = molecular_features(smi)
        for j, c in enumerate(important_cols):
            val = row[c]
            if pd.notna(val):
                try:
                    X[i, FP_N_BITS + j] = float(val)
                except (TypeError, ValueError):
                    X[i, FP_N_BITS + j] = 0.0
    return X

# Training functions for LR and RF
def run_local_model(scenario: str, df: pd.DataFrame, col_rt: str, col_smiles: str,
                     important_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[float], Any, Dict[str, float]]:
    ctx, test = split_page(df)
    X_train = build_feature_matrix(ctx, col_smiles, important_cols)
    y_train = ctx[col_rt].to_numpy(dtype=float)
    X_test = build_feature_matrix(test, col_smiles, important_cols)

    if scenario == "E":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    col_std = {}
    for j, c in enumerate(important_cols):
        col_std[c] = float(np.std(X_train[:, FP_N_BITS + j]))

    return ctx, test, [float(p) for p in preds], model, col_std

#Name selection function
def _feature_names(important_cols: List[str]) -> List[str]:
    return [f"fp_{i}" for i in range(FP_N_BITS)] + list(important_cols)

#Function to save the values plot
def _save_plot_png(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# Gemini configuration function
def configure_gemini(idx: int):
    if not API_KEYS:
        raise RuntimeError("No API keys configured in .env")
    idx = idx % len(API_KEYS)
    return genai.Client(api_key=API_KEYS[idx])

# Function to search ena check if the next key is availeable
def _next_available_key(current: int) -> Optional[int]:
    n = len(API_KEYS)
    now = time.time()
    for off in range(1, n + 1):
        cand = (current + off) % n
        if cand not in state["exhausted_keys"]:
            return cand
    cooldown_candidates = [
        (idx, ts) for idx, ts in state["exhausted_keys"].items()
        if (now - ts) >= KEY_COOLDOWN_SECONDS
    ]
    if cooldown_candidates:
        cooldown_candidates.sort(key=lambda x: x[1])
        chosen_idx = cooldown_candidates[0][0]
        state["exhausted_keys"].pop(chosen_idx, None)
        return chosen_idx
    return None

#Function to read the response from Gemini
def read_text_safe(resp: Any) -> str:
    try:
        if getattr(resp, "candidates", None):
            for i, c in enumerate(resp.candidates):
                fr = getattr(c, "finish_reason", None)
                print(f"🔍 candidate[{i}].finish_reason = {fr}")
    except Exception as e:
        print(f"🔍 finish_reason access failed: {e}")

    try:
        text = getattr(resp, "text", None)
        if text:
            return text.strip()
    except Exception:
        pass
    try:
        if getattr(resp, "candidates", None):
            parts = getattr(resp.candidates[0].content, "parts", []) or []
            return "\n".join(p.text for p in parts if getattr(p, "text", "")).strip()
    except Exception:
        pass
    return ""

# Check api keys
def _is_key_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(t in s for t in [
        "429", "quota", "resource_exhausted", "403", "permission_denied",
        "api_key_invalid", "invalid api key", "api key not valid", "401",
    ])

#Gemini configuration values
MAX_OUTPUT_TOKENS_BASE = 4000
MAX_OUTPUT_TOKENS_PER_ROW = 200
MAX_OUTPUT_TOKENS_CAP = 32000

#Function to configure the max token for the response depending on the amount o compounds predicted
def _max_output_tokens_for(n_test: int) -> int:
    return max(MAX_OUTPUT_TOKENS_BASE,
               min(MAX_OUTPUT_TOKENS_CAP,
                   MAX_OUTPUT_TOKENS_BASE + n_test * MAX_OUTPUT_TOKENS_PER_ROW))

#Promt biulder function
def build_page_prompt(scenario: str, train_table: Optional[str], test: pd.DataFrame,
                       col_smiles: str, important_cols: List[str]) -> str:
    n = len(test)
    query_lines = []
    for i, (_, row) in enumerate(test.iterrows(), start=1):
        smi = str(row[col_smiles]).strip()[:MAX_QUERY_SMILES_LEN]
        if scenario in ("B", "C"):
            present = [c for c in important_cols if c in row.index and pd.notna(row[c])]
            cond = (" | " + " | ".join(f"{c}={row[c]}" for c in present)) if present else ""
            query_lines.append(f"{i}\t{smi}{cond}")
        else:
            query_lines.append(f"{i}\t{smi}")
    query_block = "\n".join(query_lines)

    header = (
        "You are a retention-time (RT) calculator. For EACH molecule listed "
        "at the end, output ONE predicted RT value in minutes.\n\n"
        "ABSOLUTE OUTPUT RULES (violating any of these = task failure):\n"
        f"  1. Output EXACTLY {n} lines. No more, no fewer.\n"
        "  2. Each line contains ONLY a number with 3 decimals (e.g. 4.521).\n"
        "  3. No explanations, no labels, no units, no blank lines, no headers.\n"
        "  4. It is FORBIDDEN to refuse, to say 'I cannot', to ask for more "
        "info, or to write anything other than the numbers.\n"
        "  5. Do NOT repeat the same number for different molecules.\n"
        "  6. Do NOT stop halfway. You must produce all the lines.\n\n"
    )

    if scenario == "B":
        header += (
            "CONTEXT - pairs (RT\\tSMILES\\tconditions) measured in the SAME "
            "chromatographic system. Learn the scale and pattern from them.\n"
            f"--- EXAMPLES START ---\n{train_table}\n--- EXAMPLES END ---\n\n"
        )
    elif scenario == "C":
        header += (
            "CONTEXT - pairs (RT\\tSMILES\\tconditions) from the SAME system, "
            "plus an attached paper that MAY describe the method. If the paper "
            "does not describe a concrete method, ignore it.\n"
            f"--- EXAMPLES START ---\n{train_table}\n--- EXAMPLES END ---\n\n"
        )
    elif scenario == "D":
        header += (
            "CONTEXT - an attached scientific paper. It MAY be purely "
            "theoretical and NOT describe a concrete method. If so, IGNORE it "
            "and estimate each RT from the molecule's chemistry (size, logP, "
            "polarity, charge, aromatic rings). Larger / more hydrophobic "
            "molecules → larger RT (typical range 0.5–30 min). Smaller / "
            "polar / ionised molecules → smaller RT.\n\n"
        )
    else:
        header += (
            "CONTEXT - none. Estimate each RT from the molecule's chemistry "
            "(size, logP, polarity, charge, aromatic rings). Larger / more "
            "hydrophobic molecules → larger RT (typical range 0.5–30 min). "
            "Smaller / polar / ionised molecules → smaller RT.\n\n"
        )

    footer = (
        f"Now output exactly {n} lines, one per molecule below, in the SAME "
        f"order, nothing else. Remember: {n} lines, not fewer.\n\n"
        f"--- MOLECULES ({n}) ---\n{query_block}\n--- END MOLECULES ---\n\n"
        f"YOUR {n} LINES:\n"
    )

    return header + footer

#Promt builder funtion
def build_single_smiles_prompt(smiles: str, train_table: Optional[str] = None,
                                context_sheet: Optional[str] = None,
                                context_n_rows: int = 0) -> str:
    header = (
        "You are a retention-time (RT) calculator. Estimate the RT of the "
        "following molecule as accurately as you can.\n\n"
        "ABSOLUTE OUTPUT RULES (violating any of these = task failure):\n"
        "  1. Output EXACTLY 1 line.\n"
        "  2. The line contains ONLY a number with 3 decimals (e.g. 4.521).\n"
        "  3. No explanations, no labels, no units, no blank lines, no headers.\n"
        "  4. It is FORBIDDEN to refuse, to say 'I cannot', or to ask for more info.\n\n"
    )

    if train_table:
        header += (
            f"CONTEXT - pairs (RT\\tSMILES\\tconditions) measured in the SAME "
            f"chromatographic system (sheet \"{context_sheet}\", {context_n_rows} "
            f"rows provided). Learn the scale and pattern from them.\n"
            f"--- EXAMPLES START ---\n{train_table}\n--- EXAMPLES END ---\n\n"
        )
    else:
        header += (
            "CONTEXT - none. Estimate the RT from the molecule's chemistry "
            "(size, logP, polarity, charge, aromatic rings). Larger / more "
            "hydrophobic molecules → larger RT (typical range 0.5–30 min). "
            "Smaller / polar / ionised molecules → smaller RT.\n\n"
        )

    header += (
        f"--- MOLECULE ---\n{smiles}\n--- END MOLECULE ---\n\n"
        "YOUR 1 LINE:\n"
    )
    return header

#Extracts numbers from response no matter the format.
def _extract_all_numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text or "")]

# Parses gemini response
def parse_predictions(text: str, n_expected: int) -> List[float]:
    if not text or not text.strip():
        raise ValueError(
            f"Gemini devolvió una respuesta vacía. Se esperaban {n_expected} "
            f"números."
        )

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    numbers: List[float] = []
    for l in lines:
        m = re.match(r"^\s*[-+]?\d+(?:\.\d+)?\s*$", l)
        if m:
            numbers.append(float(m.group(0)))
    if len(numbers) == n_expected:
        return numbers

    all_numbers = _extract_all_numbers(text)
    if len(all_numbers) == n_expected:
        return all_numbers
    if len(all_numbers) > n_expected:
        return all_numbers[:n_expected]

    raise ValueError(
        f"Gemini devolvió {len(all_numbers)} número(s) y se esperaban "
        f"{n_expected}. Respuesta cruda (primeros 500 chars): "
        f"{(text or '')[:500]!r}"
    )

#Sends data to gemini
def _send_page_with_key(idx: int, prompt_text: str, attach_paper: bool,
                         max_out: int, n_expected: int = 0) -> str:
    last_err = None
    for attempt in range(MAX_TRIES):
        try:
            client = configure_gemini(idx)
            parts: List[Any] = []
            if attach_paper:
                handle = client.files.upload(
                    file=state["paper_path"],
                    config=genai_types.UploadFileConfig(mime_type="application/pdf"),
                )
                parts.append(handle)
            parts.append(prompt_text)

            _wait_turn_for_key(idx)
            _count_request()

            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=parts,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_out,
                ),
            )
            text = read_text_safe(resp)

            try:
                fr_str = ""
                if getattr(resp, "candidates", None):
                    fr = getattr(resp.candidates[0], "finish_reason", None)
                    fr_str = str(fr).upper() if fr else ""
                if "MAX_TOKENS" in fr_str and len(_extract_all_numbers(text)) < n_expected:
                    doubled = min(max_out * 2, MAX_OUTPUT_TOKENS_CAP)
                    print(f"⚠️  finish_reason = MAX_TOKENS. Reintentando con "
                          f"{doubled} tokens...")
                    _wait_turn_for_key(idx)
                    _count_request()
                    resp2 = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=parts,
                        config=genai_types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=doubled,
                        ),
                    )
                    text2 = read_text_safe(resp2)
                    if len(_extract_all_numbers(text2)) > len(_extract_all_numbers(text)):
                        text = text2
                        print(f"✅ Reintento OK: "
                              f"{len(_extract_all_numbers(text2))}/{n_expected} números.")
            except Exception as e:
                print(f"⚠️  Reintento por MAX_TOKENS falló: {e}")

            if text and n_expected > 0 and len(_extract_all_numbers(text)) < n_expected:
                print(f"⚠️  Primera respuesta insuficiente "
                      f"({len(_extract_all_numbers(text))}/{n_expected} números). "
                      f"Lanzando rescate...")
                rescue_prompt = (
                    f"Output exactly {n_expected} lines. Each line must contain "
                    f"ONLY a number with 3 decimals. No other text. Just "
                    f"{n_expected} lines of numbers. Do not stop early."
                )
                _wait_turn_for_key(idx)
                _count_request()
                try:
                    rescue_resp = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=[rescue_prompt],
                        config=genai_types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=max_out,
                        ),
                    )
                    rescue_text = read_text_safe(rescue_resp)
                    if rescue_text and len(_extract_all_numbers(rescue_text)) >= n_expected:
                        return rescue_text
                except Exception as rescue_err:
                    print(f"⚠️  Rescate falló: {rescue_err}")

            if text:
                return text
            time.sleep(1.5)
        except Exception as e:
            last_err = e
            if _is_key_error(e):
                raise
            wait = min(8.0, 1.0 + attempt * 0.8)
            time.sleep(wait)
    raise last_err or RuntimeError("Gemini did not return a response.")

#Send data to gemini
def send_page(prompt_text: str, attach_paper: bool, n_test: int) -> str:
    max_out = _max_output_tokens_for(n_test)
    rotations = 0
    max_rotations = max(len(API_KEYS), 1) + 1
    idx = state["key_idx"]
    while True:
        try:
            text = _send_page_with_key(idx, prompt_text, attach_paper, max_out,
                                        n_expected=n_test)
            state["key_idx"] = idx
            return text
        except Exception as e:
            if not _is_key_error(e) or rotations >= max_rotations:
                raise
            state["exhausted_keys"][idx] = time.time()
            new_idx = _next_available_key(idx)
            if new_idx is None:
                raise RuntimeError(
                    f"Todas las {len(API_KEYS)} claves están agotadas o han "
                    f"fallado en los últimos {int(KEY_COOLDOWN_SECONDS)}s. "
                    f"Espera o añade más claves al .env."
                )
            idx = new_idx
            state["key_idx"] = idx
            rotations += 1

#Functions that constructs and calculates the statistical data
def _stats_from_data(data: List[dict]) -> dict:
    n = len(data)
    if n == 0:
        return {"n": 0}

    reals = np.array([d["rt_real"] for d in data], dtype=float)
    preds = np.array([d["rt_pred"] for d in data], dtype=float)
    err = preds - reals
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    medae = float(np.median(abs_err))

    ss_res = float(np.sum((reals - preds) ** 2))
    ss_tot = float(np.sum((reals - np.mean(reals)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    mre = float(np.mean(abs_err / np.abs(reals))) if np.all(reals != 0) else float("nan")

    if n >= 2 and np.std(reals) > 0 and np.std(preds) > 0:
        slope, intercept = np.polyfit(reals, preds, 1)
        pearson = float(np.corrcoef(reals, preds)[0, 1])
        rank_r = pd.Series(reals).rank().to_numpy()
        rank_p = pd.Series(preds).rank().to_numpy()
        spearman = float(np.corrcoef(rank_r, rank_p)[0, 1])
    else:
        slope = intercept = pearson = spearman = float("nan")

    return {
        "n": n, "mae": mae, "rmse": rmse, "r2": r2, "mre": mre, "medae": medae,
        "slope": float(slope), "intercept": float(intercept),
        "pearson": pearson, "spearman": spearman,
    }

#Values for pdf format results
_STAT_LABELS_PDF = {
    "mae": "MAE", "rmse": "RMSE", "r2": "R2 (coefficient of determination)",
    "mre": "MRE (mean relative error)", "medae": "MedAE (median absolute error)",
    "slope": "Regression slope", "intercept": "Regression intercept",
    "pearson": "Pearson correlation", "spearman": "Spearman correlation",
}

#PDF format
def _pdf_safe(text: str) -> str:
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2192": "->",
        "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")

# Statistical number to texto or pdf
def _fmt_stat(key: str, value: Optional[float]) -> str:
    if value is None or value != value:
        return "-"
    if key == "mre":
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"

# Builds stats in pdf
def build_stats_pdf(scenario: str, description: str, stats: dict,
                     processed_sheets: List[str]) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"RT Predictor - Scenario {scenario} Statistics", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 6, _pdf_safe(description))
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    n = stats.get("n", 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Accumulated points: {n}", ln=True)
    sheets_txt = ", ".join(processed_sheets) if processed_sheets else "-"
    pdf.multi_cell(0, 7, _pdf_safe(f"Sheets processed ({len(processed_sheets)}): {sheets_txt}"))
    pdf.ln(4)

    if n == 0:
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 8, "No predictions accumulated yet in this scenario.", ln=True)
    else:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Metrics", ln=True)
        pdf.ln(1)

        col1_w, col2_w = 100, 70
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(col1_w, 8, "Metric", border=1, fill=True)
        pdf.cell(col2_w, 8, "Value", border=1, fill=True, ln=True)

        pdf.set_font("Helvetica", "", 11)
        for key, label in _STAT_LABELS_PDF.items():
            pdf.cell(col1_w, 8, label, border=1)
            pdf.cell(col2_w, 8, _fmt_stat(key, stats.get(key)), border=1, ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 5, "Generated by RT Predictor.", ln=True)

    out = pdf.output(dest="S")
    return bytes(out)

#BACK END FUNCTIONS TO UPLOAD EXCEL
#UPLOAD PAPER, SELECT SCENARIO, DOWNLOAD PDF OR EVERY OTHER ACTION.
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return HTML


@app.route("/api/upload_excel", methods=["POST"])
def upload_excel():
    f = request.files["file"]
    state["xls_bytes"] = f.read()
    xls = pd.ExcelFile(BytesIO(state["xls_bytes"]))
    return jsonify({"ok": True, "sheets": xls.sheet_names})


@app.route("/api/upload_paper", methods=["POST"])
def upload_paper():
    import tempfile
    f = request.files["file"]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(f.read())
    tmp.flush()
    tmp.close()
    state["paper_path"] = tmp.name
    state["paper_name"] = f.filename
    return jsonify({"ok": True, "name": f.filename})


@app.route("/api/set_scenario", methods=["POST"])
def set_scenario():
    scenario = (request.json or {}).get("scenario")
    if scenario not in SCENARIOS:
        return jsonify({"error": "Invalid scenario. Use A, B, C, D, E, F, G or H."}), 400
    state["scenario"] = scenario
    return jsonify({
        "ok": True,
        "scenario": scenario,
        "description": SCENARIOS[scenario],
        "requires_paper": scenario == "D",
        "allows_paper": scenario in ("C", "D"),
        "is_local_ml": scenario in LOCAL_ML_SCENARIOS,
        "is_manual": scenario in MANUAL_SCENARIOS,
        "is_external": scenario in EXTERNAL_SCENARIOS,
        "total_accumulated": len(state["scenario_data"][scenario]),
        "processed_sheets": sorted(state["processed_sheets"][scenario]),
    })


@app.route("/api/sheet_rows", methods=["POST"])
def sheet_rows():
    sheet = (request.json or {}).get("sheet")
    if not sheet:
        return jsonify({"error": "Missing 'sheet'."}), 400
    if not state["xls_bytes"]:
        return jsonify({"error": "No Excel file loaded."}), 400
    try:
        raw = pd.read_excel(BytesIO(state["xls_bytes"]), sheet_name=sheet)
        df, col_rt, col_smiles, _ = prepare_df(raw)
        ctx, test = split_page(df)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    scenario = state.get("scenario")
    already_processed = scenario in SCENARIOS and sheet in state["processed_sheets"][scenario]

    return jsonify({
        "ok": True,
        "total_valid": len(df),
        "n_context": len(ctx),
        "n_prediction": len(test),
        "already_processed": already_processed,
    })


@app.route("/api/run_page", methods=["POST"])
def run_page():
    body = request.json or {}
    sheet = body.get("sheet")
    force = bool(body.get("force"))
    scenario = state.get("scenario")

    if not scenario:
        return jsonify({"error": "First choose a scenario (A, B, C, D, E, F, G or H)."}), 400
    if scenario in MANUAL_SCENARIOS:
        return jsonify({
            "error": "Scenario G works differently: use the manual SMILES panel."
        }), 400
    if scenario in EXTERNAL_SCENARIOS:
        return jsonify({
            "error": "Scenario H works differently: upload the external dataset "
                     "and use '/api/run_external' instead of running a page."
        }), 400
    if not sheet:
        return jsonify({"error": "Missing 'sheet'."}), 400
    if not state["xls_bytes"]:
        return jsonify({"error": "No Excel file loaded."}), 400
    if scenario == "D" and not state["paper_path"]:
        return jsonify({"error": "Scenario D requires uploading a paper first."}), 400
    if scenario in LLM_SCENARIOS and not API_KEYS:
        return jsonify({"error": "No API keys configured in .env."}), 500

    if sheet in state["processed_sheets"][scenario] and not force:
        return jsonify({
            "warning": "already_processed",
            "message": f"Sheet '{sheet}' was already processed in scenario {scenario}. "
                       f"Resending it would duplicate those points in the statistics.",
        }), 409

    try:
        raw = pd.read_excel(BytesIO(state["xls_bytes"]), sheet_name=sheet)
        df, col_rt, col_smiles, important_cols = prepare_df(raw)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    if scenario in LOCAL_ML_SCENARIOS:
        try:
            ctx, test, predictions, trained_model, col_std = run_local_model(scenario, df, col_rt, col_smiles, important_cols)
            state["trained_model_info"].setdefault(scenario, {})[sheet] = {
                "model": trained_model, "important_cols": important_cols, "col_std": col_std,
                "train_smiles": ctx[col_smiles].astype(str).tolist(),
            }
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {e}"}), 500
    else:
        try:
            ctx, test = split_page(df)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400

        train_table = None
        if scenario in ("B", "C"):
            train_table = build_context_table(ctx, col_rt, col_smiles, important_cols)

        prompt_text = build_page_prompt(scenario, train_table, test, col_smiles, important_cols)
        attach_paper = (scenario == "D") or (scenario == "C" and bool(state["paper_path"]))

        try:
            response_text = send_page(prompt_text, attach_paper, len(test))
            print(f"📨 Respuesta de Gemini (primeros 300 chars): "
                  f"{response_text[:300]!r}")
            print(f"📨 Total números extraídos: "
                  f"{len(_extract_all_numbers(response_text))}/{len(test)}")
            predictions = parse_predictions(response_text, len(test))
        except Exception as e:
            state["last_gemini_error"] = str(e)
            return jsonify({
                "error": (
                    f"Gemini no devolvió predicciones válidas para la hoja "
                    f"'{sheet}' (escenario {scenario}).\n\n"
                    f"Detalle: {e}\n\n"
                    f"Sugerencias:\n"
                    f"  1. Reintenta la hoja (a veces el modelo se corta "
                    f"transitoriamente).\n"
                    f"  2. Comprueba la cuota: https://ai.dev/rate-limit\n"
                    f"  3. Si tienes más claves, añádelas al .env.\n"
                    f"  4. Revisa la consola del servidor para ver la "
                    f"respuesta cruda de Gemini."
                )
            }), 502

    if force:
        state["scenario_data"][scenario] = [
            d for d in state["scenario_data"][scenario] if d["sheet"] != sheet
        ]

    results = []
    for (_, row), rt_pred in zip(test.iterrows(), predictions):
        smi = str(row[col_smiles]).strip()
        rt_real = float(row[col_rt])
        item = {
            "scenario": scenario,
            "sheet": sheet,
            "smiles": smi,
            "rt_real": rt_real,
            "rt_pred": rt_pred,
            "abs_err": abs(rt_pred - rt_real),
        }
        results.append(item)
        state["scenario_data"][scenario].append(item)

    state["processed_sheets"][scenario].add(sheet)

    return jsonify({
        "ok": True,
        "scenario": scenario,
        "sheet": sheet,
        "n_context": len(ctx),
        "n_test": len(test),
        "results": results,
        "total_accumulated": len(state["scenario_data"][scenario]),
        "processed_sheets": sorted(state["processed_sheets"][scenario]),
    })


@app.route("/api/set_manual_context", methods=["POST"])
def set_manual_context():
    body = request.json or {}
    sheet = body.get("sheet")
    use_context = bool(body.get("use_context"))

    if not use_context:
        state["manual_context_sheet"] = None
        state["manual_context_table"] = None
        state["manual_context_n_rows"] = 0
        state["manual_context_important_cols"] = []
        return jsonify({"ok": True, "use_context": False})

    if not sheet:
        return jsonify({"error": "Missing 'sheet'."}), 400
    if not state["xls_bytes"]:
        return jsonify({"error": "No Excel file loaded."}), 400

    try:
        raw = pd.read_excel(BytesIO(state["xls_bytes"]), sheet_name=sheet)
        df, col_rt, col_smiles, important_cols = prepare_df(raw)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    ctx, _ = split_page(df)
    table = build_context_table(ctx, col_rt, col_smiles, important_cols)

    state["manual_context_sheet"] = sheet
    state["manual_context_table"] = table
    state["manual_context_n_rows"] = min(len(ctx), MAX_CONTEXT_ROWS)
    state["manual_context_important_cols"] = important_cols

    return jsonify({
        "ok": True,
        "use_context": True,
        "sheet": sheet,
        "n_context": min(len(ctx), MAX_CONTEXT_ROWS),
        "important_cols": important_cols,
    })


@app.route("/api/manual_context_status")
def manual_context_status():
    return jsonify({
        "ok": True,
        "use_context": state["manual_context_table"] is not None,
        "sheet": state["manual_context_sheet"],
        "n_context": state["manual_context_n_rows"],
    })


@app.route("/api/predict_manual", methods=["POST"])
def predict_manual():
    body = request.json or {}
    smiles = (body.get("smiles") or "").strip()

    if not smiles:
        return jsonify({"error": "Missing SMILES."}), 400
    if not API_KEYS:
        return jsonify({"error": "No API keys configured in .env."}), 500

    train_table = state["manual_context_table"]
    context_sheet = state["manual_context_sheet"]
    context_n_rows = state["manual_context_n_rows"]

    prompt_text = build_single_smiles_prompt(
        smiles,
        train_table=train_table,
        context_sheet=context_sheet,
        context_n_rows=context_n_rows,
    )

    try:
        response_text = send_page(prompt_text, attach_paper=False, n_test=1)
        preds = parse_predictions(response_text, 1)
        pred = float(preds[0])
    except Exception as e:
        state["last_gemini_error"] = str(e)
        return jsonify({
            "error": (
                f"Gemini no devolvió una predicción válida para este SMILES.\n\n"
                f"Detalle: {e}\n\n"
                f"Sugerencias:\n"
                f"  1. Reintenta (a veces el modelo se corta transitoriamente).\n"
                f"  2. Comprueba la cuota: https://ai.dev/rate-limit\n"
                f"  3. Si tienes más claves, añádelas al .env."
            )
        }), 502

    item = {
        "smiles": smiles,
        "rt_pred": pred,
        "engine": "gemini",
        "model": MODEL_NAME,
        "used_context": bool(train_table),
        "context_sheet": context_sheet,
    }
    state["manual_predictions"].append(item)

    return jsonify({"ok": True, "prediction": item, "total": len(state["manual_predictions"])})


@app.route("/api/manual_download_csv")
def manual_download_csv():
    data = state["manual_predictions"]
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["smiles", "rt_pred", "engine", "model", "used_context", "context_sheet"])
    for d in data:
        writer.writerow([
            d["smiles"], f'{d["rt_pred"]:.4f}',
            d.get("engine", "gemini"), d.get("model", MODEL_NAME),
            d.get("used_context", False), d.get("context_sheet") or "",
        ])
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=manual_predictions.csv"},
    )


@app.route("/api/manual_reset", methods=["POST"])
def manual_reset():
    state["manual_predictions"] = []
    return jsonify({"ok": True})


def _read_external_table(sheet: Optional[str] = None) -> pd.DataFrame:
    raw = state.get("external_bytes")
    if not raw:
        raise RuntimeError("No external dataset uploaded.")

    if state.get("external_is_csv"):
        text = raw.decode("utf-8", errors="replace")
        sample = text[:4096]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
        except Exception:
            delimiter = ";" if sample.count(";") > sample.count(",") else ","
        return pd.read_csv(StringIO(text), sep=delimiter)

    return pd.read_excel(BytesIO(raw), sheet_name=sheet if sheet else 0)


@app.route("/api/upload_external", methods=["POST"])
def upload_external():
    f = request.files["file"]
    filename = f.filename or "external_dataset"
    raw = f.read()
    is_csv = filename.lower().endswith(".csv") or filename.lower().endswith(".tsv")

    state["external_bytes"] = raw
    state["external_filename"] = filename
    state["external_is_csv"] = is_csv

    sheets: List[str] = []
    if not is_csv:
        try:
            xls = pd.ExcelFile(BytesIO(raw))
            sheets = xls.sheet_names
        except Exception as e:
            return jsonify({"error": f"Could not read this Excel file: {e}"}), 400

    return jsonify({"ok": True, "is_csv": is_csv, "sheets": sheets, "filename": filename})


@app.route("/api/external_preview", methods=["POST"])
def external_preview():
    body = request.json or {}
    sheet = body.get("sheet")
    try:
        raw_df = _read_external_table(sheet)
        df, col_rt, col_smiles, important_cols, source_type, n_dropped = prepare_external_df(raw_df)
        ctx, test = split_page(df)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Could not parse this dataset: {e}"}), 400

    return jsonify({
        "ok": True,
        "detected_rt_column": col_rt,
        "detected_compound_column_type": source_type,
        "important_cols": important_cols,
        "total_valid": len(df),
        "n_dropped": n_dropped,
        "n_context": len(ctx),
        "n_prediction": len(test),
    })


@app.route("/api/run_external", methods=["POST"])
def run_external():
    body = request.json or {}
    sheet = body.get("sheet")
    algorithm = (body.get("algorithm") or "rf").lower()
    force = bool(body.get("force"))
    scenario = "H"
    sheet_key = sheet or state.get("external_filename") or "external_dataset"

    if sheet_key in state["processed_sheets"][scenario] and not force:
        return jsonify({
            "warning": "already_processed",
            "message": f"'{sheet_key}' was already processed in scenario H. "
                       f"Resending it would duplicate those points in the statistics.",
        }), 409

    try:
        raw_df = _read_external_table(sheet)
        df, col_rt, col_smiles, important_cols, source_type, n_dropped = prepare_external_df(raw_df)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Could not parse this dataset: {e}"}), 400

    try:
        ctx, test = split_page(df)

        if algorithm == "llm":
            if not API_KEYS:
                return jsonify({"error": "No API keys configured in .env. "
                                          "Choose Random Forest or Linear Regression instead, "
                                          "or add a Gemini key to use the LLM option."}), 500

            subsampled = False
            if len(test) > MAX_EXTERNAL_LLM_TEST:
                test = test.sample(n=MAX_EXTERNAL_LLM_TEST, random_state=SEED).reset_index(drop=True)
                subsampled = True

            train_table = build_context_table(ctx, col_rt, col_smiles, important_cols)
            prompt_text = build_page_prompt("B", train_table, test, col_smiles, important_cols)

            try:
                response_text = send_page(prompt_text, attach_paper=False, n_test=len(test))
                predictions = parse_predictions(response_text, len(test))
            except Exception as e:
                state["last_gemini_error"] = str(e)
                return jsonify({
                    "error": (
                        f"Gemini did not return valid predictions for this dataset.\n\n"
                        f"Detail: {e}\n\n"
                        f"Suggestions:\n"
                        f"  1. Retry (the model sometimes truncates transiently).\n"
                        f"  2. Check your quota: https://ai.dev/rate-limit\n"
                        f"  3. Add more keys to .env.\n"
                        f"  4. Or switch to Random Forest / Linear Regression, which run locally."
                    )
                }), 502
        else:
            subsampled = False
            X_train = build_feature_matrix(ctx, col_smiles, important_cols)
            y_train = ctx[col_rt].to_numpy(dtype=float)
            X_test = build_feature_matrix(test, col_smiles, important_cols)

            if algorithm == "linear":
                model = LinearRegression()
            else:
                algorithm = "rf"
                model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)

            model.fit(X_train, y_train)
            predictions = [float(p) for p in model.predict(X_test)]
            col_std = {c: float(np.std(X_train[:, FP_N_BITS + j])) for j, c in enumerate(important_cols)}
            state["trained_model_info"].setdefault("H", {})[sheet_key] = {
                "model": model, "important_cols": important_cols, "col_std": col_std,
                "train_smiles": ctx[col_smiles].astype(str).tolist(),
            }
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    if force:
        state["scenario_data"][scenario] = [
            d for d in state["scenario_data"][scenario] if d["sheet"] != sheet_key
        ]

    results = []
    for (_, row), rt_pred in zip(test.iterrows(), predictions):
        smi = str(row[col_smiles]).strip()
        rt_real = float(row[col_rt])
        item = {
            "scenario": scenario,
            "sheet": sheet_key,
            "smiles": smi,
            "rt_real": rt_real,
            "rt_pred": rt_pred,
            "abs_err": abs(rt_pred - rt_real),
        }
        results.append(item)
        state["scenario_data"][scenario].append(item)

    state["processed_sheets"][scenario].add(sheet_key)

    return jsonify({
        "ok": True,
        "scenario": scenario,
        "sheet": sheet_key,
        "algorithm": algorithm,
        "source_type": source_type,
        "subsampled": subsampled,
        "n_context": len(ctx),
        "n_test": len(test),
        "results": results,
        "total_accumulated": len(state["scenario_data"][scenario]),
        "processed_sheets": sorted(state["processed_sheets"][scenario]),
    })


@app.route("/api/scenario_stats", methods=["POST"])
def scenario_stats():
    scenario = (request.json or {}).get("scenario") or state.get("scenario")
    if scenario not in SCENARIOS:
        return jsonify({"error": "Invalid scenario."}), 400
    data = state["scenario_data"][scenario]
    stats = _stats_from_data(data)
    return jsonify({
        "ok": True,
        "scenario": scenario,
        "description": SCENARIOS[scenario],
        "stats": stats,
        "points": [
            {"real": d["rt_real"], "pred": d["rt_pred"], "sheet": d["sheet"], "smiles": d["smiles"]}
            for d in data
        ],
        "processed_sheets": sorted(state["processed_sheets"][scenario]),
    })


@app.route("/api/scenario_download_csv")
def scenario_download_csv():
    scenario = request.args.get("scenario")
    if scenario not in SCENARIOS:
        return jsonify({"error": "Invalid scenario."}), 400
    data = state["scenario_data"][scenario]
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["scenario", "sheet", "smiles", "rt_real", "rt_pred", "abs_err"])
    for d in data:
        writer.writerow([d["scenario"], d["sheet"], d["smiles"],
                          f'{d["rt_real"]:.4f}', f'{d["rt_pred"]:.4f}', f'{d["abs_err"]:.4f}'])
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=scenario_{scenario}_data.csv"},
    )


@app.route("/api/scenario_download_pdf")
def scenario_download_pdf():
    scenario = request.args.get("scenario")
    if scenario not in SCENARIOS:
        return jsonify({"error": "Invalid scenario."}), 400
    data = state["scenario_data"][scenario]
    stats = _stats_from_data(data)
    pdf_bytes = build_stats_pdf(
        scenario, SCENARIOS[scenario], stats, sorted(state["processed_sheets"][scenario])
    )
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=scenario_{scenario}_statistics.pdf"},
    )


@app.route("/api/model_sheets")
def model_sheets():
    scenario = request.args.get("scenario")
    if scenario != "F":
        return jsonify({"ok": True, "sheets": []})
    sheets_dict = state["trained_model_info"].get("F", {})
    return jsonify({"ok": True, "sheets": sorted(sheets_dict.keys())})


def _pick_random_forest_model(scenario: Optional[str], sheet: Optional[str]):
    if scenario != "F":
        return None, None, None
    sheets_dict = state["trained_model_info"].get("F", {})
    if not sheets_dict:
        return None, None, None
    if sheet and sheet in sheets_dict:
        info = sheets_dict[sheet]
        if info and isinstance(info.get("model"), RandomForestRegressor):
            return "F", sheet, info
    for s, info in sheets_dict.items():
        if info and isinstance(info.get("model"), RandomForestRegressor):
            return "F", s, info
    return None, None, None


@app.route("/api/model_tree_png")
def model_tree_png():
    scenario = request.args.get("scenario")
    sheet = request.args.get("sheet")
    actual_scn, actual_sheet, info = _pick_random_forest_model(scenario, sheet)
    if info is None:
        return jsonify({
            "error": "The decision tree can only be downloaded for scenario F "
                     "(Random Forest). Run scenario F on at least one sheet first."
        }), 400

    model = info["model"]
    one_tree = model.estimators_[0]
    feature_names = _feature_names(info["important_cols"])

    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(one_tree, max_depth=3, feature_names=feature_names,
              filled=True, fontsize=8, ax=ax, proportion=True)
    ax.set_title(
        f"One tree out of 300 in the Random Forest (scenario {actual_scn}, "
        f"sheet: {actual_sheet}). Only the first 3 levels are shown, the "
        f"real tree goes much deeper.", fontsize=11,
    )
    png_bytes = _save_plot_png(fig)

    return Response(
        png_bytes, mimetype="image/png",
        headers={"Content-Disposition": f"attachment; filename=scenario_{actual_scn}_{actual_sheet}_tree.png"},
    )


@app.route("/api/rdkit_check")
def rdkit_check():
    import sys
    result = {
        "available": RDKIT_AVAILABLE,
        "version": RDKIT_VERSION,
        "error": RDKIT_IMPORT_ERROR,
        "python_executable": sys.executable,
        "python_version": sys.version,
    }
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles("CCO")
            result["test_smiles_ok"] = mol is not None
            result["test_smiles_atoms"] = mol.GetNumAtoms() if mol else None
        except Exception as e:
            result["test_smiles_ok"] = False
            result["test_smiles_error"] = f"{type(e).__name__}: {e}"
    return jsonify(result)


@app.route("/api/scenario_reset", methods=["POST"])
def scenario_reset():
    scenario = (request.json or {}).get("scenario")
    if scenario not in SCENARIOS:
        return jsonify({"error": "Invalid scenario."}), 400
    state["scenario_data"][scenario] = []
    state["processed_sheets"][scenario] = set()
    return jsonify({"ok": True})


@app.route("/api/status")
def status():
    return jsonify({
        "ok": True,
        "model": MODEL_NAME,
        "n_api_keys": len(API_KEYS),
        "rdkit_available": RDKIT_AVAILABLE,
        "rdkit_version": RDKIT_VERSION,
        "rdkit_error": RDKIT_IMPORT_ERROR,
        "feature_backend": "rdkit_morgan_fingerprint" if RDKIT_AVAILABLE else "pure_python_hashing_fallback",
        "excel_loaded": state["xls_bytes"] is not None,
        "paper_loaded": state["paper_path"] is not None,
        "paper_name": state["paper_name"],
        "scenario": state.get("scenario"),
        "accumulated": {s: len(v) for s, v in state["scenario_data"].items()},
        "requests_made": state["requests_made"],
        "last_gemini_error": state["last_gemini_error"],
        "manual_predictions_count": len(state["manual_predictions"]),
        "manual_context_active": state["manual_context_table"] is not None,
        "manual_context_sheet": state["manual_context_sheet"],
        "pubchempy_available": PUBCHEMPY_AVAILABLE,
        "external_loaded": state["external_bytes"] is not None,
        "external_filename": state.get("external_filename"),
        "trained_model_scenarios": list(state["trained_model_info"].keys()),
    })

#HTML AND JAVASCRIPT CODE FOR INTERFACE AND COMUNICATION WITH BACK END
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RT Predictor - Scenarios A–H</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#ffffff; --surface:#f5f5f7; --border:#d2d2d7;
    --blue:#0071e3; --blue-dark:#0056b3; --text:#1d1d1f; --subtle:#6e6e73;
    --success:#1a8917; --error:#c00; --warn:#a86a00; --radius:16px;
    --font:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  }
  body{font-family:var(--font);background:var(--bg);color:var(--text);min-height:100vh}
  header{display:flex;align-items:center;gap:12px;padding:22px 40px;border-bottom:1px solid var(--border)}
  .logo-dot{width:26px;height:26px;border-radius:50%;background:linear-gradient(135deg,#0071e3,#42a5f5)}
  header h1{font-size:19px;font-weight:700;letter-spacing:-.3px}
  header span{font-size:13px;color:var(--subtle);margin-left:4px}
  .content{max-width:960px;margin:32px auto 80px;padding:0 40px}
  .panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
    padding:32px 36px;margin-bottom:22px}
  .panel-title{font-size:20px;font-weight:700;letter-spacing:-.3px;margin-bottom:6px}
  .panel-sub{font-size:13.5px;color:var(--subtle);margin-bottom:22px;line-height:1.5}
  .hidden{display:none !important}
  .scn-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
  .scn-card{border:2px solid var(--border);border-radius:12px;padding:18px 18px;cursor:pointer;
    background:var(--bg);transition:border-color .15s,background .15s}
  .scn-card:hover{border-color:var(--blue);background:#f0f7ff}
  .scn-card.selected{border-color:var(--blue);background:#eaf3ff}
  .scn-letter{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;
    border-radius:50%;background:var(--blue);color:#fff;font-weight:700;font-size:13px;margin-right:8px}
  .scn-name{font-weight:700;font-size:14.5px}
  .scn-desc{font-size:12.5px;color:var(--subtle);margin-top:6px;line-height:1.4}
  .btn{display:inline-flex;align-items:center;gap:8px;padding:11px 24px;border:none;
    border-radius:980px;font-size:14.5px;font-weight:600;cursor:pointer;
    transition:background .2s,opacity .2s;font-family:var(--font)}
  .btn:active{transform:scale(.97)}
  .btn-primary{background:var(--blue);color:#fff}
  .btn-primary:hover{background:var(--blue-dark)}
  .btn-primary:disabled{opacity:.4;cursor:not-allowed}
  .btn-ghost{background:transparent;color:var(--blue);border:1.5px solid var(--blue)}
  .btn-ghost:hover{background:#f0f7ff}
  .btn-row{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}
  .btn-stats{background:#eaf3ff;color:var(--blue-dark);border:1.5px solid var(--blue)}
  .btn-stats:hover{background:#dcedff}
  select,input[type=number],input[type=text]{width:100%;padding:11px 14px;border:1.5px solid var(--border);
    border-radius:10px;font-size:14.5px;font-family:var(--font);background:var(--bg)}
  .drop-zone{background:var(--bg);border:2px dashed var(--border);border-radius:12px;
    padding:26px 18px;text-align:center;cursor:pointer;position:relative;transition:border-color .2s,background .2s}
  .drop-zone:hover{border-color:var(--blue);background:#f0f7ff}
  .drop-zone.filled{border-style:solid;border-color:var(--success);background:#f3faf3}
  .drop-zone.loading{border-style:solid;border-color:var(--blue);background:#eef5ff;pointer-events:none}
  .drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .drop-label{font-size:13.5px;font-weight:600}
  .drop-hint{font-size:11.5px;color:var(--subtle);margin-top:4px}
  .drop-name{font-size:11.5px;color:var(--success);margin-top:6px;font-weight:600}
  .drop-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}
  .field-label{font-size:12.5px;font-weight:600;color:var(--subtle);margin-bottom:6px;
    text-transform:uppercase;letter-spacing:.4px}
  .badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11.5px;font-weight:700}
  .badge-req{background:#ffe9e5;color:var(--error)}
  .badge-opt{background:#fff3d6;color:var(--warn)}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:14px}
  th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}
  th{color:var(--subtle);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.3px}
  td.smiles{font-family:monospace;font-size:11.5px;max-width:260px;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
  .num{text-align:right;font-variant-numeric:tabular-nums}
  .info-row{display:flex;gap:22px;flex-wrap:wrap;margin:14px 0;font-size:13px;color:var(--subtle)}
  .info-row b{color:var(--text)}
  .stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
  .stat-card{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
  .stat-name{font-size:11px;color:var(--subtle);font-weight:700;text-transform:uppercase;letter-spacing:.4px}
  .stat-val{font-size:20px;font-weight:700;margin-top:4px}
  .toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(20px);
    background:var(--text);color:#fff;padding:12px 22px;border-radius:10px;font-size:13.5px;
    opacity:0;pointer-events:none;transition:opacity .25s,transform .25s;z-index:999;max-width:600px}
  .toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
  .toast.error{background:#c00}
  canvas#chart{max-width:100%}
  .chart-wrap{background:var(--bg);border:1px solid var(--border);border-radius:12px;padding:18px;margin:16px 0}
  .spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.4);
    border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite}
  .spinner-dark{display:inline-block;width:14px;height:14px;border:2px solid rgba(0,113,227,.25);
    border-top-color:var(--blue);border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle}
  @keyframes spin{to{transform:rotate(360deg)}}
  .loading-row{display:flex;align-items:center;gap:8px;color:var(--blue);font-size:13px;font-weight:600;
    margin:14px 0}
  .switch-row{display:flex;align-items:center;gap:10px;margin:16px 0 6px;font-size:14px}
  .switch-row input[type=checkbox]{width:auto;padding:0;accent-color:var(--blue);transform:scale(1.15)}
  .switch-row label{font-weight:600;color:var(--text);cursor:pointer}
</style>
</head>
<body>

<header>
  <div class="logo-dot"></div>
  <h1>RT Predictor <span></span></h1>
  <span style="margin-left:auto;font-size:12.5px;color:var(--subtle);display:flex;align-items:center;gap:16px">
    Model: <b id="model-name" style="color:var(--text)">-</b>
    · Fingerprints: <b id="fp-backend" style="color:var(--text)">-</b>
    · Gemini API calls made: <b id="req-counter" style="color:var(--text)">0</b>
    <button class="btn btn-stats" style="padding:7px 16px;font-size:12.5px" onclick="showStats()">📊 Statistics</button>
  </span>
</header>

<div class="content">

  <div class="panel" id="panel-scenario">
    <div class="panel-title">1. Choose the scenario</div>
    <div class="panel-sub">Each scenario defines what information the predictor gets before estimating a molecule's RT.</div>
    <div class="scn-grid">
      <div class="scn-card" data-scn="A" onclick="selectScenario('A')">
        <div><span class="scn-letter">A</span><span class="scn-name">No context</span></div>
        <div class="scn-desc">Only the SMILES of the sheet's 20% is sent to Gemini. No examples, no paper.</div>
      </div>
      <div class="scn-card" data-scn="B" onclick="selectScenario('B')">
        <div><span class="scn-letter">B</span><span class="scn-name">80% context</span></div>
        <div class="scn-desc">80% of the sheet (RT+SMILES+conditions) is sent as examples → Gemini predicts the remaining 20%.</div>
      </div>
      <div class="scn-card" data-scn="C" onclick="selectScenario('C')">
        <div><span class="scn-letter">C</span><span class="scn-name">80% context + paper</span></div>
        <div class="scn-desc">Same as B, and you can also upload a scientific paper (optional) as extra support.</div>
      </div>
      <div class="scn-card" data-scn="D" onclick="selectScenario('D')">
        <div><span class="scn-letter">D</span><span class="scn-name">Paper only</span></div>
        <div class="scn-desc">No RT+SMILES examples. Only the SMILES of the 20% + a <b>required</b> paper.</div>
      </div>
      <div class="scn-card" data-scn="E" onclick="selectScenario('E')">
        <div><span class="scn-letter">E</span><span class="scn-name">Linear Regression</span></div>
        <div class="scn-desc">Trains a Linear Regression model on the sheet's 80% (fingerprints + conditions) and predicts the remaining 20%. Local, no LLM.</div>
      </div>
      <div class="scn-card" data-scn="F" onclick="selectScenario('F')">
        <div><span class="scn-letter">F</span><span class="scn-name">Random Forest</span></div>
        <div class="scn-desc">Same as E, but with a Random Forest regressor instead of a linear model. Local, no LLM.</div>
      </div>
      <div class="scn-card" data-scn="G" onclick="selectScenario('G')">
        <div><span class="scn-letter">G</span><span class="scn-name">Manual SMILES (Gemini)</span></div>
        <div class="scn-desc">Type in any SMILES by hand and Gemini predicts its RT. You can optionally provide 80% context from a sheet (no local model).</div>
      </div>
      <div class="scn-card" data-scn="H" onclick="selectScenario('H')">
        <div><span class="scn-letter">H</span><span class="scn-name">External dataset</span></div>
        <div class="scn-desc">Upload a CSV or Excel file from a recent study (e.g. SMRT, Kumari et al.). Auto-detects the compound (SMILES, InChI or name) and RT columns, then runs the same 80/20 evaluation. Local, no LLM.</div>
      </div>
    </div>
    <div class="btn-row">
      <button class="btn btn-primary" id="btn-confirm-scenario" onclick="confirmScenario()" disabled>Confirm scenario →</button>
    </div>
  </div>

  <div class="panel hidden" id="panel-files">
    <div class="panel-title">2. Upload your files</div>
    <div class="panel-sub" id="files-sub">Upload the Excel file with the sheets (RT + SMILES).</div>

    <div id="standard-upload-wrap">
      <div class="drop-row">
        <div>
          <div class="field-label">Excel <span class="badge" id="excel-badge">Required</span></div>
          <div class="drop-zone" id="dz-excel" onclick="document.getElementById('file-excel').click()">
            <input type="file" id="file-excel" accept=".xlsx,.xls" onchange="handleExcel(this.files[0])"/>
            <div class="drop-label" id="excel-drop-label">📊 Drag & drop or click</div>
            <div class="drop-hint">.xlsx / .xls</div>
            <div class="drop-name" id="name-excel"></div>
          </div>
        </div>
        <div id="paper-slot">
          <div class="field-label">Paper (PDF) <span class="badge" id="paper-badge">-</span></div>
          <div class="drop-zone" id="dz-paper" onclick="document.getElementById('file-paper').click()">
            <input type="file" id="file-paper" accept=".pdf" onchange="handlePaper(this.files[0])"/>
            <div class="drop-label">📄 Drag & drop or click</div>
            <div class="drop-hint">.pdf</div>
            <div class="drop-name" id="name-paper"></div>
          </div>
        </div>
      </div>
    </div>

    <div id="external-upload-wrap" class="hidden">
      <div class="field-label">Dataset (CSV or Excel) <span class="badge badge-req">Required</span></div>
      <div class="drop-zone" id="dz-external" onclick="document.getElementById('file-external').click()">
        <input type="file" id="file-external" accept=".csv,.tsv,.xlsx,.xls" onchange="handleExternal(this.files[0])"/>
        <div class="drop-label">🧪 Drag & drop or click</div>
        <div class="drop-hint">.csv / .xlsx / .xls - e.g. SMRT, Kumari et al., or any recent study's supplementary table</div>
        <div class="drop-name" id="name-external"></div>
      </div>

      <div id="external-sheet-wrap" class="hidden" style="margin-top:14px">
        <div class="field-label">Sheet</div>
        <select id="external-sheet-select" onchange="previewExternal()"></select>
      </div>

      <div style="margin-top:14px">
        <div class="field-label">Algorithm</div>
        <select id="external-algorithm" onchange="onExternalAlgorithmChange()">
          <option value="rf">Random Forest (recommended)</option>
          <option value="linear">Linear Regression</option>
          <option value="llm">LLM (Gemini)</option>
        </select>
        <div class="drop-hint" id="external-llm-note" style="margin-top:6px"></div>
      </div>

      <div class="info-row hidden" id="external-info">
        <div>Detected RT column: <b id="ext-rt-col"></b></div>
        <div>Compound column type: <b id="ext-compound-type"></b></div>
        <div>Valid compounds: <b id="ext-total"></b></div>
        <div id="ext-dropped-wrap" class="hidden">Dropped (invalid RT/structure): <b id="ext-dropped"></b></div>
        <div>Context (80%): <b id="ext-ctx"></b></div>
        <div>Prediction (20%): <b id="ext-test"></b></div>
      </div>
    </div>

    <div class="btn-row">
      <button class="btn btn-ghost" onclick="backToPanel('panel-scenario')">← Back</button>
      <button class="btn btn-primary" id="btn-to-page" onclick="handleFilesNext()" disabled>Choose sheet →</button>
      <button class="btn btn-ghost" onclick="changeScenario()">🔁 Change scenario</button>
    </div>
  </div>

  <div class="panel hidden" id="panel-manual">
    <div class="panel-title">3. Predict RT for a SMILES you type in</div>
    <div class="panel-sub" id="manual-trained-info">
      Type any SMILES below and Gemini will predict its retention time.
    </div>

    <div class="switch-row">
      <input type="checkbox" id="manual-use-context" onchange="onManualUseContextChange()"/>
      <label for="manual-use-context">Use 80% context from an Excel sheet (optional)</label>
    </div>

    <div id="manual-context-block" class="hidden">
      <div class="field-label" style="margin-top:12px">Excel file with the context sheet</div>
      <div class="drop-zone" id="dz-excel-g" onclick="document.getElementById('file-excel-g').click()">
        <input type="file" id="file-excel-g" accept=".xlsx,.xls" onchange="handleExcelG(this.files[0])"/>
        <div class="drop-label" id="excel-g-drop-label">📊 Drag & drop or click</div>
        <div class="drop-hint">.xlsx / .xls</div>
        <div class="drop-name" id="name-excel-g"></div>
      </div>

      <div id="manual-context-sheet-wrap" class="hidden" style="margin-top:14px">
        <div class="field-label">Context sheet</div>
        <select id="manual-context-sheet" onchange="onManualContextSheetChange()"></select>
      </div>

      <div class="loading-row hidden" id="manual-context-loading">
        <span class="spinner-dark"></span> Loading context from the sheet…
      </div>
      <div class="drop-hint" id="manual-context-status" style="margin-top:6px"></div>
    </div>

    <div class="field-label" style="margin-top:16px">SMILES</div>
    <input type="text" id="manual-smiles" placeholder="e.g. CCO (ethanol)" style="margin-bottom:16px;font-family:monospace"/>

    <div class="btn-row">
      <button class="btn btn-primary" id="btn-manual-predict" onclick="predictManualSmiles()">Predict RT with Gemini</button>
      <button class="btn btn-ghost" onclick="changeScenario()">🔁 Change scenario</button>
    </div>

    <table id="manual-results-table" class="hidden">
      <thead><tr><th>SMILES</th><th class="num">Predicted RT</th><th>Engine</th><th>Context</th><th>Model</th></tr></thead>
      <tbody id="manual-results-tbody"></tbody>
    </table>

    <div class="btn-row hidden" id="manual-download-row">
      <button class="btn btn-primary" onclick="downloadManualCSV()">⬇️ Download predictions (CSV)</button>
      <button class="btn btn-ghost" onclick="resetManualPredictions()">🗑️ Clear list</button>
    </div>
  </div>

  <div class="panel hidden" id="panel-page">
    <div class="panel-title">4. Choose a sheet from the Excel file</div>
    <div class="panel-sub">Active scenario: <b id="scn-active-label"></b></div>

    <div class="field-label">Sheet</div>
    <select id="sheet-select" onchange="onSheetChange()"></select>

    <div id="sheet-loading" class="loading-row hidden">
      <span class="spinner-dark"></span> Loading sheet info (valid rows, context and prediction split)…
    </div>

    <div class="info-row hidden" id="page-info">
      <div>Valid rows: <b id="pi-total"></b></div>
      <div id="pi-ctx-wrap">Context / training set (80%): <b id="pi-ctx"></b></div>
      <div id="pi-noctx-wrap" class="hidden" style="color:var(--subtle)">No context sent (scenario <span id="pi-noctx-scn"></span>)</div>
      <div id="pi-test-wrap">Prediction set (20%, no RT): <b id="pi-test"></b></div>
      <div id="pi-processed" class="hidden" style="color:var(--warn)">⚠️ Already processed in this scenario</div>
    </div>

    <div class="btn-row">
      <button class="btn btn-ghost" onclick="backToPanel('panel-files')">← Back</button>
      <button class="btn btn-primary" id="btn-run-page" onclick="handlePageAction()" disabled>
        <span id="run-page-label">Predict this sheet</span>
      </button>
      <button class="btn btn-stats hidden" id="btn-go-stats-1" onclick="showStats()">View scenario statistics →</button>
      <button class="btn btn-ghost" onclick="changeScenario()">🔁 Change scenario</button>
    </div>
  </div>

  <div class="panel hidden" id="panel-results">
    <div class="panel-title">Results for sheet <span id="res-sheet-name"></span></div>
    <div class="info-row">
      <div>Predictions in this sheet: <b id="res-count"></b></div>
      <div>Scenario total accumulated: <b id="res-total"></b></div>
      <div>Sheets processed: <b id="res-pages"></b></div>
    </div>
    <table>
      <thead><tr><th>SMILES</th><th class="num">Real RT</th><th class="num">Predicted RT</th><th class="num">Abs. error</th></tr></thead>
      <tbody id="res-tbody"></tbody>
    </table>
    <div class="btn-row">
      <button class="btn btn-ghost" onclick="backFromResults()">Choose another sheet</button>
      <button class="btn btn-primary" onclick="showStats()">Calculate scenario statistics →</button>
      <button class="btn btn-ghost" onclick="changeScenario()">🔁 Change scenario</button>
    </div>
  </div>

  <div class="panel hidden" id="panel-stats">
    <div class="panel-title">Statistics for scenario <span id="stats-scn-label"></span></div>
    <div class="panel-sub" id="stats-sub"></div>

    <div class="stats-grid" id="stats-grid"></div>

    <div class="info-row hidden" id="model-insights-row">
      <div style="min-width:220px">
        <div class="field-label">Sheet (for the decision tree)</div>
        <select id="model-sheet-select"></select>
      </div>
    </div>

    <div class="chart-wrap">
      <canvas id="chart" height="110"></canvas>
    </div>

    <div class="btn-row">
      <button class="btn btn-ghost" onclick="backToPanel('panel-page')">← Predict more sheets</button>
      <button class="btn btn-primary" onclick="downloadPDF()">⬇️ Download statistics (PDF)</button>
      <button class="btn btn-primary" onclick="downloadCSV()">⬇️ Download raw data (CSV)</button>
      <button class="btn btn-primary" onclick="downloadChart()">⬇️ Download chart (PNG)</button>
      <button class="btn btn-primary hidden" id="btn-tree-png" onclick="downloadTreePNG()">🌳 Download a tree (PNG)</button>
      <button class="btn btn-ghost" onclick="resetScenario()">🗑️ Reset scenario data</button>
      <button class="btn btn-ghost" onclick="changeScenario()">🔁 Change scenario</button>
    </div>
  </div>

</div>

<div class="toast" id="toast"></div>

<script>
let selectedScenario = null;
let activeScenario = null;
let isManual = false;
let isExternal = false;
let sheets = [];
let externalSheets = [];
let currentChart = null;
let gExcelReady = false;

function customConfirm(message){
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:2000;'
      + 'display:flex;align-items:center;justify-content:center;padding:20px';
    const box = document.createElement('div');
    box.style.cssText = 'background:#fff;border-radius:14px;padding:24px 28px;max-width:440px;'
      + 'box-shadow:0 10px 40px rgba(0,0,0,.25)';
    const msg = document.createElement('div');
    msg.style.cssText = 'font-size:14.5px;line-height:1.5;white-space:pre-wrap;margin-bottom:20px;color:#1d1d1f';
    msg.textContent = message;
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:10px;justify-content:flex-end';
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn btn-ghost';
    cancelBtn.textContent = 'Cancel';
    const okBtn = document.createElement('button');
    okBtn.className = 'btn btn-primary';
    okBtn.textContent = 'Resend';
    row.appendChild(cancelBtn);
    row.appendChild(okBtn);
    box.appendChild(msg);
    box.appendChild(row);
    overlay.appendChild(box);
    document.body.appendChild(overlay);
    const cleanup = (result) => { document.body.removeChild(overlay); resolve(result); };
    okBtn.onclick = () => cleanup(true);
    cancelBtn.onclick = () => cleanup(false);
    overlay.onclick = (e) => { if(e.target === overlay) cleanup(false); };
  });
}

function showToast(msg, isError){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (isError ? ' error' : '');
  setTimeout(()=> t.classList.remove('show'), 6000);
}

function panelShow(id){
  document.querySelectorAll('.panel').forEach(p => p.classList.add('hidden'));
  document.getElementById(id).classList.remove('hidden');
  window.scrollTo({top:0, behavior:'smooth'});
}

function backToPanel(id){
  panelShow(id);
  if(id === 'panel-page'){ refreshPageStatsButton(); }
}

function selectScenario(s){
  selectedScenario = s;
  document.querySelectorAll('.scn-card').forEach(c => c.classList.toggle('selected', c.dataset.scn === s));
  document.getElementById('btn-confirm-scenario').disabled = false;
}

async function confirmScenario(){
  try{
    const r = await fetch('/api/set_scenario', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({scenario: selectedScenario})
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Error');
    activeScenario = d.scenario;
    isManual = !!d.is_manual;
    isExternal = !!d.is_external;

    if(isManual){
      document.getElementById('manual-trained-info').textContent =
        'Scenario G: type any SMILES below and Gemini will predict its RT. ' +
        'You can optionally enable "Use 80% context from an Excel sheet" to send ' +
        'the RT+SMILES+conditions of a sheet (80%) as examples.';
      document.getElementById('manual-context-status').textContent = '';
      document.getElementById('manual-context-sheet').innerHTML = '';
      document.getElementById('manual-context-sheet-wrap').classList.add('hidden');
      document.getElementById('manual-use-context').checked = false;
      document.getElementById('manual-context-block').classList.add('hidden');
      document.getElementById('dz-excel-g').classList.remove('filled');
      document.getElementById('dz-excel-g').classList.remove('loading');
      document.getElementById('name-excel-g').textContent = '';
      document.getElementById('file-excel-g').value = '';
      document.getElementById('excel-g-drop-label').textContent = '📊 Drag & drop or click';
      gExcelReady = false;
      panelShow('panel-manual');
      return;
    }

    document.getElementById('files-sub').textContent =
      'Scenario ' + activeScenario + ': ' + d.description;

    document.getElementById('standard-upload-wrap').classList.toggle('hidden', isExternal);
    document.getElementById('external-upload-wrap').classList.toggle('hidden', !isExternal);

    const paperSlot = document.getElementById('paper-slot');
    const badge = document.getElementById('paper-badge');
    if(!isExternal && d.allows_paper){
      paperSlot.classList.remove('hidden');
      badge.textContent = d.requires_paper ? 'Required' : 'Optional';
      badge.className = 'badge ' + (d.requires_paper ? 'badge-req' : 'badge-opt');
    } else {
      paperSlot.classList.add('hidden');
    }

    document.getElementById('scn-active-label').textContent = activeScenario + '-' + d.description;

    const btn = document.getElementById('btn-to-page');
    btn.textContent = isExternal ? 'Run 80/20 evaluation →' : 'Choose sheet →';
    btn.disabled = true;

    if(isExternal){
      document.getElementById('external-info').classList.add('hidden');
      document.getElementById('dz-external').classList.remove('filled');
      document.getElementById('dz-external').classList.remove('loading');
      document.getElementById('name-external').textContent = '';
      document.getElementById('file-external').value = '';
    } else {
      checkFilesReady();
    }

    panelShow('panel-files');
  }catch(e){ showToast(e.message, true); }
}

let excelReady = false, paperReady = false;

function setExcelLoading(on){
  const dz = document.getElementById('dz-excel');
  const lbl = document.getElementById('excel-drop-label');
  const nameEl = document.getElementById('name-excel');
  if(on){
    dz.classList.add('loading');
    lbl.innerHTML = '<span class="spinner-dark"></span> Uploading and reading sheets…';
    nameEl.textContent = '';
  } else {
    dz.classList.remove('loading');
    lbl.textContent = '📊 Drag & drop or click';
  }
}

async function handleExcel(file){
  if(!file) return;
  excelReady = false;
  document.getElementById('btn-to-page').disabled = true;
  setExcelLoading(true);
  const fd = new FormData(); fd.append('file', file);
  try{
    const r = await fetch('/api/upload_excel', {method:'POST', body: fd});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Error uploading Excel file');
    sheets = d.sheets;
    excelReady = true;
    document.getElementById('dz-excel').classList.add('filled');
    document.getElementById('name-excel').textContent = file.name + ' (' + sheets.length + ' sheets)';
    showToast('Excel file loaded.');
  }catch(e){
    showToast(e.message, true);
    document.getElementById('dz-excel').classList.remove('filled');
    document.getElementById('name-excel').textContent = '';
  }finally{
    setExcelLoading(false);
    checkFilesReady();
  }
}

async function handlePaper(file){
  if(!file) return;
  const fd = new FormData(); fd.append('file', file);
  try{
    const r = await fetch('/api/upload_paper', {method:'POST', body: fd});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Error uploading paper');
    paperReady = true;
    document.getElementById('dz-paper').classList.add('filled');
    document.getElementById('name-paper').textContent = file.name;
    showToast('Paper loaded.');
    checkFilesReady();
  }catch(e){ showToast(e.message, true); }
}

function checkFilesReady(){
  const paperVisible = !document.getElementById('paper-slot').classList.contains('hidden');
  const paperBadgeReq = document.getElementById('paper-badge').textContent === 'Required';
  const paperOk = !paperVisible || !paperBadgeReq || paperReady;
  document.getElementById('btn-to-page').disabled = !(excelReady && paperOk);
}

function onExternalAlgorithmChange(){
  const algo = document.getElementById('external-algorithm').value;
  const note = document.getElementById('external-llm-note');
  if(algo === 'llm'){
    note.textContent = 'Uses the Gemini API (needs a key configured in .env) and consumes quota. ' +
      'If the prediction set has more than 500 compounds, it is randomly capped to 500 so it fits the context window.';
  } else {
    note.textContent = 'Runs entirely locally, no API calls, no quota used.';
  }
}

async function handleExternal(file){
  if(!file) return;
  const dz = document.getElementById('dz-external');
  const nameEl = document.getElementById('name-external');
  dz.classList.add('loading');
  nameEl.textContent = '';
  document.getElementById('btn-to-page').disabled = true;
  const fd = new FormData(); fd.append('file', file);
  try{
    const r = await fetch('/api/upload_external', {method:'POST', body: fd});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Error uploading dataset');

    dz.classList.add('filled');
    nameEl.textContent = file.name;

    externalSheets = d.sheets || [];
    const sheetWrap = document.getElementById('external-sheet-wrap');
    const sheetSel = document.getElementById('external-sheet-select');
    if(externalSheets.length > 1){
      sheetSel.innerHTML = externalSheets.map(s => `<option value="${s}">${s}</option>`).join('');
      sheetWrap.classList.remove('hidden');
    } else {
      sheetWrap.classList.add('hidden');
    }

    await previewExternal();
    document.getElementById('btn-to-page').disabled = false;
    onExternalAlgorithmChange();
    showToast('Dataset loaded.');
  }catch(e){
    showToast(e.message, true);
    dz.classList.remove('filled');
    nameEl.textContent = '';
  }finally{
    dz.classList.remove('loading');
  }
}

async function previewExternal(){
  const sheet = externalSheets.length > 1 ? document.getElementById('external-sheet-select').value : null;
  try{
    const r = await fetch('/api/external_preview', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({sheet})
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Error');
    document.getElementById('ext-rt-col').textContent = d.detected_rt_column;
    document.getElementById('ext-compound-type').textContent = d.detected_compound_column_type.toUpperCase();
    document.getElementById('ext-total').textContent = d.total_valid;
    document.getElementById('ext-dropped-wrap').classList.toggle('hidden', !d.n_dropped);
    document.getElementById('ext-dropped').textContent = d.n_dropped;
    document.getElementById('ext-ctx').textContent = d.n_context;
    document.getElementById('ext-test').textContent = d.n_prediction;
    document.getElementById('external-info').classList.remove('hidden');
  }catch(e){
    showToast(e.message, true);
    document.getElementById('external-info').classList.add('hidden');
    document.getElementById('btn-to-page').disabled = true;
  }
}

async function runExternalEvaluation(force){
  const sheet = externalSheets.length > 1 ? document.getElementById('external-sheet-select').value : null;
  const algorithm = document.getElementById('external-algorithm').value;
  const btn = document.getElementById('btn-to-page');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Evaluating…';
  try{
    let useForce = !!force;
    let r, d;
    while(true){
      r = await fetch('/api/run_external', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({sheet, algorithm, force: useForce})
      });
      d = await r.json();

      if(r.status === 409 && d.warning === 'already_processed'){
        const resend = await customConfirm(d.message + '\n\nResend anyway?');
        if(!resend){ return; }
        useForce = true;
        continue;
      }
      break;
    }
    if(!r.ok) throw new Error(d.error || 'Evaluation error');

    activeScenario = 'H';
    document.getElementById('res-sheet-name').textContent = '"' + d.sheet + '"';
    document.getElementById('res-count').textContent = d.results.length;
    document.getElementById('res-total').textContent = d.total_accumulated;
    document.getElementById('res-pages').textContent = d.processed_sheets.length;
    document.getElementById('res-tbody').innerHTML = d.results.map(row => `
      <tr>
        <td class="smiles">${row.smiles}</td>
        <td class="num">${row.rt_real.toFixed(3)}</td>
        <td class="num">${row.rt_pred.toFixed(3)}</td>
        <td class="num">${row.abs_err.toFixed(3)}</td>
      </tr>`).join('');

    panelShow('panel-results');
    const subNote = d.subsampled ? ' (capped to 500 molecules for the LLM)' : '';
    showToast('Evaluation complete (' + d.algorithm.toUpperCase() + ', source: ' + d.source_type.toUpperCase() + ')' + subNote + '.');
  }catch(e){
    showToast(e.message, true);
  }finally{
    btn.disabled = false;
    btn.textContent = 'Run 80/20 evaluation →';
  }
}

function handleFilesNext(){
  if(isExternal){ runExternalEvaluation(); }
  else { goToPageSelection(); }
}

function goToPageSelection(){
  const sel = document.getElementById('sheet-select');
  sel.innerHTML = '<option value="">- choose a sheet -</option>' +
    sheets.map(s => `<option value="${s}">${s}</option>`).join('');
  document.getElementById('page-info').classList.add('hidden');
  document.getElementById('sheet-loading').classList.add('hidden');
  document.getElementById('btn-run-page').disabled = true;
  panelShow('panel-page');
  refreshPageStatsButton();
}

async function refreshPageStatsButton(){
  const btn = document.getElementById('btn-go-stats-1');
  if(!activeScenario || isManual){ btn.classList.add('hidden'); return; }
  try{
    const r = await fetch('/api/status');
    const d = await r.json();
    const total = (d.accumulated && d.accumulated[activeScenario]) || 0;
    btn.classList.toggle('hidden', total === 0);
    btn.innerHTML = total > 0
      ? `📊 View scenario statistics (${total} accumulated points) →`
      : 'View scenario statistics →';
  }catch(e){}
}

async function onSheetChange(){
  const sheet = document.getElementById('sheet-select').value;
  const btn = document.getElementById('btn-run-page');
  const loadingEl = document.getElementById('sheet-loading');
  const infoEl = document.getElementById('page-info');

  if(!sheet){
    infoEl.classList.add('hidden');
    loadingEl.classList.add('hidden');
    btn.disabled = true;
    return;
  }

  btn.disabled = true;
  infoEl.classList.add('hidden');
  loadingEl.classList.remove('hidden');

  try{
    const r = await fetch('/api/sheet_rows', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({sheet})
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Error');
    document.getElementById('pi-total').textContent = d.total_valid;
    document.getElementById('pi-test').textContent = d.n_prediction;

    const noContext = (activeScenario === 'A' || activeScenario === 'D');
    document.getElementById('pi-ctx-wrap').classList.toggle('hidden', noContext);
    document.getElementById('pi-noctx-wrap').classList.toggle('hidden', !noContext);
    document.getElementById('pi-test-wrap').classList.remove('hidden');
    if(noContext){
      document.getElementById('pi-noctx-scn').textContent = activeScenario;
    } else {
      document.getElementById('pi-ctx').textContent = d.n_context;
    }
    document.getElementById('pi-processed').classList.toggle('hidden', !d.already_processed);
    document.getElementById('run-page-label').textContent =
      d.already_processed ? 'Resend (will overwrite previous data)' : 'Predict this sheet';

    infoEl.classList.remove('hidden');
    btn.disabled = false;
  }catch(e){
    showToast(e.message, true);
    infoEl.classList.add('hidden');
    btn.disabled = true;
  }finally{
    loadingEl.classList.add('hidden');
  }
}

function changeScenario(){
  selectedScenario = null;
  document.querySelectorAll('.scn-card').forEach(c => c.classList.remove('selected'));
  document.getElementById('btn-confirm-scenario').disabled = true;
  panelShow('panel-scenario');
}

function backFromResults(){
  if(isExternal){ panelShow('panel-files'); }
  else { backToPanel('panel-page'); }
}

function handlePageAction(){
  runPage();
}

async function onManualUseContextChange(){
  const on = document.getElementById('manual-use-context').checked;
  const block = document.getElementById('manual-context-block');
  const statusEl = document.getElementById('manual-context-status');
  const sheetWrap = document.getElementById('manual-context-sheet-wrap');

  if(!on){
    block.classList.add('hidden');
    try{
      await fetch('/api/set_manual_context', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({use_context: false})
      });
    }catch(e){}
    return;
  }

  block.classList.remove('hidden');
  statusEl.textContent = '';
  statusEl.style.color = 'var(--subtle)';

  if(gExcelReady && sheets.length > 0){
    populateManualContextSheets();
  } else {
    sheetWrap.classList.add('hidden');
    statusEl.style.color = 'var(--subtle)';
    statusEl.textContent = 'Load an Excel file above to enable context.';
  }
}

function populateManualContextSheets(){
  const sheetWrap = document.getElementById('manual-context-sheet-wrap');
  const sheetSel = document.getElementById('manual-context-sheet');
  const statusEl = document.getElementById('manual-context-status');
  if(!sheets || sheets.length === 0){
    statusEl.style.color = 'var(--error)';
    statusEl.textContent = 'No sheets available in the loaded Excel file.';
    return;
  }
  sheetSel.innerHTML = sheets.map(s => `<option value="${s}">${s}</option>`).join('');
  sheetWrap.classList.remove('hidden');
  onManualContextSheetChange();
}

async function handleExcelG(file){
  if(!file) return;
  gExcelReady = false;
  const dz = document.getElementById('dz-excel-g');
  const lbl = document.getElementById('excel-g-drop-label');
  const nameEl = document.getElementById('name-excel-g');
  const statusEl = document.getElementById('manual-context-status');
  dz.classList.add('loading');
  lbl.innerHTML = '<span class="spinner-dark"></span> Uploading and reading sheets…';
  nameEl.textContent = '';
  statusEl.textContent = '';

  const fd = new FormData(); fd.append('file', file);
  try{
    const r = await fetch('/api/upload_excel', {method:'POST', body: fd});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Error uploading Excel file');
    sheets = d.sheets;
    gExcelReady = true;
    dz.classList.add('filled');
    nameEl.textContent = file.name + ' (' + sheets.length + ' sheets)';
    showToast('Excel file loaded.');
    populateManualContextSheets();
  }catch(e){
    showToast(e.message, true);
    dz.classList.remove('filled');
    nameEl.textContent = '';
  }finally{
    dz.classList.remove('loading');
    lbl.textContent = '📊 Drag & drop or click';
  }
}

async function onManualContextSheetChange(){
  const sheet = document.getElementById('manual-context-sheet').value;
  const loading = document.getElementById('manual-context-loading');
  const statusEl = document.getElementById('manual-context-status');
  if(!sheet){ statusEl.textContent = ''; return; }

  loading.classList.remove('hidden');
  statusEl.textContent = '';
  try{
    const r = await fetch('/api/set_manual_context', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({use_context: true, sheet})
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Error');
    statusEl.style.color = 'var(--success)';
    statusEl.textContent = `Context active - ${d.n_context} example rows from sheet "${d.sheet}".`;
  }catch(e){
    statusEl.style.color = 'var(--error)';
    statusEl.textContent = e.message;
  }finally{
    loading.classList.add('hidden');
  }
}

async function predictManualSmiles(){
  const smiles = document.getElementById('manual-smiles').value.trim();
  if(!smiles){ showToast('Enter a SMILES first.', true); return; }

  const btn = document.getElementById('btn-manual-predict');
  const originalLabel = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';

  try{
    const r = await fetch('/api/predict_manual', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({smiles})
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Prediction error');
    const p = d.prediction;
    const tbody = document.getElementById('manual-results-tbody');
    const tr = document.createElement('tr');
    const ctxTxt = p.used_context ? ('with context (' + (p.context_sheet || '?') + ')') : 'no context';
    tr.innerHTML = `<td class="smiles">${p.smiles}</td><td class="num">${p.rt_pred.toFixed(3)}</td><td>${p.engine.toUpperCase()}</td><td>${ctxTxt}</td><td>${p.model}</td>`;
    tbody.prepend(tr);
    document.getElementById('manual-results-table').classList.remove('hidden');
    document.getElementById('manual-download-row').classList.remove('hidden');
    showToast('Predicted RT: ' + p.rt_pred.toFixed(3) + ' min');
  }catch(e){
    showToast(e.message, true);
  }finally{
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
}

function downloadManualCSV(){ window.open('/api/manual_download_csv', '_blank'); }

async function resetManualPredictions(){
  const ok = await customConfirm('Clear all manual predictions from this session?');
  if(!ok) return;
  try{
    await fetch('/api/manual_reset', {method:'POST'});
    document.getElementById('manual-results-tbody').innerHTML = '';
    document.getElementById('manual-results-table').classList.add('hidden');
    document.getElementById('manual-download-row').classList.add('hidden');
    showToast('Cleared.');
  }catch(e){ showToast(e.message, true); }
}

async function runPage(force){
  const sheet = document.getElementById('sheet-select').value;
  if(!sheet) return;
  const btn = document.getElementById('btn-run-page');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';
  try{
    let useForce = !!force;
    let r, d;
    while(true){
      r = await fetch('/api/run_page', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({sheet, force: useForce})
      });
      d = await r.json();

      if(r.status === 409 && d.warning === 'already_processed'){
        const resend = await customConfirm(d.message + '\n\nResend anyway?');
        if(!resend){ return; }
        useForce = true;
        continue;
      }
      break;
    }
    if(!r.ok) throw new Error(d.error || 'Prediction error');

    document.getElementById('res-sheet-name').textContent = '"' + d.sheet + '"';
    document.getElementById('res-count').textContent = d.results.length;
    document.getElementById('res-total').textContent = d.total_accumulated;
    document.getElementById('res-pages').textContent = d.processed_sheets.length;
    document.getElementById('res-tbody').innerHTML = d.results.map(row => `
      <tr>
        <td class="smiles">${row.smiles}</td>
        <td class="num">${row.rt_real.toFixed(3)}</td>
        <td class="num">${row.rt_pred.toFixed(3)}</td>
        <td class="num">${row.abs_err.toFixed(3)}</td>
      </tr>`).join('');

    panelShow('panel-results');
    showToast('Sheet predicted successfully.');
  }catch(e){
    showToast(e.message, true);
  }finally{
    btn.disabled = false;
    btn.innerHTML = '<span id="run-page-label">Predict this sheet</span>';
    refreshPageStatsButton();
  }
}

const STAT_LABELS = {
  mae:'MAE', rmse:'RMSE', r2:'R²', mre:'MRE', medae:'MedAE',
  slope:'Regression slope', intercept:'Regression intercept',
  pearson:'Pearson correlation', spearman:'Spearman correlation'
};

async function showStats(){
  if(!activeScenario){ showToast('First choose a scenario.', true); return; }
  if(isManual){ showToast('Scenario G has no accumulated statistics - it predicts single molecules on demand.', true); return; }
  try{
    const r = await fetch('/api/scenario_stats', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({scenario: activeScenario})
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Error');

    document.getElementById('stats-scn-label').textContent = activeScenario;
    document.getElementById('stats-sub').textContent =
      d.description + ' · ' + d.stats.n + ' accumulated points from ' + d.processed_sheets.length + ' sheet(s).';

    if(d.stats.n === 0){
      document.getElementById('stats-grid').innerHTML =
        '<div style="grid-column:1/-1;color:var(--subtle);font-size:13.5px">No predictions accumulated yet in this scenario.</div>';
    } else {
      document.getElementById('stats-grid').innerHTML = Object.entries(STAT_LABELS).map(([k,label]) => {
        let v = d.stats[k];
        let txt = (v === null || v === undefined || Number.isNaN(v)) ? '-' : v.toFixed(4);
        if(k === 'mre') txt = (Number.isNaN(v) ? '-' : (v*100).toFixed(2) + '%');
        return `<div class="stat-card"><div class="stat-name">${label}</div><div class="stat-val">${txt}</div></div>`;
      }).join('');
    }

    renderChart(d.points);

    const isScenarioF = (activeScenario === 'F');
    if(isScenarioF){
      try{
        const ms = await fetch('/api/model_sheets?scenario=F');
        const msd = await ms.json();
        const modelSheets = msd.sheets || [];
        const sel = document.getElementById('model-sheet-select');
        sel.innerHTML = modelSheets.map(s => `<option value="${s}">${s}</option>`).join('');
        const hasModel = modelSheets.length > 0;
        document.getElementById('model-insights-row').classList.toggle('hidden', !hasModel);
        document.getElementById('btn-tree-png').classList.toggle('hidden', !hasModel);
      }catch(e){
        document.getElementById('model-insights-row').classList.add('hidden');
        document.getElementById('btn-tree-png').classList.add('hidden');
      }
    } else {
      document.getElementById('model-insights-row').classList.add('hidden');
      document.getElementById('btn-tree-png').classList.add('hidden');
    }

    panelShow('panel-stats');
  }catch(e){ showToast(e.message, true); }
}

function _selectedModelSheet(){
  const sel = document.getElementById('model-sheet-select');
  return sel && sel.value ? sel.value : '';
}

function downloadTreePNG(){
  window.open('/api/model_tree_png?scenario=F&sheet=' + encodeURIComponent(_selectedModelSheet()), '_blank');
}

function renderChart(points){
  const ctx = document.getElementById('chart').getContext('2d');
  if(currentChart) currentChart.destroy();
  if(points.length === 0){ currentChart = null; return; }

  const reals = points.map(p => p.real);
  const preds = points.map(p => p.pred);
  const minV = Math.min(...reals, ...preds);
  const maxV = Math.max(...reals, ...preds);
  const pad = (maxV - minV) * 0.05 || 1;
  const lo = minV - pad, hi = maxV + pad;

  currentChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        { type:'line', label:'Ideal (y = x)', data:[{x:lo,y:lo},{x:hi,y:hi}],
          borderColor:'#c00', borderWidth:1.5, borderDash:[6,4], pointRadius:0, fill:false, order:1 },
        { type:'scatter', label:'Predictions',
          data: points.map(p => ({x:p.real, y:p.pred, sheet:p.sheet, smiles:p.smiles})),
          backgroundColor:'rgba(0,113,227,0.65)', pointRadius:4, order:2 }
      ]
    },
    options: {
      responsive:true,
      plugins:{
        legend:{position:'top'},
        tooltip:{callbacks:{label:(item)=>{
          if(item.dataset.label === 'Ideal (y = x)') return 'y = x';
          const p = item.raw;
          return `real=${p.x.toFixed(3)}  pred=${p.y.toFixed(3)}  (${p.sheet})`;
        }}}
      },
      scales:{
        x:{title:{display:true,text:'Real RT (min)'},min:lo,max:hi},
        y:{title:{display:true,text:'Predicted RT (min)'},min:lo,max:hi}
      }
    }
  });
}

function downloadCSV(){ window.open('/api/scenario_download_csv?scenario=' + activeScenario, '_blank'); }
function downloadPDF(){ window.open('/api/scenario_download_pdf?scenario=' + activeScenario, '_blank'); }
function downloadChart(){
  if(!currentChart){ showToast('No chart to download.', true); return; }
  const a = document.createElement('a');
  a.href = currentChart.toBase64Image();
  a.download = 'scenario_' + activeScenario + '_chart.png';
  a.click();
}

async function resetScenario(){
  const ok = await customConfirm('Delete all accumulated data for scenario ' + activeScenario + '?');
  if(!ok) return;
  try{
    const r = await fetch('/api/scenario_reset', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({scenario: activeScenario})
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Error');
    showToast('Scenario data reset.');
    backToPanel('panel-page');
  }catch(e){ showToast(e.message, true); }
}

async function pollStatus(){
  try{
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('req-counter').textContent = d.requests_made;
    document.getElementById('model-name').textContent = d.model || '-';
    const fpEl = document.getElementById('fp-backend');
    if(d.rdkit_available){
      fpEl.textContent = 'RDKit ' + (d.rdkit_version || '') + ' (real Morgan fingerprint)';
      fpEl.style.color = 'var(--success)';
    } else {
      fpEl.textContent = 'hash fallback - ' + (d.rdkit_error || 'RDKit no disponible');
      fpEl.style.color = 'var(--error)';
    }
  }catch(e){}
}
pollStatus();
setInterval(pollStatus, 4000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    def open_browser():
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:5000")

    print(f"🚀 Modelo activo: {MODEL_NAME}")
    print(f"🚀 Claves cargadas: {len(API_KEYS)}")
    print(f"🚀 Max output tokens base: {MAX_OUTPUT_TOKENS_BASE}")
    print(f"🚀 Max output tokens por fila: {MAX_OUTPUT_TOKENS_PER_ROW}")
    threading.Thread(target=open_browser).start()
    app.run(debug=False)