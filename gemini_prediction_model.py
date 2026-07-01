import os
import re
import time
import threading
import webbrowser
import tempfile
from io import BytesIO
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# ============================================================
# 🔑 CONFIGURACIÓN DE API KEY - CORREGIDA
# ============================================================
load_dotenv()

# Obtener y LIMPIAR la API key
API_KEY = os.getenv("API_KEY")
if API_KEY:
    API_KEY = API_KEY.strip()  # Eliminar espacios en blanco
else:
    # Intentar cargar desde variable de entorno del sistema
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if API_KEY:
        API_KEY = API_KEY.strip()

# Verificar que existe
if not API_KEY:
    print("⚠️ ERROR: No se encontró API_KEY en .env")
    print("   Asegúrate de que el archivo .env existe y contiene:")
    print("   API_KEY=AQ.Ab8RN6KUV8zn5FHSYH2ctIBxz09UE_GSTKkeEXNeotP1XZRb2A")
else:
    print(f"✅ API_KEY cargada correctamente: {API_KEY[:10]}...")

MODEL_NAME = "gemini-2.5-flash"
SEED = 42
CONTEXT_FRAC = 0.80  # 80% para contexto
MAX_CONTEXT_ROWS = 80
MAX_CONTEXT_SMILES_LEN = 160
MAX_QUERY_SMILES_LEN = 500
MAX_TRIES = 10
SHOW_80_ROWS = 300

# Data
state = {
    "chat": None,
    "df80": None,
    "col_rt": None,
    "col_smiles": None,
    "xls_bytes": None,
    "paper_path": None,
    "logs": [],
}

# Function for debugging.
def log(msg):
    state["logs"].append(msg)
    state["logs"] = state["logs"][-200:]


# Data preparation methods.


# Search for columns in Excel
def buscar_columna(df: pd.DataFrame, posibles: List[str]) -> Optional[str]:
    cols = {c.lower().strip(): c for c in df.columns}
    for p in posibles:
        if p.lower().strip() in cols:
            return cols[p.lower().strip()]
    for c in df.columns:
        for p in posibles:
            if p.lower() in c.lower():
                return c
    return None


# Get all important columns from the dataset
def obtener_columnas_importantes(df: pd.DataFrame) -> List[str]:
    """
    Identify all relevant chromatographic columns in the dataset.
    Returns a list of column names that contain useful information.
    """
    important_patterns = [
        # Column information
        "column.name", "column.usp.code", "column.length", "column.id", 
        "column.particle.size", "column.temperature", "column.flowrate",
        
        # Eluent A
        "eluent.A.h2o", "eluent.A.formic", "eluent.A.pH", 
        "eluent.A.acetic", "eluent.A.trifluoroacetic", "eluent.A.phosphor",
        "eluent.A.nh4ac", "eluent.A.nh4form", "eluent.A.nh4carb",
        "eluent.A.nh4bicarb", "eluent.A.nh4f", "eluent.A.nh4oh",
        "eluent.A.trieth", "eluent.A.triprop", "eluent.A.tribut",
        "eluent.A.nndimethylhex", "eluent.A.medronic",
        
        # Eluent B
        "eluent.B.meoh", "eluent.B.formic", "eluent.B.pH",
        "eluent.B.acetic", "eluent.B.trifluoroacetic", "eluent.B.phosphor",
        "eluent.B.nh4ac", "eluent.B.nh4form", "eluent.B.nh4carb",
        "eluent.B.nh4bicarb", "eluent.B.nh4f", "eluent.B.nh4oh",
        "eluent.B.trieth", "eluent.B.triprop", "eluent.B.tribut",
        "eluent.B.nndimethylhex", "eluent.B.medronic",
        
        # Eluent C (if present)
        "eluent.C.formic", "eluent.C.acetic", "eluent.C.trifluoroacetic",
        "eluent.C.phosphor", "eluent.C.nh4ac", "eluent.C.nh4form",
        "eluent.C.nh4carb", "eluent.C.nh4bicarb", "eluent.C.nh4f",
        "eluent.C.nh4oh", "eluent.C.trieth", "eluent.C.triprop",
        "eluent.C.tribut", "eluent.C.nndimethylhex", "eluent.C.medronic",
        
        # Eluent D (if present)
        "eluent.D.formic", "eluent.D.acetic", "eluent.D.trifluoroacetic",
        "eluent.D.phosphor", "eluent.D.nh4ac", "eluent.D.nh4form",
        "eluent.D.nh4carb", "eluent.D.nh4bicarb", "eluent.D.nh4f",
        "eluent.D.nh4oh", "eluent.D.trieth", "eluent.D.triprop",
        "eluent.D.tribut", "eluent.D.nndimethylhex", "eluent.D.medronic",
        
        # Gradient
        "gradient.start.A", "gradient.end.B"
    ]
    
    present = []
    for pattern in important_patterns:
        col = buscar_columna(df, [pattern])
        if col is not None:
            present.append(col)
    
    return present

# Prepare the dataframe that is later used for context and selection
def preparar_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, List[str]]:
    col_rt = buscar_columna(df, ["rt", "rt_min", "retention_time"])
    col_smiles = buscar_columna(df, ["pubchem.smiles.canonical", "smiles"])
    if col_rt is None or col_smiles is None:
        raise RuntimeError("No se encontraron columnas RT o SMILES.")
    
    df = df.copy()
    df[col_rt] = pd.to_numeric(df[col_rt], errors="coerce")
    df[col_smiles] = df[col_smiles].astype(str)
    df = df.dropna(subset=[col_rt, col_smiles])
    df = df[df[col_smiles].str.strip().ne("")]
    df = df.reset_index(drop=True)
    
    # Get all important columns
    important_cols = obtener_columnas_importantes(df)
    
    return df, col_rt, col_smiles, important_cols

# Divides the Excel into 80(context) and 20(prediction)
def split_80_20(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_shuf = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = int(len(df_shuf) * CONTEXT_FRAC)  # 80% para contexto
    return df_shuf.iloc[:n].reset_index(drop=True), df_shuf.iloc[n:].reset_index(drop=True)


# Build the context table that will be sent to Gemini based on the uploaded Excel file
def construir_tabla_contexto(ctx: pd.DataFrame, col_rt: str, col_smiles: str, important_cols: List[str] = None) -> str:
    # If no important columns provided, use default list
    if important_cols is None:
        important_cols = [
            "column.name", "column.usp.code", "column.length", "column.particle.size",
            "column.temperature", "column.flowrate",
            "eluent.A.h2o", "eluent.A.pH",
            "eluent.B.meoh", "eluent.B.pH",
            "gradient.start.A", "gradient.end.B"
        ]
    
    # Filter to only columns that actually exist in the dataframe
    present = [c for c in important_cols if c in ctx.columns]

    lines = []
    for _, r in ctx.head(MAX_CONTEXT_ROWS).iterrows():
        smi = str(r[col_smiles]).strip()[:MAX_CONTEXT_SMILES_LEN]
        rt = float(r[col_rt])
        
        # Build extras string with all available important columns
        extras_parts = []
        for c in present:
            val = r[c]
            if pd.notna(val):
                # Format the value nicely
                if isinstance(val, float):
                    extras_parts.append(f"{c}={val:.3f}" if val % 1 != 0 else f"{c}={int(val)}")
                else:
                    extras_parts.append(f"{c}={val}")
        
        extras = " | ".join(extras_parts)
        lines.append(f"{rt:.3f}\t{smi}\t{extras}")
    
    return "\n".join(lines)
# Gemini Methods.

def configurar_gemini() -> genai.GenerativeModel:
    global API_KEY  # Asegura que usa la variable global
    
    if not API_KEY:
        # Intentar recargar desde .env
        load_dotenv()
        API_KEY = os.getenv("API_KEY")
        if API_KEY:
            API_KEY = API_KEY.strip()
    
    if not API_KEY:
        raise RuntimeError("Falta API_KEY en .env")
    
    print(f"🔑 Configurando Gemini con: {API_KEY[:10]}...")  # Debug
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


# Read Gemini’s response, clean it, and merge it for final visualization
def leer_texto_seguro(resp: Any) -> str:
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            parts = getattr(resp.candidates[0].content, "parts", []) or []
            return "\n".join(p.text for p in parts if getattr(p, "text", "")).strip()
    except Exception:
        pass
    return ""


# Search the messages for the number of seconds before retrying
def parse_retry(err: Exception) -> float:
    s = str(err)
    m = re.search(r"Please retry in\s+([0-9.]+)s", s)
    if m:
        return float(m.group(1))
    m = re.search(r"retry_delay\s*{\s*seconds:\s*([0-9]+)", s)
    if m:
        return float(m.group(1))
    return 4.0


# Send the message, read the text, and attempt the retries with up to 10 seconds between waits
def enviar(chat, mensaje, log_fn=None) -> str:
    last_err = None
    for attempt in range(MAX_TRIES):
        try:
            resp = chat.send_message(
                mensaje,
                generation_config={"temperature": 0.0, "max_output_tokens": 128},
            )
            texto = leer_texto_seguro(resp)
            if not texto:
                wait = min(10.0, 1.0 + attempt * 0.9)
                if log_fn:
                    log_fn(f"[EMPTY] retry {attempt+1}/{MAX_TRIES} in {wait:.1f}s")
                time.sleep(wait)
                continue
            return texto.strip()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            wait = parse_retry(e) if ("429" in msg or "quota" in msg) else min(10.0, 1.0 + attempt * 0.9)
            if log_fn:
                log_fn(f"[ERR] {e} retry in {wait:.1f}s")
            time.sleep(max(1.0, wait))
    raise last_err


# Find the first number in the response or return an error
def extraer_numero(texto: str) -> float:
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", texto or "")
    if not m:
        raise ValueError(f"No se encontró número en: {texto!r}")
    return float(m.group(1))


# Promt Methods

def prompt_1(train_table: str) -> str:
    return (
        "You predict chromatography retention time (RT) in minutes.\n"
        "You are given example pairs (rt_minutes\\tsmiles) from the SAME chromatographic system.\n"
        "Infer the time scale and patterns from these examples.\n\n"
        "OUTPUT RULE (for later predictions):\n"
        "- Reply ONLY one number with exactly 3 decimals.\n"
        "- No words. No units. No punctuation. No explanations.\n"
        "- If uncertain, still output your best estimate.\n\n"
        f"EXAMPLES (rt_minutes\\tsmiles):\n{train_table}\n\n"
        "Reply now with: 0.000"
    )


def prompt_1_no_excel() -> str:
    return (
        "You predict chromatography retention time (RT) in minutes.\n"
        "You have no example pairs from the chromatographic system.\n"
        "Use your chemical knowledge and any context provided (scientific paper) to estimate RT.\n\n"
        "OUTPUT RULE (for later predictions):\n"
        "- Reply ONLY one number with exactly 3 decimals.\n"
        "- No words. No units. No punctuation. No explanations.\n"
        "- If uncertain, still output your best estimate.\n\n"
        "Reply now with: 0.000"
    )


def prompt_2(smiles: str, row: pd.Series = None) -> str:
    smi = (smiles or "").strip()[:MAX_QUERY_SMILES_LEN]

    extra_cols = [
        "column.name", "column.usp.code", "column.length", "column.particle.size",
        "column.temperature", "column.flowrate",
        "eluent.A.h2o", "eluent.A.pH",
        "eluent.B.meoh", "eluent.B.pH",
        "gradient.start.A", "gradient.end.B"
    ]

    conditions = ""
    if row is not None:
        present = [c for c in extra_cols if c in row.index and pd.notna(row[c])]
        if present:
            conditions = "\nChromatographic conditions: " + " | ".join(f"{c}={row[c]}" for c in present)

    return (
        "Return ONLY one number with exactly 3 decimals.\n"
        "No words. No units. No punctuation.\n"
        f"SMILES: {smi}"
        f"{conditions}"
    )


# Aplication modules

# Generate de app and allow CORS
app = Flask(__name__)
CORS(app)


# We define the home execution if coming from the /# route.
@app.route("/")
def home():
    return HTML


# Extract the names of the sheets from the uploaded Excel file.
@app.route("/api/upload_excel", methods=["POST"])
def upload_excel():
    f = request.files["file"]
    state["xls_bytes"] = f.read()
    xls = pd.ExcelFile(BytesIO(state["xls_bytes"]))
    return jsonify({"ok": True, "sheets": xls.sheet_names})


# We temporarily store the .pdf file for sending it to Gemini.
@app.route("/api/upload_paper", methods=["POST"])
def upload_paper():
    f = request.files["file"]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(f.read())
    tmp.flush()
    tmp.close()
    state["paper_path"] = tmp.name
    log(f"Paper recibido: {f.filename}")
    return jsonify({"ok": True, "name": f.filename})


# “Function that sends the data to Gemini (reads the Excel file, uploads the paper, creates the context table) and saves the response.
@app.route("/api/load", methods=["POST"])
def load():
    sheet = request.json["sheet"]
    if not state["xls_bytes"]:
        return jsonify({"error": "No hay Excel cargado"}), 400

    raw = pd.read_excel(BytesIO(state["xls_bytes"]), sheet_name=sheet)
    df, col_rt, col_smiles, important_cols = preparar_df(raw)  # ✅ Now returns important_cols

    
    ctx80, rest20 = split_80_20(df)
    train_table = construir_tabla_contexto(ctx80, col_rt, col_smiles, important_cols)  # ✅ Now includes all columns

    model = configurar_gemini()
    chat = model.start_chat(history=[])

    if state["paper_path"]:
        try:
            paper_handle = genai.upload_file(path=state["paper_path"], mime_type="application/pdf")
            init_msg = [paper_handle, prompt_1(train_table)]
            log("Enviando PROMPT 1 + paper...")
        except Exception as e:
            log(f"Error al subir paper: {e}. Continuando sin paper.")
            init_msg = prompt_1(train_table)
    else:
        init_msg = prompt_1(train_table)
        log("Enviando PROMPT 1 (sin paper)...")

    resp = enviar(chat, init_msg, log_fn=log)
    log(f"PROMPT 1 OK. Resp: {resp[:60]}")

    state.update({
        "chat": chat,
        "df80": rest20,  
        "col_rt": col_rt,
        "col_smiles": col_smiles,
    })

    rows = rest20[[col_rt, col_smiles]].head(SHOW_80_ROWS).copy()
    rows.insert(0, "row_id", rows.index)
    rows[col_smiles] = rows[col_smiles].str.slice(0, 80)

    return jsonify({
        "ok": True,
        "total20": len(rest20),  
        "total80": len(ctx80),   
        "col_rt": col_rt,
        "col_smiles": col_smiles,
        "rows": rows.to_dict("records"),
    })
    
# Function that handles scenarios without Excel, either with a paper or with nothing.
@app.route("/api/init_no_excel", methods=["POST"])
def init_no_excel():
    try:
        # Obtener el SMILES del frontend
        data = request.get_json()
        smiles = data.get("smiles", "").strip() if data else ""
        
        if not smiles:
            return jsonify({"error": "SMILES es requerido"}), 400
        
        log(f"Inicializando Gemini sin Excel para SMILES: {smiles[:50]}...")
        
        model = configurar_gemini()
        chat = model.start_chat(history=[])

        # 🔧 MANEJO DEL PAPER CORREGIDO
        if state["paper_path"]:
            try:
                # Verificar que el archivo existe
                if not os.path.exists(state["paper_path"]):
                    log(f"⚠️ El archivo paper no existe en: {state['paper_path']}")
                    raise FileNotFoundError(f"No se encuentra el archivo: {state['paper_path']}")
                
                # Verificar que el archivo no está vacío
                if os.path.getsize(state["paper_path"]) == 0:
                    log("⚠️ El archivo paper está vacío")
                    raise ValueError("El archivo PDF está vacío")
                
                log(f"📄 Subiendo paper: {state['paper_path']} ({os.path.getsize(state['paper_path'])} bytes)")
                paper_handle = genai.upload_file(path=state["paper_path"], mime_type="application/pdf")
                init_msg = [paper_handle, prompt_1_no_excel()]
                log("Enviando PROMPT 1 + paper...")
            except Exception as e:
                log(f"❌ Error al subir paper: {str(e)}. Continuando sin paper.")
                init_msg = prompt_1_no_excel()
                log("Enviando PROMPT 1 (sin paper por error)...")
        else:
            init_msg = prompt_1_no_excel()
            log("Enviando PROMPT 1 (sin paper ni Excel)...")

        resp = enviar(chat, init_msg, log_fn=log)
        log(f"PROMPT 1 OK. Resp: {resp[:60]}")

        state.update({
            "chat": chat,
            "df80": None,
            "col_rt": None,
            "col_smiles": None,
        })

        return jsonify({"ok": True})
        
    except Exception as e:
        log(f"ERROR en init_no_excel: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# Send the manually entered SMILES, read the response, and truncate it to keep only the number.
@app.route("/api/predict_smiles", methods=["POST"])
def predict_smiles():
    smiles = request.json.get("smiles", "").strip()
    if not smiles:
        return jsonify({"error": "SMILES vacío"}), 400
    if state["chat"] is None:
        return jsonify({"error": "Gemini no inicializado"}), 400

    log(f"Prediciendo SMILES manual: {smiles[:60]}...")
    raw = enviar(state["chat"], prompt_2(smiles), log_fn=log)
    rt_pred = extraer_numero(raw)
    log(f"pred={rt_pred:.3f}")

    return jsonify({"ok": True, "rt_pred": rt_pred, "smiles": smiles})


# Send the Excel file, generate the prompt, and read the response received from Gemini.
@app.route("/api/predict", methods=["POST"])
def predict():
    row_id = int(request.json["row_id"])
    if state["chat"] is None or state["df80"] is None:
        return jsonify({"error": "Primero carga una hoja"}), 400

    df80 = state["df80"]
    col_rt = state["col_rt"]
    col_smiles = state["col_smiles"]

    smi = str(df80.loc[row_id, col_smiles]).strip()
    rt_real = float(df80.loc[row_id, col_rt])

    log(f"Prediciendo row_id={row_id}...")
    row = df80.loc[row_id]
    raw = enviar(state["chat"], prompt_2(smi, row=row), log_fn=log)
    rt_pred = extraer_numero(raw)
    abs_err = abs(rt_pred - rt_real)

    log(f"real={rt_real:.3f} pred={rt_pred:.3f} err={abs_err:.3f}")

    return jsonify({
        "ok": True,
        "row_id": row_id,
        "rt_real": rt_real,
        "rt_pred": rt_pred,
        "abs_err": abs_err,
        "smiles": smi,
    })


# It retrieves the logs that the system keeps storing.
@app.route("/api/logs")
def get_logs():
    return jsonify({"logs": state["logs"]})



HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Retention Time Predictor</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #ffffff;
    --surface:   #f5f5f7;
    --border:    #d2d2d7;
    --blue:      #0071e3;
    --blue-dark: #0056b3;
    --text:      #1d1d1f;
    --subtle:    #6e6e73;
    --success:   #1a8917;
    --error:     #c00;
    --radius:    16px;
    --font:      -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
  }

  body { font-family: var(--font); background: var(--bg); color: var(--text); min-height: 100vh; }

  header {
    display: flex; align-items: center; gap: 12px;
    padding: 24px 48px;
    border-bottom: 1px solid var(--border);
  }
  .logo-dot {
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg, #0071e3 0%, #42a5f5 100%);
  }
  header h1 { font-size: 20px; font-weight: 600; letter-spacing: -.3px; }
  header span { font-size: 13px; color: var(--subtle); margin-left: 4px; }

  .stepper {
    display: flex; align-items: center;
    padding: 32px 48px 0;
    max-width: 860px; margin: 0 auto;
  }
  .step-item {
    display: flex; flex-direction: column; align-items: center;
    gap: 6px; flex: 1; position: relative;
  }
  .step-item:not(:last-child)::after {
    content: ""; position: absolute;
    top: 17px;
    left: calc(50% + 17px); right: calc(-50% + 17px);
    height: 2px; background: var(--border); transition: background .3s;
  }
  .step-item.done:not(:last-child)::after { background: var(--blue); }
  .step-bubble {
    width: 34px; height: 34px; border-radius: 50%;
    border: 2px solid var(--border); background: var(--bg);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 600; color: var(--subtle);
    transition: all .3s; position: relative; z-index: 1;
  }
  .step-item.active .step-bubble { border-color: var(--blue); background: var(--blue); color: #fff; }
  .step-item.done .step-bubble   { border-color: var(--blue); background: var(--blue); color: #fff; }
  .step-item.done .step-bubble::after { content: "✓"; position: absolute; font-size: 14px; }
  .step-item.done .step-num { display: none; }
  .step-label { font-size: 11px; color: var(--subtle); text-align: center; font-weight: 500; letter-spacing: .2px; }
  .step-item.active .step-label { color: var(--blue); }
  .step-item.done .step-label   { color: var(--text); }

  .content { max-width: 860px; margin: 40px auto 80px; padding: 0 48px; }

  .panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 36px 40px;
    margin-bottom: 20px; display: none;
    animation: fadeUp .35s ease;
  }
  .panel.active { display: block; }
  @keyframes fadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }

  .panel-title { font-size: 22px; font-weight: 700; letter-spacing: -.4px; margin-bottom: 6px; }
  .panel-sub   { font-size: 14px; color: var(--subtle); margin-bottom: 28px; line-height: 1.5; }

  .drop-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 28px; }
  .drop-zone {
    background: var(--bg); border: 2px dashed var(--border);
    border-radius: 12px; padding: 32px 20px; text-align: center;
    cursor: pointer; transition: border-color .2s, background .2s; position: relative;
  }
  .drop-zone:hover { border-color: var(--blue); background: #f0f7ff; }
  .drop-zone.filled { border-style: solid; border-color: var(--success); background: #f3faf3; }
  .drop-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .drop-icon  { font-size: 32px; margin-bottom: 10px; }
  .drop-label { font-size: 14px; font-weight: 600; color: var(--text); }
  .drop-hint  { font-size: 12px; color: var(--subtle); margin-top: 4px; }
  .drop-name  { font-size: 12px; color: var(--success); margin-top: 8px; font-weight: 600; }

  .btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 12px 28px; border: none; border-radius: 980px;
    font-size: 15px; font-weight: 600; cursor: pointer;
    transition: background .2s, transform .1s, opacity .2s;
    font-family: var(--font);
  }
  .btn:active { transform: scale(.97); }
  .btn-primary { background: var(--blue); color: #fff; }
  .btn-primary:hover { background: var(--blue-dark); }
  .btn-primary:disabled { opacity: .45; cursor: not-allowed; }
  .btn-ghost { background: transparent; color: var(--blue); border: 1.5px solid var(--blue); }
  .btn-ghost:hover { background: #f0f7ff; }

  select {
    width: 100%; padding: 12px 16px;
    border: 1.5px solid var(--border); border-radius: 10px;
    font-size: 15px; font-family: var(--font);
    background: var(--bg); color: var(--text); appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%236e6e73' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 14px center;
    margin-bottom: 20px; transition: border-color .2s;
  }
  select:focus { outline: none; border-color: var(--blue); }

  /* SMILES / RT input fields */
  .field-wrap { margin-bottom: 18px; }
  .field-wrap label {
    display: block; font-size: 13px; font-weight: 600;
    color: var(--subtle); text-transform: uppercase; letter-spacing: .5px;
    margin-bottom: 8px;
  }
  .field-wrap textarea,
  .field-wrap input[type=number] {
    width: 100%; padding: 14px 16px;
    border: 1.5px solid var(--border); border-radius: 10px;
    font-size: 14px; font-family: var(--font);
    background: var(--bg); color: var(--text);
    transition: border-color .2s;
  }
  .field-wrap textarea {
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 13px;
    resize: vertical; min-height: 80px; line-height: 1.6;
  }
  .field-wrap textarea:focus,
  .field-wrap input[type=number]:focus { outline: none; border-color: var(--blue); }
  .field-wrap textarea::placeholder,
  .field-wrap input[type=number]::placeholder { color: var(--subtle); }
  .field-hint {
    font-size: 11px; color: var(--subtle); margin-top: 5px;
  }

  .loader-wrap { display: none; flex-direction: column; align-items: center; gap: 14px; padding: 20px 0; }
  .loader-wrap.show { display: flex; }
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid var(--border); border-top-color: var(--blue);
    border-radius: 50%; animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loader-msg { font-size: 14px; color: var(--subtle); font-weight: 500; }

  .table-wrap {
    overflow-x: auto; border-radius: 10px;
    border: 1px solid var(--border); background: var(--bg); margin-bottom: 24px;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead tr { background: var(--surface); }
  th {
    padding: 12px 16px; text-align: left; font-weight: 600;
    color: var(--subtle); font-size: 11px; text-transform: uppercase;
    letter-spacing: .6px; border-bottom: 1px solid var(--border);
  }
  td {
    padding: 11px 16px; border-bottom: 1px solid var(--border);
    color: var(--text); font-variant-numeric: tabular-nums;
  }
  tr:last-child td { border-bottom: none; }
  tr.selected td { background: #e8f1fd; }
  tr:not(.selected):hover td { background: #fafafa; cursor: pointer; }
  .smiles-cell { font-family: "SF Mono","Fira Code",monospace; font-size: 11px; color: var(--subtle); }

  .result-card {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 12px; padding: 28px 32px;
    animation: fadeUp .3s ease;
  }
  .result-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 24px; margin-bottom: 20px; }
  .result-grid.two-col { grid-template-columns: repeat(2,1fr); }
  .result-kpi label {
    font-size: 11px; text-transform: uppercase; letter-spacing: .6px;
    color: var(--subtle); font-weight: 600; display: block; margin-bottom: 6px;
  }
  .result-kpi .value {
    font-size: 32px; font-weight: 700; letter-spacing: -1px;
    font-variant-numeric: tabular-nums;
  }
  .result-kpi.error-kpi .value { color: var(--blue); }
  .result-smiles {
    font-family: "SF Mono","Fira Code",monospace; font-size: 11px; color: var(--subtle);
    background: var(--surface); padding: 10px 14px; border-radius: 8px;
    word-break: break-all; line-height: 1.6;
  }

  /* Mode badge shown in step 2 */
  .mode-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #e8f1fd; color: var(--blue);
    border-radius: 980px; padding: 5px 14px;
    font-size: 12px; font-weight: 600; margin-bottom: 22px;
  }

  /* Upload status indicator in step 1 */
  .upload-status {
    font-size: 13px; color: var(--subtle);
    margin-bottom: 20px; min-height: 20px;
    display: flex; align-items: center; gap: 8px;
  }
  .upload-status.uploading { color: var(--blue); }

  #toast {
    position: fixed; bottom: 32px; left: 50%;
    transform: translateX(-50%) translateY(80px);
    background: #1d1d1f; color: #fff;
    padding: 12px 24px; border-radius: 980px;
    font-size: 14px; font-weight: 500;
    opacity: 0; transition: opacity .3s, transform .3s; z-index: 999; white-space: nowrap;
  }
  #toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  #toast.error-toast { background: var(--error); }

  .divider { height: 1px; background: var(--border); margin: 24px 0; }
  .section-label {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .5px; color: var(--subtle); margin-bottom: 10px;
  }
</style>
</head>
<body>

<header>
  <div class="logo-dot"></div>
  <h1>Retention Time Predictor</h1>
  <span>Powered by Gemini API</span>
</header>

<div class="stepper" id="stepper">
  <div class="step-item active" id="si-1">
    <div class="step-bubble"><span class="step-num">1</span></div>
    <div class="step-label">Files</div>
  </div>
  <div class="step-item" id="si-2">
    <div class="step-bubble"><span class="step-num">2</span></div>
    <div class="step-label">Configuration</div>
  </div>
  <div class="step-item" id="si-3">
    <div class="step-bubble"><span class="step-num">3</span></div>
    <div class="step-label">Data</div>
  </div>
  <div class="step-item" id="si-4">
    <div class="step-bubble"><span class="step-num">4</span></div>
    <div class="step-label">Selection</div>
  </div>
  <div class="step-item" id="si-5">
    <div class="step-bubble"><span class="step-num">5</span></div>
    <div class="step-label">Result</div>
  </div>
</div>

<div class="content">

  <!-- STEP 1: Upload files-->
  <div class="panel active" id="p1">
    <div class="panel-title">Upload your files</div>
    <div class="panel-sub">The paper and excel are both optional. You can use any combination or none.</div>

    <div class="drop-row">
      <div class="drop-zone" id="dz-excel"
           ondragover="dzOver(event,'dz-excel')" ondragleave="dzLeave('dz-excel')" ondrop="dzDrop(event,'excel')">
        <input type="file" id="file-excel" accept=".xlsx" onchange="handleExcel(this)"/>
        <div class="drop-icon">📊</div>
        <div class="drop-label">Dataset Excel</div>
        <div class="drop-hint">.xlsx — drag or click here</div>
        <div class="drop-name" id="name-excel"></div>
      </div>

      <div class="drop-zone" id="dz-paper"
           ondragover="dzOver(event,'dz-paper')" ondragleave="dzLeave('dz-paper')" ondrop="dzDrop(event,'paper')">
        <input type="file" id="file-paper" accept=".pdf" onchange="handlePaper(this)"/>
        <div class="drop-icon">📄</div>
        <div class="drop-label">Scientific Paper</div>
        <div class="drop-hint">.pdf — optional</div>
        <div class="drop-name" id="name-paper"></div>
      </div>
    </div>

    <!-- Upload in-progress indicator -->
    <div class="upload-status" id="upload-status"></div>

    <div class="loader-wrap" id="loader-1">
      <div class="spinner"></div>
      <div class="loader-msg" id="loader-1-msg">Uploading…</div>
    </div>

    <button class="btn btn-primary" id="btn-continue-1" onclick="continueStep1()">
      Continue →
    </button>
  </div>


  <!-- ═ STEP 2: Configure dynamic content -->
  <div class="panel" id="p2">
    <div class="panel-title" id="p2-title">Configure</div>
    <div class="panel-sub" id="p2-sub"></div>

    <div id="p2-mode-badge" class="mode-badge" style="display:none;"></div>

    <!-- Sheet selector (Excel mode) -->
    <div id="p2-sheet-wrap" style="display:none;">
      <div class="section-label">Available pages</div>
      <select id="sheet-select"></select>
    </div>

    <!-- SMILES + optional RT real (no-Excel mode) -->
    <div id="p2-smiles-wrap" style="display:none;">
      <div class="field-wrap">
        <label>SMILES</label>
        <textarea id="smiles-manual" placeholder="Introduce the SMILES formula..."></textarea>
      </div>
      <div class="field-wrap">
        <label>Real RT (min) — optional</label>
        <input type="number" id="rt-real-manual" step="any" min="0"
               placeholder="e.g. 4.231  (leave blank if unknown)"/>
        <div class="field-hint">If provided, the absolute error will be computed at the end.</div>
      </div>
    </div>

    <div class="loader-wrap" id="loader-2">
      <div class="spinner"></div>
      <div class="loader-msg" id="loader-2-msg">Sending context to Gemini…</div>
    </div>

    <div style="display:flex;gap:12px;">
      <button class="btn btn-ghost" onclick="goStep(1)">← Back</button>
      <button class="btn btn-primary" id="btn-step2-action" onclick="step2Action()">Continue →</button>
    </div>
  </div>


  <!-- STEP 3: Data table (Excel mode only)  -->
  <div class="panel" id="p3">
    <div class="panel-title">Available data</div>
    <div class="panel-sub" id="p3-sub">The remaining 20% of the dataset (to predict). Click on a row to select it.</div>

    <div class="table-wrap">
      <table id="data-table">
        <thead><tr><th>Row ID</th><th id="th-rt">RT</th><th id="th-smiles">SMILES</th></tr></thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>

    <div style="display:flex;gap:12px;">
      <button class="btn btn-ghost" onclick="goStep(2)">← Back</button>
      <button class="btn btn-primary" id="btn-to-step4" onclick="goStep(4)" disabled>Select row →</button>
    </div>
  </div>


  <!--  STEP 4: Confirm & predict  -->
  <div class="panel" id="p4">
    <div class="panel-title">Confirm your selection</div>
    <div class="panel-sub" id="p4-sub">Review the selected row and run the prediction.</div>

    <!-- Row preview (Excel mode) -->
    <div id="p4-row-wrap" class="table-wrap" style="margin-bottom:28px;display:none;">
      <table>
        <thead><tr><th>Row ID</th><th>Real RT</th><th>SMILES</th></tr></thead>
        <tbody id="selected-preview"></tbody>
      </table>
    </div>

    <!-- SMILES + optional real RT (no-Excel mode) -->
    <div id="p4-smiles-wrap" style="display:none;margin-bottom:28px;">
      <div class="field-wrap">
        <label>SMILES to predict</label>
        <textarea id="smiles-predict" placeholder="Introduce the SMILES formula…"></textarea>
      </div>
      <div class="field-wrap">
        <label>Real RT (min) — optional</label>
        <input type="number" id="rt-real-predict" step="any" min="0"
               placeholder="e.g. 4.231  (leave blank if unknown)"/>
        <div class="field-hint">If provided, the absolute error will be computed at the end.</div>
      </div>
    </div>

    <div class="loader-wrap" id="loader-4">
      <div class="spinner"></div>
      <div class="loader-msg">Asking Gemini…</div>
    </div>

    <div style="display:flex;gap:12px;">
      <button class="btn btn-ghost" id="btn-back-step4" onclick="backFromStep4()">← Back</button>
      <button class="btn btn-primary" id="btn-predict" onclick="predict()">Predict →</button>
    </div>
  </div>


  <!--  STEP 5: Result  -->
  <div class="panel" id="p5">
    <div class="panel-title">Result</div>
    <div class="panel-sub">Prediction Completed.</div>

    <div class="result-card" id="result-card">
      <div class="result-grid" id="result-grid">
        <div class="result-kpi" id="kpi-real">
          <label>Real RT (min)</label>
          <div class="value" id="r-real">—</div>
        </div>
        <div class="result-kpi">
          <label>Predicted RT (min)</label>
          <div class="value" id="r-pred">—</div>
        </div>
        <div class="result-kpi error-kpi" id="kpi-err">
          <label>Absolute error</label>
          <div class="value" id="r-err">—</div>
        </div>
      </div>
      <div class="divider"></div>
      <div class="section-label">SMILES</div>
      <div class="result-smiles" id="r-smiles">—</div>
    </div>

    <div style="display:flex;gap:12px;margin-top:24px;">
      <button class="btn btn-ghost" id="btn-predict-another" onclick="predictAnother()">← Predict another</button>
      <button class="btn btn-primary" onclick="restart()">New Session</button>
    </div>
  </div>

</div>

<div id="toast"></div>

<script>
// ── Global state ──
let sheets      = [];
let tableRows   = [];
let selectedRow = null;
let colRt       = "";
let colSmiles   = "";
let excelReady  = false;
let paperReady  = false;

// Tracks active uploads to block Continue until all finish
let uploadsInProgress = 0;

// mode: "excel" or "excel+paper"or "paper" or "none"
let mode = "none";

// Upload lock helpers 
// Call before starting an upload; disables Continue and shows status text.
function uploadStart(msg) {
  uploadsInProgress++;
  const btn = document.getElementById("btn-continue-1");
  btn.disabled = true;
  const st = document.getElementById("upload-status");
  st.textContent = "⏳ " + msg;
  st.className = "upload-status uploading";
}

// Call when an upload finishes .
function uploadEnd() {
  uploadsInProgress = Math.max(0, uploadsInProgress - 1);
  if (uploadsInProgress === 0) {
    const btn = document.getElementById("btn-continue-1");
    btn.disabled = false;
    const st = document.getElementById("upload-status");
    st.textContent = "";
    st.className = "upload-status";
  }
}

// Step control 
function goStep(n) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.getElementById("p" + n).classList.add("active");
  for (let i = 1; i <= 5; i++) {
    const si = document.getElementById("si-" + i);
    si.classList.remove("active", "done");
    if (i < n)  si.classList.add("done");
    if (i === n) si.classList.add("active");
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Toast 
function showToast(msg, isError=false) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className   = "show" + (isError ? " error-toast" : "");
  setTimeout(() => { t.className = ""; }, 3200);
}

// Drop zones for Excel and paper
function dzOver(e, id) {
  e.preventDefault();
  document.getElementById(id).style.borderColor = "#0071e3";
  document.getElementById(id).style.background  = "#f0f7ff";
}
function dzLeave(id) {
  const dz = document.getElementById(id);
  dz.style.borderColor = "";
  dz.style.background  = "";
}
function dzDrop(e, type) {
  e.preventDefault();
  dzLeave(type === "excel" ? "dz-excel" : "dz-paper");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (type === "excel") processExcel(file);
  else                  processPaper(file);
}

// Upload Excel logic.
function handleExcel(input) { if (input.files[0]) processExcel(input.files[0]); }
async function processExcel(file) {
  uploadStart("Uploading Excel…");
  setLoader(1, true, "Uploading Excel…");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/api/upload_excel", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Error uploading Excel");
    sheets = d.sheets;
    document.getElementById("name-excel").textContent = "✓ " + file.name;
    document.getElementById("dz-excel").classList.add("filled");
    excelReady = true;
    showToast("Excel loaded · " + sheets.length + " sheet(s)");
  } catch(e) {
    showToast(e.message, true);
  } finally {
    setLoader(1, false);
    uploadEnd();
  }
}

// Upload Paper logic
function handlePaper(input) { if (input.files[0]) processPaper(input.files[0]); }
async function processPaper(file) {
  uploadStart("Uploading paper…");
  setLoader(1, true, "Uploading paper…");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/api/upload_paper", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Error uploading paper");
    document.getElementById("name-paper").textContent = "✓ " + file.name;
    document.getElementById("dz-paper").classList.add("filled");
    paperReady = true;
    showToast("Paper ready");
  } catch(e) {
    showToast(e.message, true);
  } finally {
    setLoader(1, false);
    uploadEnd();
  }
}

// Continue from Step 1 to Step 2 
function continueStep1() {
  // Extra safety guard: should never fire while uploading, but just in case
  if (uploadsInProgress > 0) {
    showToast("Please wait — upload in progress", true);
    return;
  }

  if (excelReady && paperReady) mode = "excel+paper";
  else if (excelReady)          mode = "excel";
  else if (paperReady)          mode = "paper";
  else                          mode = "none";

  setupStep2();
  goStep(2);
}

function setupStep2() {
  const titleEl   = document.getElementById("p2-title");
  const subEl     = document.getElementById("p2-sub");
  const badgeEl   = document.getElementById("p2-mode-badge");
  const sheetWrap = document.getElementById("p2-sheet-wrap");
  const smilesWrap= document.getElementById("p2-smiles-wrap");
  const actionBtn = document.getElementById("btn-step2-action");

  sheetWrap.style.display  = "none";
  smilesWrap.style.display = "none";
  badgeEl.style.display    = "none";

  if (mode === "excel" || mode === "excel+paper") {
    titleEl.textContent  = "Select the sheet";
    subEl.textContent    = mode === "excel+paper"
      ? "Choose the Excel sheet. The 80% will be used as context along with the scientific paper."
      : "Choose the Excel sheet. The 80% will be used as context for Gemini.";
    badgeEl.textContent  = mode === "excel+paper" ? "📊 Excel + 📄 Paper" : "📊 Excel only";
    badgeEl.style.display= "inline-flex";

    const sel = document.getElementById("sheet-select");
    sel.innerHTML = sheets.map(s => `<option value="${s}">${s}</option>`).join("");
    sheetWrap.style.display = "block";
    actionBtn.textContent   = "Load and send context →";

  } else {
    titleEl.textContent  = "Introduce the compound";
    subEl.textContent    = mode === "paper"
      ? "No Excel dataset. Gemini will use the scientific paper as context. Enter the SMILES to predict."
      : "No files uploaded. Gemini will use its general knowledge. Enter the SMILES to predict.";
    badgeEl.textContent  = mode === "paper" ? "📄 Paper only" : "Without files";
    badgeEl.style.display= "inline-flex";

    smilesWrap.style.display = "block";
    actionBtn.textContent    = "Start Gemini →";
  }
}

// Step 2 action button 
async function step2Action() {
  if (mode === "excel" || mode === "excel+paper") {
    await loadSheet();
  } else {
    await initNoExcel();
  }
}

//Load Excel sheet 
async function loadSheet() {
  const sheet = document.getElementById("sheet-select").value;
  setLoader(2, true, "Preparing data and sending context to Gemini…");
  document.getElementById("btn-step2-action").disabled = true;

  try {
    const r = await fetch("/api/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sheet })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Error loading sheet");

    colRt     = d.col_rt;
    colSmiles = d.col_smiles;
    tableRows = d.rows;

    document.getElementById("th-rt").textContent     = colRt;
    document.getElementById("th-smiles").textContent = colSmiles;
    document.getElementById("p3-sub").textContent    =
      `${d.total20} compounds to predict (20% of dataset). ${d.total80} compounds used as context.`;

    const tbody = document.getElementById("table-body");
    tbody.innerHTML = tableRows.map((row, idx) =>
      `<tr onclick="selectRow(${idx})" id="tr-${idx}">
        <td>${row.row_id}</td>
        <td>${Number(row[colRt]).toFixed(3)}</td>
        <td class="smiles-cell">${row[colSmiles]}</td>
      </tr>`
    ).join("");

    showToast("Context sent. Gemini ready.");
    goStep(3);

  } catch(e) {
    showToast(e.message, true);
  } finally {
    setLoader(2, false);
    document.getElementById("btn-step2-action").disabled = false;
  }
}

// Initialization of Gemini without Excel
async function initNoExcel() {
  const smiles = document.getElementById("smiles-manual").value.trim();
  if (!smiles) { showToast("Please enter a SMILES first", true); return; }

  setLoader(2, true, "Initialising Gemini…");
  document.getElementById("btn-step2-action").disabled = true;

  try {
    const r = await fetch("/api/init_no_excel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: smiles })
    });
    
    // Verificar que la respuesta es JSON
    const contentType = r.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      const text = await r.text();
      throw new Error(`El servidor devolvió HTML en lugar de JSON: ${text.substring(0, 100)}...`);
    }
    
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Error initialising");

    showToast("Gemini ready.");

    // Fill step 4 fields from step 2 inputs
    document.getElementById("smiles-predict").value  = smiles;
    const rtVal = document.getElementById("rt-real-manual").value.trim();
    document.getElementById("rt-real-predict").value = rtVal;

    setupStep4NoExcel();
    goStep(4);

  } catch(e) {
    showToast(e.message, true);
    console.error("Error en initNoExcel:", e);
  } finally {
    setLoader(2, false);
    document.getElementById("btn-step2-action").disabled = false;
  }
}

// Row selection for Excel mode
function selectRow(idx) {
  if (selectedRow !== null) {
    document.getElementById("tr-" + selectedRow)?.classList.remove("selected");
  }
  selectedRow = idx;
  document.getElementById("tr-" + idx).classList.add("selected");
  document.getElementById("btn-to-step4").disabled = false;

  const row = tableRows[idx];
  document.getElementById("selected-preview").innerHTML =
    `<tr>
      <td>${row.row_id}</td>
      <td>${Number(row[colRt]).toFixed(3)}</td>
      <td class="smiles-cell">${row[colSmiles]}</td>
    </tr>`;

  setupStep4Excel();
}

// Setup Step 4 for each mode
function setupStep4Excel() {
  document.getElementById("p4-sub").textContent          = "Review the selected row and launch the prediction.";
  document.getElementById("p4-row-wrap").style.display   = "block";
  document.getElementById("p4-smiles-wrap").style.display= "none";
}

function setupStep4NoExcel() {
  document.getElementById("p4-sub").textContent          = "Review the SMILES (and real RT if available) and launch the prediction.";
  document.getElementById("p4-row-wrap").style.display   = "none";
  document.getElementById("p4-smiles-wrap").style.display= "block";
}

// Back from step 4
function backFromStep4() {
  if (mode === "excel" || mode === "excel+paper") {
    goStep(3);
  } else {
    goStep(2);
  }
}

// Predict
async function predict() {
  setLoader(4, true);
  document.getElementById("btn-predict").disabled = true;

  try {
    let d;

    if (mode === "excel" || mode === "excel+paper") {
      // ── Excel mode: ground truth always available ──
      if (selectedRow === null) { showToast("Select a row first", true); return; }
      const rowId = tableRows[selectedRow].row_id;
      const r = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row_id: rowId })
      });
      d = await r.json();
      if (!r.ok) throw new Error(d.error || "Prediction error");

      document.getElementById("r-real").textContent = d.rt_real.toFixed(3);
      document.getElementById("r-err").textContent  = d.abs_err.toFixed(3);
      document.getElementById("kpi-real").style.display = "";
      document.getElementById("kpi-err").style.display  = "";
      document.getElementById("result-grid").className  = "result-grid"; // 3-col

    } else {
      // ── No-Excel mode: RT real is optional ──
      const smiles = document.getElementById("smiles-predict").value.trim();
      if (!smiles) { showToast("Please enter a SMILES first", true); return; }

      const r = await fetch("/api/predict_smiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles })
      });
      d = await r.json();
      if (!r.ok) throw new Error(d.error || "Prediction error");

      // Check if user provided an RT real value
      const rtRealStr = document.getElementById("rt-real-predict").value.trim();
      const rtRealNum = parseFloat(rtRealStr);
      const hasRtReal = rtRealStr !== "" && !isNaN(rtRealNum);

      if (hasRtReal) {
        document.getElementById("r-real").textContent = rtRealNum.toFixed(3);
        document.getElementById("r-err").textContent  = Math.abs(d.rt_pred - rtRealNum).toFixed(3);
        document.getElementById("kpi-real").style.display = "";
        document.getElementById("kpi-err").style.display  = "";
        document.getElementById("result-grid").className  = "result-grid"; // 3-col
      } else {
        document.getElementById("kpi-real").style.display = "none";
        document.getElementById("kpi-err").style.display  = "none";
        document.getElementById("result-grid").className  = "result-grid two-col"; // 2-col
      }
    }

    document.getElementById("r-pred").textContent   = d.rt_pred.toFixed(3);
    document.getElementById("r-smiles").textContent = d.smiles;

    goStep(5);
    showToast("Prediction completed");

  } catch(e) {
    showToast(e.message, true);
  } finally {
    setLoader(4, false);
    document.getElementById("btn-predict").disabled = false;
  }
}

// Predict another
function predictAnother() {
  if (mode === "excel" || mode === "excel+paper") {
    goStep(3);
  } else {
    setupStep4NoExcel();
    goStep(4);
  }
}

// Restart
function restart() {
  excelReady = paperReady = false;
  selectedRow = null;
  tableRows = []; sheets = [];
  mode = "none";
  uploadsInProgress = 0;
  document.getElementById("btn-continue-1").disabled = false;
  document.getElementById("upload-status").textContent = "";
  document.getElementById("upload-status").className = "upload-status";
  document.getElementById("name-excel").textContent = "";
  document.getElementById("name-paper").textContent = "";
  document.getElementById("dz-excel").classList.remove("filled");
  document.getElementById("dz-paper").classList.remove("filled");
  document.getElementById("file-excel").value = "";
  document.getElementById("file-paper").value = "";
  document.getElementById("smiles-manual").value   = "";
  document.getElementById("rt-real-manual").value  = "";
  document.getElementById("smiles-predict").value  = "";
  document.getElementById("rt-real-predict").value = "";
  goStep(1);
}

// Loader util
function setLoader(step, show, msg) {
  const lw = document.getElementById("loader-" + step);
  if (!lw) return;
  lw.classList.toggle("show", show);
  if (msg) {
    const lm = document.getElementById("loader-" + step + "-msg");
    if (lm) lm.textContent = msg;
  }
}
</script>
</body>
</html>"""



if __name__ == "__main__":
    def open_browser():
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser).start()
    app.run(debug=False)