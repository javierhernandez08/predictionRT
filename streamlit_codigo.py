import os
import re
import time
import tempfile
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai


# =========================
# 🔴 HARDCODE API KEY (SOLO PRUEBAS)
# =========================
API_KEY = "AIzaSyCtK0oP1PVulsZUHAacYitnFfj5X0gqAq0"


# =========================
# CONFIG FIJA (NO MODIFICABLE POR USUARIO)
# =========================
MODEL_NAME = "gemini-2.5-flash"
SEED = 42
CONTEXT_FRAC = 0.20  # 20% contexto, 80% tabla para elegir
MAX_CONTEXT_ROWS = 80
MAX_CONTEXT_SMILES_LEN = 160
MAX_QUERY_SMILES_LEN = 500
MAX_TRIES = 10
SHOW_80_ROWS = 300


# =========================
# EXCEL UTILS
# =========================
def buscar_columna(df: pd.DataFrame, posibles: List[str]) -> Optional[str]:
    cols = {c.lower().strip(): c for c in df.columns}
    for p in posibles:
        key = p.lower().strip()
        if key in cols:
            return cols[key]
    # fallback: contains
    for c in df.columns:
        cl = c.lower()
        for p in posibles:
            if p.lower() in cl:
                return c
    return None


def cargar_excel_y_hojas(file_bytes: bytes) -> List[str]:
    xls = pd.ExcelFile(file_bytes)
    return xls.sheet_names


def leer_hoja_excel(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(file_bytes, sheet_name=sheet_name)


def preparar_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    col_rt = buscar_columna(df, ["rt", "rt_min", "retention_time"])
    col_smiles = buscar_columna(df, ["pubchem.smiles.canonical", "smiles"])

    if col_rt is None or col_smiles is None:
        raise RuntimeError("No se encontraron columnas RT o SMILES en esa hoja.")

    df = df.copy()
    df[col_rt] = pd.to_numeric(df[col_rt], errors="coerce")
    df[col_smiles] = df[col_smiles].astype(str)

    df = df.dropna(subset=[col_rt, col_smiles])
    df = df[df[col_smiles].str.strip().ne("")]
    df = df.reset_index(drop=True)
    return df, col_rt, col_smiles


def split_contexto_20_tabla_80(
    df: pd.DataFrame,
    frac_context: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_shuf = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_ctx = int(len(df_shuf) * frac_context)
    ctx = df_shuf.iloc[:n_ctx].reset_index(drop=True)
    rest = df_shuf.iloc[n_ctx:].reset_index(drop=True)
    return ctx, rest


def construir_tabla_contexto(
    ctx_df: pd.DataFrame,
    col_rt: str,
    col_smiles: str,
    max_rows: int,
    max_smiles_len: int
) -> str:
    if max_rows <= 0:
        return ""
    lines = []
    for _, r in ctx_df.head(max_rows).iterrows():
        rt = float(r[col_rt])
        smi = str(r[col_smiles]).strip()
        if len(smi) > max_smiles_len:
            smi = smi[:max_smiles_len]
        lines.append(f"{rt:.3f}\t{smi}")
    return "\n".join(lines)


# =========================
# GEMINI SAFE RESPONSE READING
# =========================
def leer_texto_respuesta_seguro(resp: Any) -> str:
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            c0 = resp.candidates[0]
            content = getattr(c0, "content", None)
            if content is None:
                return ""
            parts = getattr(content, "parts", None) or []
            textos = []
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t.strip():
                    textos.append(t.strip())
            return "\n".join(textos).strip()
    except Exception:
        return ""
    return ""


def finish_reason(resp: Any):
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            return getattr(resp.candidates[0], "finish_reason", None)
    except Exception:
        pass
    return None


def parse_retry_seconds_from_error(err: Exception) -> float:
    s = str(err)
    m = re.search(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", s)
    if m:
        return float(m.group(1))
    m2 = re.search(r"retry_delay\s*{\s*seconds:\s*([0-9]+)", s)
    if m2:
        return float(m2.group(1))
    return 4.0


def extraer_primer_numero(texto: str) -> float:
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", texto or "")
    if not m:
        raise ValueError(f"No se pudo extraer número de: {texto!r}")
    return float(m.group(1))


# =========================
# PROMPTS (2 PROMPTS)
# =========================
def prompt_1_contexto(train_table: str) -> str:
    return (
        "You predict chromatography retention time (RT) in minutes.\n"
        "You are given example pairs (rt_minutes\\tsmiles) from the SAME chromatographic system.\n"
        "Infer the time scale and patterns from these examples.\n\n"
        "OUTPUT RULE (for later predictions):\n"
        "- Reply ONLY one number with exactly 3 decimals.\n"
        "- No words. No units. No punctuation. No explanations.\n"
        "- If uncertain, still output your best estimate.\n\n"
        "EXAMPLES (rt_minutes\\tsmiles):\n"
        f"{train_table}\n\n"
        "Reply now with: 0.000"
    )


def prompt_2_prediccion(smiles: str, max_query_smiles_len: int) -> str:
    smi = (smiles or "").strip()
    if len(smi) > max_query_smiles_len:
        smi = smi[:max_query_smiles_len]
    return (
        "Return ONLY one number with exactly 3 decimals.\n"
        "No words. No units. No punctuation.\n"
        f"SMILES: {smi}"
    )


# =========================
# GEMINI SETUP / PAPER UPLOAD
# =========================
def configurar_gemini(model_name: str) -> genai.GenerativeModel:
    if not API_KEY or "PEGA_AQUI" in API_KEY:
        raise RuntimeError("Pon tu API key en API_KEY (hardcode) para probar.")
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(model_name)


def guardar_pdf_temp(paper_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(paper_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def subir_paper_a_gemini(pdf_path: str):
    return genai.upload_file(path=pdf_path, mime_type="application/pdf")


# =========================
# SEND WITH RETRIES
# =========================
def enviar_con_reintentos(chat, mensaje, max_tries: int, log_fn=None) -> str:
    last_err = None
    for attempt in range(max_tries):
        try:
            resp = chat.send_message(
                mensaje,
                generation_config={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_output_tokens": 128,
                },
            )

            texto = leer_texto_respuesta_seguro(resp)

            if not texto:
                fr = finish_reason(resp)
                wait_s = min(10.0, 1.0 + attempt * 0.9)
                if log_fn:
                    log_fn(f"[EMPTY] finish_reason={fr} -> retry in {wait_s:.1f}s ({attempt+1}/{max_tries})")
                time.sleep(wait_s)
                continue

            return texto.strip()

        except Exception as e:
            last_err = e
            msg = str(e).lower()

            if "429" in msg or "resourceexhausted" in msg or "quota" in msg:
                wait_s = parse_retry_seconds_from_error(e)
                wait_s = max(1.0, wait_s)
                if log_fn:
                    log_fn(f"[RATE] wait {wait_s:.1f}s ({attempt+1}/{max_tries})")
                time.sleep(wait_s)
                continue

            wait_s = min(10.0, 1.0 + attempt * 0.9)
            if log_fn:
                log_fn(f"[ERR] {e} -> retry in {wait_s:.1f}s ({attempt+1}/{max_tries})")
            time.sleep(wait_s)
            continue

    raise last_err


# =========================
# STREAMLIT UI (SIN CONFIGURACIÓN)
# =========================
st.set_page_config(page_title="RT Predictor", layout="wide")
st.title("RT Predictor — Predicción de Retention Time")

# Carga de archivos
st.subheader("1) Sube el Excel y el paper (opcional)")
excel_file = st.file_uploader("Excel (.xlsx)", type=["xlsx"])
paper_file = st.file_uploader("Paper PDF (opcional)", type=["pdf"])

log_box = st.empty()
logs = []


def log(msg: str):
    logs.append(msg)
    joined = "\n".join(logs)
    if len(joined) > 9000:
        joined = "...\n" + joined[-9000:]
    log_box.code(joined)


# Session state
if "xls_bytes" not in st.session_state:
    st.session_state.xls_bytes = None
if "sheet_names" not in st.session_state:
    st.session_state.sheet_names = None
if "paper_tmp_path" not in st.session_state:
    st.session_state.paper_tmp_path = None
if "chat" not in st.session_state:
    st.session_state.chat = None
if "context_sent" not in st.session_state:
    st.session_state.context_sent = False
if "df80" not in st.session_state:
    st.session_state.df80 = None
if "col_rt" not in st.session_state:
    st.session_state.col_rt = None
if "col_smiles" not in st.session_state:
    st.session_state.col_smiles = None


# Handle uploads
if excel_file is not None:
    st.session_state.xls_bytes = excel_file.getvalue()
    try:
        st.session_state.sheet_names = cargar_excel_y_hojas(st.session_state.xls_bytes)
        log(f"✅ Excel cargado. Hojas: {st.session_state.sheet_names}")
    except Exception as e:
        st.error(str(e))

if paper_file is not None:
    try:
        st.session_state.paper_tmp_path = guardar_pdf_temp(paper_file.getvalue())
        log(f"📄 Paper recibido: {paper_file.name}")
    except Exception as e:
        st.error(str(e))


# Step 2: choose sheet
if st.session_state.sheet_names:
    st.subheader("2) Elige la hoja (página) del Excel")
    sheet = st.selectbox("Hoja:", options=st.session_state.sheet_names, index=0)

    load_sheet_btn = st.button("📥 Cargar hoja y preparar 20%/80%")

    if load_sheet_btn:
        try:
            raw_df = leer_hoja_excel(st.session_state.xls_bytes, sheet)
            df, col_rt, col_smiles = preparar_df(raw_df)

            ctx20, rest80 = split_contexto_20_tabla_80(df, CONTEXT_FRAC, SEED)

            st.session_state.df80 = rest80
            st.session_state.col_rt = col_rt
            st.session_state.col_smiles = col_smiles

            # Init Gemini chat and send PROMPT 1 (20%)
            train_table = construir_tabla_contexto(
                ctx20, col_rt, col_smiles, MAX_CONTEXT_ROWS, MAX_CONTEXT_SMILES_LEN
            )

            log(f"✅ Hoja '{sheet}' preparada: total={len(df)} | 20%={len(ctx20)} | 80%={len(rest80)}")
            log("Inicializando Gemini chat...")

            model = configurar_gemini(MODEL_NAME)
            chat = model.start_chat(history=[])

            p1 = prompt_1_contexto(train_table)

            if st.session_state.paper_tmp_path:
                paper_handle = subir_paper_a_gemini(st.session_state.paper_tmp_path)
                init_msg = [paper_handle, p1]
                log("Adjuntando paper y enviando PROMPT 1...")
            else:
                init_msg = p1
                log("Enviando PROMPT 1 (sin paper)...")

            init_resp = enviar_con_reintentos(chat, init_msg, MAX_TRIES, log_fn=log)
            log(f"PROMPT 1 OK. Resp: {init_resp[:80].replace(chr(10),' ')}")

            st.session_state.chat = chat
            st.session_state.context_sent = True

            st.success("PROMPT 1 enviado. Ahora elige 1 fila del 80% para predecir.")

        except Exception as e:
            st.session_state.context_sent = False
            st.session_state.chat = None
            st.error(str(e))
            log(f"❌ ERROR: {e}")


# Step 3: show 80% and pick one row
if st.session_state.df80 is not None:
    col_rt = st.session_state.col_rt
    col_smiles = st.session_state.col_smiles
    df80 = st.session_state.df80

    st.subheader("3) 80% restante — elige 1 fila para predecir")
    df80_view = df80[[col_rt, col_smiles]].copy()
    df80_view["row_id"] = df80_view.index
    df80_view[col_smiles] = df80_view[col_smiles].astype(str).str.slice(0, 160)

    st.dataframe(
        df80_view[["row_id", col_rt, col_smiles]].head(SHOW_80_ROWS),
        use_container_width=True,
        height=360,
    )

    row_id = st.number_input(
        "row_id a predecir (del 80%):",
        min_value=0,
        max_value=max(0, len(df80_view) - 1),
        value=0,
        step=1,
    )

    predict_btn = st.button("🎯 Predecir", disabled=(not st.session_state.context_sent))

    if not st.session_state.context_sent:
        st.info("Primero carga hoja y envía PROMPT 1 (se hace al pulsar 'Cargar hoja y preparar 20%/80%').")

    if predict_btn:
        try:
            chat = st.session_state.chat
            if chat is None:
                raise RuntimeError("No hay chat activo. Repite el paso de cargar hoja.")

            smi = str(df80.loc[int(row_id), col_smiles]).strip()
            rt_real = float(df80.loc[int(row_id), col_rt])

            p2 = prompt_2_prediccion(smi, MAX_QUERY_SMILES_LEN)
            log(f"Enviando PROMPT 2 (row_id={int(row_id)})...")
            raw = enviar_con_reintentos(chat, p2, MAX_TRIES, log_fn=log)
            rt_pred = extraer_primer_numero(raw)

            abs_err = abs(rt_pred - rt_real)

            st.subheader("Resultado")
            res = pd.DataFrame([{
                "sheet": sheet if 'sheet' in locals() else "",
                "row_id_80": int(row_id),
                "rt_real": rt_real,
                "rt_pred": rt_pred,
                "abs_err": abs_err,
                "smiles": smi
            }])
            st.dataframe(res, use_container_width=True)

            log(f"✅ Resultado: real={rt_real:.3f} pred={rt_pred:.3f} abs_err={abs_err:.3f} | raw={raw[:60]!r}")

        except Exception as e:
            st.error(str(e))
            log(f"❌ ERROR PRED: {e}")


st.caption("Flujo: eliges hoja → se manda automáticamente 20% como contexto → eliges 1 del 80% para predecir.")