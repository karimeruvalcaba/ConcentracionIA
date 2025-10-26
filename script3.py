#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import difflib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# Prompt
# =========================

PROMPT_TEMPLATE = """Counterfactual Generation Prompt
You are an NLP assistant helping researchers generate
high-quality counterfactual examples for sentiment
classification.
Given a WhatsApp-style message and its sentiment (Positive or
Negative), generate 3 distinct versions that flip the sentiment.
Only modify necessary components. Preserve fluency and realism.
Respect informal tone.
You may flip sentiment by changing components such as:
- keywords, phrases, negation, intent framing, tone (e.g.,
sarcasm), sentiment valence, emojis/icons, code-mixing

Input:
Original message: "{original_message}"
Original sentiment: "{original_sentiment}"

Output Format (JSON List of 3 Objects):
[
{{
"cf_text": "...",
"components_changed": [...],
"flip_explanation": "..."
}},
...
]
"""

# =========================
# Mapeo de etiquetas
# =========================

def map_sentiment(value):
    """Map numeric labels to string sentiments:
       0 → Negative, 1 → Neutral, 2 → Positive."""
    v = str(value).strip()
    if v == "0":
        return "Negative"
    if v == "1":
        return "Neutral"
    if v == "2":
        return "Positive"
    return "Unknown"

# =========================
# Helpers de texto / listas
# =========================

def maybe_fix_mojibake(s: Optional[str]) -> Optional[str]:
    """
    Intenta reparar mojibake (UTF-8 leído como latin-1), p.ej. 'XÃ³chitl' -> 'Xóchitl'.
    Si no aplica, regresa el original.
    """
    if not isinstance(s, str):
        return s
    try:
        repaired = s.encode("latin-1").decode("utf-8")
        return repaired if repaired != s else s
    except Exception:
        return s

def normalize_list(x: Any) -> List[str]:
    """
    Asegura que 'components_changed' sea lista de strings.
    - Si viene como JSON string de lista, lo parsea.
    - Si es string plano, lo separa por comas.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        try:
            j = json.loads(x)
            if isinstance(j, list):
                return [str(v) for v in j]
        except Exception:
            pass
        return [v.strip() for v in x.split(",") if v.strip()]
    return [str(x)]

def infer_components_changed(original: str, cf_text: str, max_items: int = 10) -> List[str]:
    """
    Heurística simple de diff a nivel palabra: toma términos añadidos en cf_text.
    (Puedes mejorarla para considerar reemplazos/remociones si lo necesitas.)
    """
    o = original.split()
    c = cf_text.split()
    diff = list(difflib.ndiff(o, c))
    adds = []
    for tok in diff:
        if tok.startswith("+ "):
            word = tok[2:].strip(",.;:()[]{}!?¡¿\"'“”‘’")
            if word:
                adds.append(word)
    # Dedup preservando orden
    seen, out = set(), []
    for w in adds:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            out.append(w)
        if len(out) >= max_items:
            break
    return out

# =========================
# Parseo robusto de salida del modelo
# =========================

def extract_json_list(content: str):
    """
    Intenta parsear la lista JSON de counterfactuals y NORMALIZA cada CF
    para que SIEMPRE tenga:
      - cf_text (str)
      - components_changed (list[str])
      - flip_explanation (str)
    Si no puede parsear, cae en un regex para cf_text; último recurso: regresa todo como un CF.
    """
    s = content.strip()

    # Quitar fences ```json ... ```
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

    def _ensure(cf: Dict[str, Any]) -> Dict[str, Any]:
        cf_text = cf.get("cf_text", "")
        comps = normalize_list(cf.get("components_changed", []))
        flip = cf.get("flip_explanation") or ""
        return {"cf_text": cf_text, "components_changed": comps, "flip_explanation": flip}

    # Intento directo
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [_ensure(x if isinstance(x, dict) else {"cf_text": str(x)}) for x in parsed]
        if isinstance(parsed, dict):
            return [_ensure(parsed)]
    except Exception:
        pass

    # Buscar primer bloque [...]
    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list):
                return [_ensure(x if isinstance(x, dict) else {"cf_text": str(x)}) for x in parsed]
        except Exception:
            s = block

    # Fallback: extraer cf_text con regex
    cf_texts = re.findall(r'"cf_text"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
    if cf_texts:
        return [{"cf_text": bytes(ct, "utf-8").decode("unicode_escape"),
                 "components_changed": [],
                 "flip_explanation": ""} for ct in cf_texts]

    # Último recurso
    return [{"cf_text": content, "components_changed": [], "flip_explanation": ""}]

# =========================
# Aplanado y guardado
# =========================

def flatten_records(json_records: List[Dict[str, Any]], fix_mojibake: bool = True) -> pd.DataFrame:
    """
    Convierte la salida (lista de bloques con counterfactuals) en filas por CF.
    """
    rows = []
    for i, rec in enumerate(json_records):
        om = rec.get("original_message", "")
        osent = rec.get("original_sentiment", "")
        if fix_mojibake:
            om = maybe_fix_mojibake(om)
            osent = maybe_fix_mojibake(osent)

        cfs = rec.get("counterfactuals", []) or []
        if not isinstance(cfs, list):
            cfs = [cfs]

        for j, cf in enumerate(cfs, start=1):
            cf_text = cf.get("cf_text", "")
            flip_expl = cf.get("flip_explanation") or ""
            comps = normalize_list(cf.get("components_changed", []))

            if fix_mojibake:
                cf_text = maybe_fix_mojibake(cf_text)
                flip_expl = maybe_fix_mojibake(flip_expl)
                comps = [maybe_fix_mojibake(c) for c in comps]

            rows.append({
                "record_index": i,
                "cf_index": j,
                "original_message": om or "",
                "original_sentiment": osent or "",
                "cf_text": cf_text or "",
                "components_changed_list": comps,
                "components_changed_joined": " | ".join(comps),
                "components_changed_count": len(comps),
                "flip_explanation": flip_expl
            })
    return pd.DataFrame(rows)

def save_flat_outputs(df_flat: pd.DataFrame, output_dir: str, base_name: str = "counterfactual_results_flat"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / f"{base_name}.csv"
    df_flat.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # JSONL (manteniendo la lista real)
    jsonl_path = out_dir / f"{base_name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for _, row in df_flat.iterrows():
            rec = dict(row)
            rec["components_changed_list"] = list(rec.get("components_changed_list", []))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return str(csv_path), str(jsonl_path)

# =========================
# Flujo principal: genera CFs + guarda salidas
# =========================

def generate_counterfactuals(df: pd.DataFrame, client: OpenAI, output_dir: str, limit_rows: int = 3):
    # Limitar para pruebas rápidas; cambia limit_rows si quieres más
    df = df.head(limit_rows).copy()
    os.makedirs(output_dir, exist_ok=True)

    json_records: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        original_message = str(row["text"])
        original_sentiment = map_sentiment(row["Sentiment"])

        prompt = PROMPT_TEMPLATE.format(
            original_message=original_message,
            original_sentiment=original_sentiment
        )

        print(f"\n--- Row {idx+1} ---")
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert linguistic assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:
            print("OpenAI error:", e)
            content = ""

        # Parsear salida y normalizar
        cf_list = extract_json_list(content)

        # Arreglar mojibake y completar campos faltantes por inferencia
        fixed_cf_list = []
        for cf in cf_list:
            cf_text = maybe_fix_mojibake(cf.get("cf_text", ""))
            comps = normalize_list(cf.get("components_changed", []))
            flip = cf.get("flip_explanation") or ""

            if not comps:
                comps = infer_components_changed(original_message, cf_text)

            if not flip:
                flip = "Auto-inferred changes from word-level diff." if comps else "No explanation provided."

            fixed_cf_list.append({
                "cf_text": cf_text,
                "components_changed": comps,
                "flip_explanation": flip
            })

        # Guardar bloque JSON (para referencia completa)
        json_records.append({
            "original_message": maybe_fix_mojibake(original_message),
            "original_sentiment": original_sentiment,
            "counterfactuals": fixed_cf_list
        })

        # Columnas cf_text_1..3 (ya reparadas)
        cf_texts = [c["cf_text"] for c in fixed_cf_list]
        cf_texts = (cf_texts + ["", "", ""])[:3]

        table_rows.append({
            "original_message": maybe_fix_mojibake(original_message),
            "original_sentiment": original_sentiment,
            "cf_text_1": cf_texts[0],
            "cf_text_2": cf_texts[1],
            "cf_text_3": cf_texts[2],
        })

        time.sleep(0.8)

    # Salidas originales
    json_path = os.path.join(output_dir, "counterfactual_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "counterfactual_results.csv")
    pd.DataFrame(table_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Aplanado (1 fila por counterfactual)
    df_flat = flatten_records(json_records, fix_mojibake=True)
    flat_csv, flat_jsonl = save_flat_outputs(df_flat, output_dir, base_name="counterfactual_results_flat")

    print(f"\n✅ Saved:\n- {json_path}\n- {csv_path}\n- {flat_csv}\n- {flat_jsonl}")

# =========================
# Carga de dataset
# =========================

def load_dataframe(src_path: str) -> pd.DataFrame:
    """Load .xlsx (preferred) or .csv. Expects columns: text, Sentiment."""
    _, ext = os.path.splitext(src_path.lower())
    if ext == ".xlsx":
        # Si ves error, instala: pip install openpyxl
        df = pd.read_excel(src_path, engine="openpyxl")
    else:
        # CSV: usa latin1 para evitar errores por acentos; ajusta si tu CSV es UTF-8
        df = pd.read_csv(src_path, encoding="latin1")
    # Sanity check
    missing = {"text", "Sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df

# =========================
# Main
# =========================

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no está definido en el entorno/.env")
    client = OpenAI(api_key=api_key)

    # === Ajusta tus rutas aquí ===
    base_path = r"C:\Users\abdel\Downloads\concentracionIA"
    input_file = "gold_anot.xlsx"            # <-- Excel de entrada con columnas: text, Sentiment
    output_dir = os.path.join(base_path, "results_llms3")
    src_path = os.path.join(base_path, input_file)

    # Opcional: cambia el límite de filas para pruebas rápidas
    limit_rows = int(os.getenv("CF_LIMIT_ROWS", "3"))

    df = load_dataframe(src_path)
    generate_counterfactuals(df, client, output_dir, limit_rows=limit_rows)

if __name__ == "__main__":
    main()
