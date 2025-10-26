# -*- coding: utf-8 -*-
import os
import json
import textwrap
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# ========================
# Config / Paths (edit here)
# ========================
BASE_PATH   = Path.home() / "Downloads" / "concentracionIA"
INPUT_FILE  = "gold_anot.xlsx"  # e.g. "sample-labeled-sentiment-anot1(Sheet1).csv" or "gold_anot.xlsx"
SHEET_NAME  = 0                 # Excel only: 0 (first sheet) or a string like "Sheet1"
OUTPUT_DIR  = BASE_PATH / "results_llms"

# Cost controls
N_SAMPLE    = 6
MAX_CALLS   = 50

# Expected CSV/Excel columns
EXPECTED_COLS = {"platform", "text", "Sentiment"}

# ========================
# Env & OpenAI client
# ========================
def init_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no está definido en el entorno/.env")
    return OpenAI(api_key=api_key)

# ========================
# Data loading
# ========================
def load_dataset_robust(path: Path, sheet: int | str | None = 0) -> pd.DataFrame:
    """
    Reads CSV/TSV/Excel/Parquet with sensible fallbacks.
      - CSV/TSV: try utf-8, then latin1, then cp1252; fallback with errors='replace'.
      - Excel: by default loads the first sheet (sheet=0). If dict is returned, pick first.
      - Parquet: straightforward.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {path}")

    ext = path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        encodings = ["utf-8", "latin1", "cp1252"]
        last_err = None
        for enc in encodings:
            try:
                return pd.read_csv(path, sep=sep, encoding=enc)
            except Exception as e:
                last_err = e
        try:
            return pd.read_csv(path, sep=sep, encoding="utf-8", errors="replace")
        except Exception:
            raise last_err

    if ext in [".xls", ".xlsx"]:
        df_or_dict = pd.read_excel(path, sheet_name=sheet)
        if isinstance(df_or_dict, dict):
            first_key = next(iter(df_or_dict))
            return df_or_dict[first_key]
        return df_or_dict

    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)

    # Default: intentar CSV
    return pd.read_csv(path)

# ========================
# Prompting / Inference
# ========================
def build_prompt(query: str) -> str:
    return textwrap.dedent(f"""
        You are an NLP assistant for sentiment analysis.
        Task: Given a WhatsApp-style message (QUERY), classify its sentiment as "Positive", "Negative", or "Neutral".
        Provide a justification using extracted keywords and a brief explanation.
        Return a confidence score from 0 to 5.

        Output requirements:
        - Return ONLY valid JSON (no extra text).
        - JSON schema:
          {{
            "justification": {{
              "keywords": ["...","..."],
              "explanation": "..."
            }},
            "sentiment": "Positive|Negative|Neutral",
            "confidence_score": 0-5
          }}

        QUERY: "{query}"
    """).strip()

API_CALLS_USED = 0
def check_quota():
    global API_CALLS_USED
    if API_CALLS_USED >= MAX_CALLS:
        raise RuntimeError(f"Se alcanzó el límite de {MAX_CALLS} llamadas a la API.")
    API_CALLS_USED += 1

def classify_sentiment_llm(client: OpenAI, text: str) -> dict:
    """Call the model with strict JSON response."""
    check_quota()
    prompt = build_prompt(text)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful NLP assistant. "
                    "Return ONLY valid JSON per the schema. Do not include extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Modelo devolvió JSON inválido: {raw[:200]}...")
    return data

# ========================
# Runner
# ========================
def run_sentiment_demo(client: OpenAI, data_path: Path, output_dir: Path, n_rows: int = N_SAMPLE, sheet=0) -> None:
    # Ensure output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_dataset_robust(data_path, sheet=sheet)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"El archivo no tiene columnas esperadas: falta(n) {missing}")

    sample_df = df.head(n_rows).copy().reset_index(drop=True)

    results = []
    aug_rows = []

    for i, row in sample_df.iterrows():
        text = str(row["text"]).strip()
        try:
            out = classify_sentiment_llm(client, text)
            just = out.get("justification", {}) or {}
            keywords = just.get("keywords", []) or []
            explanation = just.get("explanation", "") or ""
            sentiment_pred = out.get("sentiment", "") or ""
            conf = out.get("confidence_score", "")

            # Normalize confidence to number
            try:
                conf_num = float(conf)
            except Exception:
                conf_num = None

            results.append(out)
            aug_rows.append(
                {
                    "platform": row["platform"],
                    "text": text,
                    "label_original": row["Sentiment"],
                    "pred_sentiment": sentiment_pred,
                    "pred_confidence": conf_num,
                    "pred_keywords": ", ".join(map(str, keywords)),
                    "pred_explanation": explanation,
                }
            )
            print(f"[{i+1}/{len(sample_df)}] ✓ Classified")
        except Exception as e:
            print(f"[{i+1}/{len(sample_df)}] Error: {e}")
            aug_rows.append(
                {
                    "platform": row.get("platform", ""),
                    "text": text,
                    "label_original": row.get("Sentiment", ""),
                    "pred_sentiment": "",
                    "pred_confidence": "",
                    "pred_keywords": "",
                    "pred_explanation": f"Error: {e}",
                }
            )

    # Save outputs
    jsonl_path = output_dir / "sentiment_llm_results.jsonl"
    csv_path   = output_dir / "sentiment_llm_results.csv"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pd.DataFrame(aug_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {jsonl_path}")
    print(f"Saved: {csv_path}")

# ========================
# Main
# ========================
def main():
    client = init_client()
    data_path = BASE_PATH / INPUT_FILE
    run_sentiment_demo(
        client,
        data_path=data_path,
        output_dir=OUTPUT_DIR,
        n_rows=N_SAMPLE,
        sheet=SHEET_NAME,   # Only used for Excel; ignored for CSV
    )

if __name__ == "__main__":
    main()
