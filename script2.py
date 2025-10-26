import os, json, re
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIG & COST GUARD
# =========================
N_SAMPLE = 6
MAX_CALLS = 100   # total budget for both stages
API_CALLS_USED = 0

def check_quota(n: int = 1):
    """Consume n API calls from the global budget."""
    global API_CALLS_USED
    if API_CALLS_USED + n > MAX_CALLS:
        raise RuntimeError(f"Se alcanzó el límite de {MAX_CALLS} llamadas a la API.")
    API_CALLS_USED += n

# =========================
# DATA LOADING (robust, only .xlsx)
# =========================
EXPECTED_COLS = {"platform", "text", "Sentiment"}

def detect_label_scheme(series: pd.Series):
    """
    Fixed label scheme:
    0 → Negative
    1 → Neutral
    2 → Positive
    """
    return (
        "neg0",
        {0: "Negative", 1: "Neutral", 2: "Positive"},
        {"negative": 0, "neutral": 1, "positive": 2},
    )

def load_dataset_robust(path: str) -> pd.DataFrame:
    """
    Loads only Excel (.xlsx) files, raising an error otherwise.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {p}")

    if p.suffix.lower() != ".xlsx":
        raise ValueError(f"Solo se admiten archivos .xlsx, no {p.suffix}")

    try:
        return pd.read_excel(p)
    except ImportError as e:
        raise RuntimeError(
            "Falta el motor de Excel. Instala openpyxl: pip install openpyxl"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Error al leer el archivo Excel: {e}")

def target_flip_label(src_label_str: str) -> str:
    """
    Flip Positive <-> Negative, and move Neutral -> Positive (deterministic).
    """
    lk = (src_label_str or "").strip().lower()
    if lk == "positive":
        return "Negative"
    if lk == "negative":
        return "Positive"
    # If neutral, choose Positive as the flipped target
    return "Positive"

# =========================
# PROMPTS (GENERATION & EVALUATION)
# =========================
def build_generation_prompt(original: str, src_label: str, target_label: str) -> str:
    """
    Instructs the model to rewrite the message to the target sentiment,
    preserving core meaning and context (minimal edits), Spanish allowed.
    """
    return f"""You are rewriting a WhatsApp message to flip its sentiment while preserving core meaning and context.

Original message:
\"\"\"{original}\"\"\"

Original sentiment: {src_label}
Target sentiment: {target_label}

Guidelines:
- Keep the message plausible and human-like for WhatsApp.
- Preserve entities, facts, and context where possible.
- Make minimal edits necessary to achieve the target sentiment.
- Keep the same language as the original.
- Do not add URLs or metadata that wasn't there.
- Avoid sarcasm unless it is essential for the flip.

Return ONLY valid JSON with this schema:
{{
  "flipped_text": "the rewritten message",
  "explanation": "brief note explaining how sentiment was flipped (1-2 lines)"
}}"""

def build_counterfactual_prompt(original: str, flipped: str, transformation: str, explanation: str) -> str:
    return f"""Counterfactual Evaluation Prompt
You are evaluating a synthetic (GPT-4-generated) version of a WhatsApp message.
The synthetic message is a sentiment-flipped version of the original.

Assess the quality of the synthetic message along four criteria using 0 or 1:
1. Fluency — Is the synthetic message grammatically correct and readable?
2. Naturalness — Does it sound plausible for a human to write?
3. Sentiment Flip Clarity — Is the sentiment clearly flipped from the original?
4. Meaning Preservation — Is the core meaning preserved aside from the sentiment?

Original Message: "{original}"
Synthetic Message: "{flipped}"
Transformation Type: {transformation}
GPT-4 Explanation for the Flip: "{explanation}"

Return ONLY this JSON:
{{
"fluency": 0 or 1,
"naturalness": 0 or 1,
"sentiment_flip_clarity": 0 or 1,
"meaning_preservation": 0 or 1,
"annotator_comment": "optional comment (string)"
}}"""

# =========================
# OPENAI CALLS (robust JSON parse)
# =========================
def _strip_code_fences(s: str) -> str:
    # Remove ```json ... ``` or ``` ... ```
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)

def call_json(client: OpenAI, system: str, user: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> dict:
    check_quota(1)
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    raw = resp.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try again stripping code fences if present
        raw2 = _strip_code_fences(raw)
        return json.loads(raw2)

def generate_flip(client: OpenAI, original: str, src_label: str, target_label: str) -> tuple[str, str]:
    system = (
        "You are a careful rewriting assistant. "
        "Return ONLY valid JSON as specified; no extra preamble or commentary."
    )
    prompt = build_generation_prompt(original, src_label, target_label)
    out = call_json(client, system, prompt)
    flipped_text = (out.get("flipped_text") or "").strip()
    explanation = (out.get("explanation") or "").strip()
    return flipped_text, explanation

def evaluate_counterfactual(client: OpenAI, original: str, flipped: str, transformation: str, explanation: str) -> dict:
    system = (
        "You are an NLP evaluator. Respond ONLY with valid JSON "
        "according to the given schema; no text before or after."
    )
    prompt = build_counterfactual_prompt(original, flipped, transformation, explanation)
    return call_json(client, system, prompt)

# =========================
# PIPELINE STEPS (save/read .xlsx)
# =========================
def stage_generate_counterfactuals(client: OpenAI, src_xlsx: str, out_dir: str, n_rows: int = N_SAMPLE) -> str:
    """
    Reads original XLSX (platform, text, Sentiment),
    generates flipped versions + explanations,
    saves counterfactual_pairs.xlsx and returns its path.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = load_dataset_robust(src_xlsx)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"El XLSX no tiene columnas esperadas: {missing}")

    # detect scheme and map to strings for control
    scheme, num2str, _ = detect_label_scheme(df["Sentiment"])
    df = df.head(n_rows).copy().reset_index(drop=True)
    # Ensure ints map correctly; coerce to Int64 in case of NaNs
    df["Sentiment"] = pd.to_numeric(df["Sentiment"], errors="coerce").astype("Int64")
    df["sent_str"] = df["Sentiment"].map(num2str)

    pairs = []
    for i, row in df.iterrows():
        original = str(row["text"]).strip()
        src_label = str(row["sent_str"])
        tgt_label = target_flip_label(src_label)
        try:
            flipped_text, explanation = generate_flip(client, original, src_label, tgt_label)
        except Exception as e:
            flipped_text, explanation = "", f"Error generating flip: {e}"

        pairs.append(
            {
                "original": original,
                "flipped": flipped_text,
                "transformation": "sentiment_flip",
                "explanation": explanation,
                "target_sentiment": tgt_label,
                "platform": row["platform"],
                "original_label_numeric": row["Sentiment"],
                "original_label_str": src_label,
            }
        )
        print(f"[Flip {i+1}/{len(df)}] ✓")

    pairs_df = pd.DataFrame(pairs)
    out_xlsx = os.path.join(out_dir, "counterfactual_pairs.xlsx")
    # Write as .xlsx so Stage 2 can read it with the strict loader
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        pairs_df.to_excel(w, index=False)
    print(f"Saved flips XLSX: {out_xlsx}")
    return out_xlsx

def stage_evaluate_counterfactuals(client: OpenAI, pairs_xlsx: str, out_dir: str) -> None:
    """
    Reads counterfactual_pairs.xlsx with columns:
    original, flipped, transformation, explanation
    Runs counterfactual evaluation and saves JSONL + XLSX.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = load_dataset_robust(pairs_xlsx)
    expected = {"original", "flipped", "transformation", "explanation"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas esperadas en pairs XLSX: {missing}")

    results = []
    aug_rows = []
    for i, row in df.iterrows():
        original = str(row["original"])
        flipped = str(row["flipped"])
        transformation = str(row["transformation"])
        explanation = str(row["explanation"])

        try:
            res = evaluate_counterfactual(client, original, flipped, transformation, explanation)
            results.append(res)
            aug_rows.append({**row.to_dict(), **res})
            print(f"[Eval {i+1}/{len(df)}] ✓")
        except Exception as e:
            aug_rows.append({**row.to_dict(), "error": str(e)})
            print(f"[Eval {i+1}/{len(df)}] Error: {e}")

    # Save raw JSONL
    jsonl_path = os.path.join(out_dir, "counterfactual_eval_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save augmented rows as XLSX
    xlsx_path = os.path.join(out_dir, "counterfactual_eval_results.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        pd.DataFrame(aug_rows).to_excel(w, index=False)

    print(f"Saved eval JSONL: {jsonl_path}")
    print(f"Saved eval XLSX:  {xlsx_path}")

# =========================
# MAIN
# =========================
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no está definido en el entorno/.env")
    client = OpenAI(api_key=api_key)

    base_path = r"C:\Users\abdel\Downloads\concentracionIA"
    input_file = "gold_anot.xlsx"  # <- Debe ser .xlsx
    output_dir = os.path.join(base_path, "results_llms2")
    src_xlsx = os.path.join(base_path, input_file)

    # Stage 1: generate flips XLSX from original
    pairs_xlsx = stage_generate_counterfactuals(client, src_xlsx, output_dir, n_rows=N_SAMPLE)

    # Stage 2: evaluate flips with Counterfactual Evaluation Prompt
    stage_evaluate_counterfactuals(client, pairs_xlsx, output_dir)

if __name__ == "__main__":
    main()
