import os
import re
import json
import time
import difflib
import textwrap, sys, math, argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# CONFIG & PATHS (EDITABLES)
# =========================================================
BASE_PATH   = Path(os.getenv("BASE_PATH", str(Path.home() / "concentracionIA")))
INPUT_FILE  = os.getenv("INPUT_FILE", "gold_anot.xlsx")
SHEET_NAME  = 0  # para Excel: 0 ó nombre de hoja

OUTPUT_DIR1 = BASE_PATH / "results_llms"      # Script 1: clasificación
OUTPUT_DIR2 = BASE_PATH / "results_llms2"     # Script 2: flips + evaluación
OUTPUT_DIR3 = BASE_PATH / "results_llms3"     # Script 3: cf triples + aplanado
OUTPUT_DIR4 = BASE_PATH / "results_llms4"     # Script 4: filtro (elige mejor CF)
OUTPUT_DIR5 = BASE_PATH / "results_llms5"     # Script 5: evaluación de explicaciones

# Cost guard compartido
N_SAMPLE       = int(os.getenv("N_SAMPLE", "6"))           # usado por 1 y 2
CF_LIMIT_ROWS  = N_SAMPLE      # usado por 3
MAX_CALLS      = int(os.getenv("MAX_CALLS", "200"))
API_CALLS_USED = 0

EXPECTED_COLS = {"platform", "text", "Sentiment"}

USE_INLINE_PROGRESS = True  # set False if you prefer new lines instead of in-place updates

def _pct(i: int, total: int) -> int:
    # i is 1-based progress
    if total <= 0: return 100
    return min(100, max(0, math.floor(i * 100 / total)))

def show_progress(stage: str, i: int, total: int):
    percent = _pct(i, total)
    msg = f"{stage}: {percent}%  ({i}/{total})"
    if USE_INLINE_PROGRESS:
        # overwrite same line
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
        if i == total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        print(msg)

def apply_n_override(n: str | int | None, data_path: Path | None = None):
    """
    If --n is provided:
      - If it's "full" or -1 → use ALL rows in the dataset (len(file))
      - Else set N_SAMPLE and CF_LIMIT_ROWS = n
    """
    global N_SAMPLE, CF_LIMIT_ROWS

    if n is None:
        return

    # handle 'full' mode
    if isinstance(n, str) and n.lower() == "full":
        if data_path and data_path.exists():
            try:
                # quick load just to get number of rows
                import pandas as pd
                ext = data_path.suffix.lower()
                if ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(data_path, nrows=None)
                else:
                    df = pd.read_csv(data_path, encoding="utf-8", errors="ignore")
                total_rows = len(df)
                N_SAMPLE = total_rows
                CF_LIMIT_ROWS = total_rows
                print(f"🔢 Using full dataset: {total_rows} rows.")
                return
            except Exception as e:
                print(f"⚠️ Could not count rows automatically ({e}), defaulting to all available.")
                N_SAMPLE = 999999
                CF_LIMIT_ROWS = 999999
                return
        else:
            N_SAMPLE = 999999
            CF_LIMIT_ROWS = 999999
            print("⚠️ File not found for row count; will attempt all rows.")
            return

    # numeric n
    try:
        n = int(n)
        if n < 0:
            # treat -1 as full
            return apply_n_override("full", data_path)
        n = max(1, n)
        N_SAMPLE = n
        CF_LIMIT_ROWS = n
        print(f"Overriding N_SAMPLE and CF_LIMIT_ROWS to {n}.")
    except Exception:
        print(f"Invalid --n value: {n}. Ignored.")

# ---- Balanced sampling across Positive/Negative/Neutral (for scripts 1–3) ----

def _normalize_sent_str(v: Any) -> str:
    """
    Map various Sentiment encodings to canonical strings:
      - 2 or '2' or 'positive'/'pos' -> 'Positive'
      - 1 or '1' or 'neutral'/'neu'  -> 'Neutral'
      - 0 or '0' or 'negative'/'neg' -> 'Negative'
    Anything else -> '' (ignored)
    """
    if v is None:
        return ""
    s = str(v).strip().lower()
    if s in {"2", "positive", "pos", "positiva", "positivo", "positif"}:
        return "Positive"
    if s in {"1", "neutral", "neu", "neutro"}:
        return "Neutral"
    if s in {"0", "negative", "neg", "negativo"}:
        return "Negative"
    # try numeric cast fallback
    try:
        n = int(float(s))
        return {2: "Positive", 1: "Neutral", 0: "Negative"}.get(n, "")
    except Exception:
        return ""

def balanced_subset(df: pd.DataFrame, per_class: int, label_col: str = "Sentiment", seed: int = 42) -> pd.DataFrame:
    """
    Return up to `per_class` rows for each of Positive/Negative/Neutral.
    If a class has fewer rows than requested, take all that exist.
    Drops any rows whose Sentiment can't be normalized.
    """
    if label_col not in df.columns:
        raise ValueError(f"Expected column '{label_col}' not found")

    tmp = df.copy()
    tmp["_sent_str"] = tmp[label_col].apply(_normalize_sent_str)
    tmp = tmp[tmp["_sent_str"].isin(["Positive", "Neutral", "Negative"])]

    frames = []
    for lab in ["Positive", "Negative", "Neutral"]:
        block = tmp[tmp["_sent_str"] == lab]
        take = min(per_class, len(block))
        if take > 0:
            frames.append(block.sample(n=take, random_state=seed, replace=False))

    if not frames:
        raise ValueError("No rows matched Positive/Negative/Neutral after normalization.")

    out = pd.concat(frames, axis=0).drop(columns=["_sent_str"]).reset_index(drop=True)
    # Shuffle for good measure (preserve randomness but reproducible)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def run_prompt_mode(client: OpenAI, data_path: Path, mode: str | None, per_class: int | None):
    mode = (mode or "all").lower()

    if mode in (None, "all", "everything"):
        # 1
        run_sentiment_demo(client, data_path, OUTPUT_DIR1, n_rows=N_SAMPLE, sheet=SHEET_NAME, per_class=per_class)
        # 2
        pairs_xlsx = stage_generate_counterfactuals(client, data_path, OUTPUT_DIR2, n_rows=N_SAMPLE, per_class=per_class)
        stage_evaluate_counterfactuals(client, pairs_xlsx, OUTPUT_DIR2)
        # 3
        df3 = load_dataframe_script3(data_path)
        generate_counterfactuals_script3(df3, client, OUTPUT_DIR3, limit_rows=CF_LIMIT_ROWS, per_class=per_class)
        # 4
        src_json = OUTPUT_DIR3 / "counterfactual_results.json"
        data = load_counterfactual_json_v4(src_json)
        filter_counterfactuals_v4(data, client, OUTPUT_DIR4)
        # 5
        src_filtered = OUTPUT_DIR4 / "filtered_counterfactuals.json"
        data2 = load_filtered_json_v5(src_filtered)
        evaluate_explanations_v5(data2, client, OUTPUT_DIR5)
        return

    if mode == "one":
        run_sentiment_demo(client, data_path, OUTPUT_DIR1, n_rows=N_SAMPLE, sheet=SHEET_NAME, per_class=per_class)
        return

    if mode == "two":
        pairs_xlsx = stage_generate_counterfactuals(client, data_path, OUTPUT_DIR2, n_rows=N_SAMPLE, per_class=per_class)
        stage_evaluate_counterfactuals(client, pairs_xlsx, OUTPUT_DIR2)
        return

    if mode == "three":
        df3 = load_dataframe_script3(data_path)
        generate_counterfactuals_script3(df3, client, OUTPUT_DIR3, limit_rows=CF_LIMIT_ROWS, per_class=per_class)
        return

    if mode == "four":
        src_json = OUTPUT_DIR3 / "counterfactual_results.json"
        data = load_counterfactual_json_v4(src_json)
        filter_counterfactuals_v4(data, client, OUTPUT_DIR4)
        return

    if mode == "five":
        src_filtered = OUTPUT_DIR4 / "filtered_counterfactuals.json"
        data2 = load_filtered_json_v5(src_filtered)
        evaluate_explanations_v5(data2, client, OUTPUT_DIR5)
        return

    raise ValueError(f"Unknown --prompt mode: {mode!r}")

# =========================================================
# ENV & CLIENT
# =========================================================
def init_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no está definido en el entorno/.env")
    return OpenAI(api_key=api_key)

def check_quota(n: int = 1):
    """Consume n llamadas del presupuesto global."""
    global API_CALLS_USED
    if API_CALLS_USED + n > MAX_CALLS:
        raise RuntimeError(f"Se alcanzó el límite de {MAX_CALLS} llamadas a la API.")
    API_CALLS_USED += n

# =========================================================
# DATA LOADING (robusto para CSV/TSV/Excel/Parquet)
# =========================================================
def load_dataset_robust(path: Path, sheet: int | str | None = 0) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {path}")

    ext = Path(path).suffix.lower()
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

    # Default: intentar CSV directo
    return pd.read_csv(path)

# =========================================================
# ================== SCRIPT 1: CLASIFICACIÓN ==================
# (PROMPT MANTENIDO TAL CUAL)
# =========================================================

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

def classify_sentiment_llm(client: OpenAI, text: str) -> dict:
    """Llama al modelo con respuesta JSON estricta."""
    check_quota(1)
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
            {"role": "user", "content": build_prompt(text)},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Modelo devolvió JSON inválido: {raw[:200]}...")
    return data

def run_sentiment_demo(client: OpenAI, data_path: Path, output_dir: Path,
                       n_rows: int = N_SAMPLE, sheet=0, per_class: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset_robust(data_path, sheet=sheet)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"El archivo no tiene columnas esperadas: falta(n) {missing}")

    if per_class is not None:
        df = balanced_subset(df, per_class, label_col="Sentiment")
        sample_df = df  # use exactly the balanced set
    else:
        sample_df = df.head(n_rows).copy().reset_index(drop=True)

    sample_df = sample_df.reset_index(drop=True)

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
            show_progress("Classify", i+1, len(sample_df))
        except Exception as e:
            print(f"[CLS {i+1}/{len(sample_df)}] Error: {e}")
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

    (output_dir / "sentiment_llm_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8"
    )
    pd.DataFrame(aug_rows).to_csv(output_dir / "sentiment_llm_results.csv", index=False, encoding="utf-8-sig")
    print(f"Saved: {output_dir/'sentiment_llm_results.jsonl'}")
    print(f"Saved: {output_dir/'sentiment_llm_results.csv'}")

# =========================================================
# ============== SCRIPT 2: FLIPS + EVALUACIÓN ===============
# (PROMPTS MANTENIDOS TAL CUAL DEL SEGUNDO SCRIPT)
# =========================================================
def detect_label_scheme(series: pd.Series):
    return (
        "neg0",
        {0: "Negative", 1: "Neutral", 2: "Positive"},
        {"negative": 0, "neutral": 1, "positive": 2},
    )

def target_flip_label(src_label_str: str) -> str:
    lk = (src_label_str or "").strip().lower()
    if lk == "positive":
        return "Negative"
    if lk == "negative":
        return "Positive"
    return "Positive"

def build_generation_prompt(original: str, src_label: str, target_label: str) -> str:
    return f"""You are rewriting a WhatsApp message to flip its sentiment while preserving core meaning and context.

Original message:
\"\"\"{original}\"\"\"\

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

_CODEFENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
def _strip_code_fences(s: str) -> str:
    return _CODEFENCE_RE.sub("", (s or "").strip())

def call_json(client: OpenAI, system: str, user: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> dict:
    check_quota(1)
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    raw = resp.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw2 = _strip_code_fences(raw)
        return json.loads(raw2)

def generate_flip(client: OpenAI, original: str, src_label: str, target_label: str) -> Tuple[str, str]:
    out = call_json(
        client,
        "You are a careful rewriting assistant. Return ONLY valid JSON as specified; no extra preamble or commentary.",
        build_generation_prompt(original, src_label, target_label),
    )
    return (out.get("flipped_text") or "").strip(), (out.get("explanation") or "").strip()

def evaluate_counterfactual(client: OpenAI, original: str, flipped: str, transformation: str, explanation: str) -> dict:
    return call_json(
        client,
        "You are an NLP evaluator. Respond ONLY with valid JSON according to the given schema; no text before or after.",
        build_counterfactual_prompt(original, flipped, transformation, explanation),
    )

def stage_generate_counterfactuals(client: OpenAI, src_xlsx: Path, out_dir: Path,
                                   n_rows: int = N_SAMPLE, per_class: int | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset_robust(src_xlsx, sheet=0)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"El XLSX no tiene columnas esperadas: {missing}")

    if per_class is not None:
        df = balanced_subset(df, per_class, label_col="Sentiment")
    else:
        df = df.head(n_rows).copy().reset_index(drop=True)

    scheme, num2str, _ = detect_label_scheme(df["Sentiment"])
    # Ensure ints are OK (if already strings, this will noop some rows)
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
        show_progress("Flip gen", i+1, len(df))

    out_xlsx = out_dir / "counterfactual_pairs.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        pd.DataFrame(pairs).to_excel(w, index=False)
    print(f"Saved flips XLSX: {out_xlsx}")
    return out_xlsx

def stage_evaluate_counterfactuals(client: OpenAI, pairs_xlsx: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset_robust(pairs_xlsx, sheet=0)
    expected = {"original", "flipped", "transformation", "explanation"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas esperadas en pairs XLSX: {missing}")

    results, aug_rows = [], []
    for i, row in df.iterrows():
        try:
            res = evaluate_counterfactual(
                client,
                str(row["original"]),
                str(row["flipped"]),
                str(row["transformation"]),
                str(row["explanation"]),
            )
            results.append(res)
            aug_rows.append({**row.to_dict(), **res})
            show_progress("Counterfactual eval", i+1, len(df))
        except Exception as e:
            aug_rows.append({**row.to_dict(), "error": str(e)})
            print(f"[EVAL {i+1}/{len(df)}] Error: {e}")

    (out_dir / "counterfactual_eval_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8"
    )
    with pd.ExcelWriter(out_dir / "counterfactual_eval_results.xlsx", engine="openpyxl") as w:
        pd.DataFrame(aug_rows).to_excel(w, index=False)

    print(f"Saved eval JSONL: {out_dir/'counterfactual_eval_results.jsonl'}")
    print(f"Saved eval XLSX:  {out_dir/'counterfactual_eval_results.xlsx'}")

# =========================================================
# ============== SCRIPT 3: CF TRIPLES + APLANADO ==============
# (PROMPT MANTENIDO TAL CUAL DEL TERCER SCRIPT)
# =========================================================
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

def map_sentiment(value):
    v = str(value).strip()
    if v == "0":
        return "Negative"
    if v == "1":
        return "Neutral"
    if v == "2":
        return "Positive"
    return "Unknown"

def maybe_fix_mojibake(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return s
    try:
        repaired = s.encode("latin-1").decode("utf-8")
        return repaired if repaired != s else s
    except Exception:
        return s

def normalize_list(x: Any) -> List[str]:
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
    o = original.split()
    c = cf_text.split()
    diff = list(difflib.ndiff(o, c))
    adds, seen, out = [], set(), []
    for tok in diff:
        if tok.startswith("+ "):
            word = tok[2:].strip(",.;:()[]{}!?¡¿\"'“”‘’")
            if word:
                lw = word.lower()
                if lw not in seen:
                    seen.add(lw)
                    out.append(word)
                    if len(out) >= max_items:
                        break
    return out

def extract_json_list(content: str):
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

    def _ensure(cf: Dict[str, Any]) -> Dict[str, Any]:
        cf_text = cf.get("cf_text", "")
        comps = normalize_list(cf.get("components_changed", []))
        flip = cf.get("flip_explanation") or ""
        return {"cf_text": cf_text, "components_changed": comps, "flip_explanation": flip}

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [_ensure(x if isinstance(x, dict) else {"cf_text": str(x)}) for x in parsed]
        if isinstance(parsed, dict):
            return [_ensure(parsed)]
    except Exception:
        pass

    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list):
                return [_ensure(x if isinstance(x, dict) else {"cf_text": str(x)}) for x in parsed]
        except Exception:
            s = block

    cf_texts = re.findall(r'"cf_text"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
    if cf_texts:
        return [{"cf_text": bytes(ct, "utf-8").decode("unicode_escape"),
                 "components_changed": [],
                 "flip_explanation": ""} for ct in cf_texts]

    return [{"cf_text": content, "components_changed": [], "flip_explanation": ""}]

def flatten_records(json_records: List[Dict[str, Any]], fix_mojibake: bool = True) -> pd.DataFrame:
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

def save_flat_outputs(df_flat: pd.DataFrame, output_dir: Path, base_name: str = "counterfactual_results_flat"):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{base_name}.csv"
    df_flat.to_csv(csv_path, index=False, encoding="utf-8-sig")
    jsonl_path = output_dir / f"{base_name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for _, row in df_flat.iterrows():
            rec = dict(row)
            rec["components_changed_list"] = list(rec.get("components_changed_list", []))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(csv_path), str(jsonl_path)

def load_dataframe_script3(src_path: Path) -> pd.DataFrame:
    ext = src_path.suffix.lower()
    if ext == ".xlsx":
        df = pd.read_excel(src_path, engine="openpyxl")
    else:
        df = pd.read_csv(src_path, encoding="latin1")
    missing = {"text", "Sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df

def generate_counterfactuals_script3(df: pd.DataFrame, client: OpenAI, output_dir: Path, limit_rows: int = 3, per_class: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    if per_class is not None:
        df = balanced_subset(df, per_class, label_col="Sentiment")
    else:
        df = df.head(limit_rows).copy()

    json_records: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        original_message = str(row["text"])
        original_sentiment = map_sentiment(row["Sentiment"])

        prompt = PROMPT_TEMPLATE.format(
            original_message=original_message,
            original_sentiment=original_sentiment
        )

        show_progress("CF triples", idx+1, len(df))
        try:
            check_quota(1)
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

        cf_list = extract_json_list(content)

        fixed_cf_list = []
        for cf in cf_list:
            cf_text = maybe_fix_mojibake(cf.get("cf_text", ""))
            comps = normalize_list(cf.get("components_changed", []))
            flip = cf.get("flip_explanation") or ""
            if not comps:
                comps = infer_components_changed(original_message, cf_text)
            if not flip:
                flip = "Auto-inferred changes from word-level diff." if comps else "No explanation provided."
            fixed_cf_list.append({"cf_text": cf_text, "components_changed": comps, "flip_explanation": flip})

        json_records.append({
            "original_message": maybe_fix_mojibake(original_message),
            "original_sentiment": original_sentiment,
            "counterfactuals": fixed_cf_list
        })

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

    json_path = output_dir / "counterfactual_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "counterfactual_results.csv"
    pd.DataFrame(table_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    df_flat = flatten_records(json_records, fix_mojibake=True)
    flat_csv, flat_jsonl = save_flat_outputs(df_flat, output_dir, base_name="counterfactual_results_flat")

    print(f"\n✅ Saved:\n- {json_path}\n- {csv_path}\n- {flat_csv}\n- {flat_jsonl}")

# =========================================================
# ============== SCRIPT 4: FILTRO DE CFs (nuevo) ==============
# (PROMPT MANTENIDO TAL CUAL)
# =========================================================
FILTERING_PROMPT = """(b) Counterfactual Filtering Prompt
You are a sentiment evaluation assistant. Your task is to select
the best counterfactual rewrite of a message.
ORIGINAL MESSAGE
"{original}"
(Sentiment: {original_sentiment})
COUNTERFACTUAL CANDIDATES
1. "{cf1}"
2. "{cf2}"
3. "{cf3}"
INSTRUCTIONS
Your goal is to identify which counterfactual most effectively
flips the sentiment while remaining realistic and fluent.
- Flip sentiment plausibly
- Sound natural in WhatsApp chat
- Preserve meaning/context where possible
RESPONSE FORMAT (JSON only):
{{
"selected_cf": "...",
"justification": "...",
"predicted_sentiment": "Positive / Negative"
}}
"""

def strip_code_fences_v4(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_json_response_v4(raw: str) -> Dict[str, Any]:
    s = strip_code_fences_v4(raw)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            s = block
    def rx(key: str):
        return re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
    selected_cf = (rx("selected_cf").group(1) if rx("selected_cf") else "")
    justification = (rx("justification").group(1) if rx("justification") else "")
    predicted = (rx("predicted_sentiment").group(1) if rx("predicted_sentiment") else "")
    selected_cf = selected_cf.encode("utf-8").decode("unicode_escape")
    justification = justification.encode("utf-8").decode("unicode_escape")
    return {"selected_cf": selected_cf, "justification": justification, "predicted_sentiment": predicted}

def normalize_predicted_sentiment_v4(pred: str) -> str:
    pred = (pred or "").strip().lower()
    if "pos" in pred:
        return "Positive"
    if "neg" in pred:
        return "Negative"
    return ""

def safe_cf_texts_v4(entry: Dict[str, Any]) -> List[str]:
    if all(k in entry for k in ("cf1", "cf2", "cf3")):
        cfs = [str(entry.get("cf1", "")), str(entry.get("cf2", "")), str(entry.get("cf3", ""))]
        return [t.replace('"', '\\"') for t in cfs]
    cfs_list = entry.get("counterfactuals") or []
    texts: List[str] = []
    for cf in cfs_list:
        if isinstance(cf, dict):
            texts.append(str(cf.get("cf_text", "") or ""))
        elif isinstance(cf, str):
            texts.append(cf)
    texts = (texts + ["", "", ""])[:3]
    return [t.replace('"', '\\"') for t in texts]

def unescape_quotes_v4(s: str) -> str:
    return (s or "").replace('\\"', '"')

def maybe_fix_mojibake_v4(s: str) -> str:
    if not isinstance(s, str):
        return s
    if "Ã" in s or "Â" in s:
        try:
            return bytes(s, "latin1").decode("utf-8")
        except Exception:
            return s
    return s

def load_counterfactual_json_v4(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path.name} is not a list")
        return data

def filter_counterfactuals_v4(data: List[Dict[str, Any]], client: OpenAI, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json_records: List[Dict[str, Any]] = []
    out_rows: List[Dict[str, Any]] = []

    for i, item in enumerate(data, 1):
        original = maybe_fix_mojibake_v4(str(item.get("original_message", "")))
        original_sentiment = str(item.get("original_sentiment", ""))

        cf1_raw, cf2_raw, cf3_raw = safe_cf_texts_v4(item)
        prompt = FILTERING_PROMPT.format(
            original=original.replace('"', '\\"'),
            original_sentiment=original_sentiment,
            cf1=cf1_raw, cf2=cf2_raw, cf3=cf3_raw
        )

        show_progress("Filter CFs", i, len(data))
        try:
            check_quota(1)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a precise evaluator that outputs only valid JSON for the specified schema."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:
            print("OpenAI error:", e)
            content = '{"selected_cf":"","justification":"API error","predicted_sentiment":""}'

        parsed = parse_json_response_v4(content)
        selected_cf = parsed.get("selected_cf", "")
        justification = parsed.get("justification", "")
        predicted = normalize_predicted_sentiment_v4(parsed.get("predicted_sentiment", ""))

        cf1_clean = maybe_fix_mojibake_v4(unescape_quotes_v4(cf1_raw))
        cf2_clean = maybe_fix_mojibake_v4(unescape_quotes_v4(cf2_raw))
        cf3_clean = maybe_fix_mojibake_v4(unescape_quotes_v4(cf3_raw))
        selected_clean = maybe_fix_mojibake_v4(unescape_quotes_v4(selected_cf))

        record = {
            "original_message": original,
            "original_sentiment": original_sentiment,
            "cf1": cf1_clean,
            "cf2": cf2_clean,
            "cf3": cf3_clean,
            "selected_cf": selected_clean,
            "justification": justification,
            "predicted_sentiment": predicted
        }
        out_json_records.append(record)
        out_rows.append({
            "original_message": original,
            "original_sentiment": original_sentiment,
            "cf1": cf1_clean,
            "cf2": cf2_clean,
            "cf3": cf3_clean,
            "selected_cf": selected_clean,
            "predicted_sentiment": predicted,
            "justification": justification
        })
        time.sleep(0.5)

    with (output_dir / "filtered_counterfactuals.json").open("w", encoding="utf-8") as f:
        json.dump(out_json_records, f, indent=2, ensure_ascii=False)
    pd.DataFrame(out_rows).to_csv(output_dir / "filtered_counterfactuals.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved:\n- {output_dir/'filtered_counterfactuals.json'}\n- {output_dir/'filtered_counterfactuals.csv'}")

# =========================================================
# ============== SCRIPT 5: EVAL DE EXPLICACIONES ==============
# (PROMPT MANTENIDO TAL CUAL DEL QUINTO SCRIPT)
# =========================================================
EVAL_PROMPT = """Explanation Evaluation Prompt
You are a language
model tasked with evaluating the quality of a
sentiment explanation. Evaluate the explanation
for the following: 1. Faithfulness – Does it
reflect the original message and prediction without
hallucinating? 2. Contextual Appropriateness – Is
it culturally and linguistically aware? 3. Logical
Coherence – Is it internally consistent and justified?
4. Clarity and Completeness – Is it clear, specific,
and sufficient?
Message:
"{message}"
Predicted Sentiment: {prediction}
Explanation: "{explanation}"
Return ONLY this JSON:
{{
"faithfulness": 0 or 1,
"contextual_appropriateness": 0 or 1,
"logical_coherence": 0 or 1,
"clarity_and_completeness": 0 or 1,
"annotator_comment": "optional comment (string)"
}}
"""

def strip_code_fences_v5(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_eval_json_v5(raw: str) -> Dict[str, Any]:
    """
    Devuelve dict con:
    faithfulness, contextual_appropriateness, logical_coherence,
    clarity_and_completeness, annotator_comment.
    """
    s = strip_code_fences_v5(raw)
    # JSON directo
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Primer bloque {...}
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        blk = m.group(0)
        try:
            obj = json.loads(blk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            s = blk
    # Regex fallback
    def rx_int(k):
        m = re.search(rf'"{k}"\s*:\s*(0|1)', s)
        return int(m.group(1)) if m else None
    def rx_str(k):
        m = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
        return (m.group(1).encode("utf-8").decode("unicode_escape")) if m else ""
    parsed = {
        "faithfulness": rx_int("faithfulness"),
        "contextual_appropriateness": rx_int("contextual_appropriateness"),
        "logical_coherence": rx_int("logical_coherence"),
        "clarity_and_completeness": rx_int("clarity_and_completeness"),
        "annotator_comment": rx_str("annotator_comment"),
    }
    for k in ["faithfulness", "contextual_appropriateness", "logical_coherence", "clarity_and_completeness"]:
        if parsed[k] is None:
            parsed[k] = 0
    return parsed

def maybe_fix_mojibake_v5(s: str) -> str:
    if not isinstance(s, str):
        return s
    if "Ã" in s or "Â" in s:
        try:
            return bytes(s, "latin1").decode("utf-8")
        except Exception:
            return s
    return s

def load_filtered_json_v5(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = data.get("items") or data.get("data") or []
        if not isinstance(data, list):
            raise ValueError("filtered_counterfactuals.json must be a JSON list")
        return data

def evaluate_explanations_v5(data: List[Dict[str, Any]], client: OpenAI, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_records = []
    csv_rows = []

    for i, item in enumerate(data, 1):
        original_message = maybe_fix_mojibake_v5(str(item.get("original_message", "")))
        selected_cf = maybe_fix_mojibake_v5(str(item.get("selected_cf", "")))
        predicted = str(item.get("predicted_sentiment", ""))
        justification = str(item.get("justification", ""))

        prompt = EVAL_PROMPT.format(
            message=selected_cf.replace('"', '\\"'),
            prediction=predicted,
            explanation=justification.replace('"', '\\"')
        )

        show_progress("Explain eval", i, len(data))        
        try:
            check_quota(1)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You output only the requested JSON object."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:
            print("OpenAI error:", e)
            content = '{"faithfulness":0,"contextual_appropriateness":0,"logical_coherence":0,"clarity_and_completeness":0,"annotator_comment":"API error"}'

        parsed = parse_eval_json_v5(content)

        record = {
            "original_message": original_message,
            "selected_cf": selected_cf,
            "predicted_sentiment": predicted,
            "justification": justification,
            "faithfulness": int(parsed.get("faithfulness", 0)),
            "contextual_appropriateness": int(parsed.get("contextual_appropriateness", 0)),
            "logical_coherence": int(parsed.get("logical_coherence", 0)),
            "clarity_and_completeness": int(parsed.get("clarity_and_completeness", 0)),
            "annotator_comment": parsed.get("annotator_comment", ""),
        }
        json_records.append(record)

        csv_rows.append({
            "original_message": original_message,
            "selected_cf": selected_cf,
            "predicted_sentiment": predicted,
            "justification": justification,
            "faithfulness": record["faithfulness"],
            "contextual_appropriateness": record["contextual_appropriateness"],
            "logical_coherence": record["logical_coherence"],
            "clarity_and_completeness": record["clarity_and_completeness"],
            "annotator_comment": record["annotator_comment"],
        })

        time.sleep(0.3)

    with (output_dir / "explanation_evals.json").open("w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2, ensure_ascii=False)

    pd.DataFrame(csv_rows).to_csv(output_dir / "explanation_evals.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved:\n- {output_dir/'explanation_evals.json'}\n- {output_dir/'explanation_evals.csv'}")

# =========================================================
# MAIN: corre los cinco scripts en secuencia
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Run sentiment pipelines.")
    parser.add_argument("--prompt",
                        choices=["all", "one", "two", "three", "four", "five"],
                        help="Which part to run. Omit to run default full pipeline.",
                        default=None)
    parser.add_argument("--n", type=str,
                        help="Override N; number, 'full', or -1 for all rows.",
                        default=None)
    parser.add_argument("--per-class", type=int,
                        help="Balanced sampling: N per class for scripts 1–3.",
                        default=None)
    args = parser.parse_args()

    client = init_client()
    data_path = BASE_PATH / INPUT_FILE

    # --n can still apply (e.g., when --per-class not set)
    apply_n_override(args.n, data_path)

    # Dispatch with per-class
    run_prompt_mode(client, data_path, args.prompt, args.per_class)

if __name__ == "__main__":
    main()
