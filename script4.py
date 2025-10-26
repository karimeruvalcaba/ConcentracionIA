# -*- coding: utf-8 -*-
import os
import re
import json
import time
from typing import Dict, Any, List

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# Prompts
# =========================
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

# =========================
# Utils
# =========================
def strip_code_fences(s: str) -> str:
    """Remove ``` and optional language tag from a model response."""
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_json_response(raw: str) -> Dict[str, Any]:
    """
    Parse model response into dict with keys:
    selected_cf, justification, predicted_sentiment.

    - Try strict JSON
    - Try first {...} block
    - Fallback to regex extraction of fields
    """
    s = strip_code_fences(raw)

    # 1) Direct JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) First {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            s = block  # keep for regex

    # 3) Regex fallback
    def rx(key: str):
        return re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', s)

    selected_cf = (rx("selected_cf").group(1) if rx("selected_cf") else "")
    justification = (rx("justification").group(1) if rx("justification") else "")
    predicted = (rx("predicted_sentiment").group(1) if rx("predicted_sentiment") else "")

    # Unescape any \" inside captured values
    selected_cf = selected_cf.encode("utf-8").decode("unicode_escape")
    justification = justification.encode("utf-8").decode("unicode_escape")

    return {
        "selected_cf": selected_cf,
        "justification": justification,
        "predicted_sentiment": predicted
    }

def normalize_predicted_sentiment(pred: str) -> str:
    pred = (pred or "").strip().lower()
    if "pos" in pred:
        return "Positive"
    if "neg" in pred:
        return "Negative"
    return ""

def safe_cf_texts(entry: Dict[str, Any]) -> List[str]:
    """
    Return exactly 3 counterfactual texts (pad with empty strings if needed).
    Supports either:
      - entry["counterfactuals"] = [{"cf_text": "..."}...]
      - or direct strings
      - or previously flattened "cf1","cf2","cf3" fields
    """
    # 1) If already flattened:
    if all(k in entry for k in ("cf1", "cf2", "cf3")):
        cfs = [str(entry.get("cf1", "")), str(entry.get("cf2", "")), str(entry.get("cf3", ""))]
        return [t.replace('"', '\\"') for t in cfs]

    # 2) Nested list:
    cfs_list = entry.get("counterfactuals") or []
    texts: List[str] = []
    for cf in cfs_list:
        if isinstance(cf, dict):
            texts.append(str(cf.get("cf_text", "") or ""))
        elif isinstance(cf, str):
            texts.append(cf)

    # pad to 3
    texts = (texts + ["", "", ""])[:3]
    # escape quotes for prompt safety
    texts = [t.replace('"', '\\"') for t in texts]
    return texts

def unescape_quotes(s: str) -> str:
    """Turn \" back into normal quotes for nicer CSVs."""
    return (s or "").replace('\\"', '"')

def maybe_fix_mojibake(s: str) -> str:
    """
    Quick heuristic fix for mojibake like 'XÃ³chitl'.
    If it contains 'Ã' or 'Â', try latin1->utf8 roundtrip.
    """
    if not isinstance(s, str):
        return s
    if "Ã" in s or "Â" in s:
        try:
            return bytes(s, "latin1").decode("utf-8")
        except Exception:
            return s
    return s

def load_counterfactual_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{os.path.basename(path)} is not a list")
        return data

# =========================
# Core
# =========================
def filter_counterfactuals(data: List[Dict[str, Any]], client: OpenAI, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    out_json_records: List[Dict[str, Any]] = []
    out_rows: List[Dict[str, Any]] = []

    for i, item in enumerate(data, 1):
        original = maybe_fix_mojibake(str(item.get("original_message", "")))
        original_sentiment = str(item.get("original_sentiment", ""))

        # Collect/normalize candidate CFs (for prompt)
        cf1_raw, cf2_raw, cf3_raw = safe_cf_texts(item)

        prompt = FILTERING_PROMPT.format(
            original=original.replace('"', '\\"'),
            original_sentiment=original_sentiment,
            cf1=cf1_raw, cf2=cf2_raw, cf3=cf3_raw
        )

        print(f"\n--- Evaluating {i}/{len(data)} ---")
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise evaluator that outputs only valid JSON for the specified schema."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:
            print("OpenAI error:", e)
            content = '{"selected_cf":"","justification":"API error","predicted_sentiment":""}'

        parsed = parse_json_response(content)
        selected_cf = parsed.get("selected_cf", "")
        justification = parsed.get("justification", "")
        predicted = normalize_predicted_sentiment(parsed.get("predicted_sentiment", ""))

        # For outputs (clean up quotes + mojibake so CSV is readable)
        cf1_clean = maybe_fix_mojibake(unescape_quotes(cf1_raw))
        cf2_clean = maybe_fix_mojibake(unescape_quotes(cf2_raw))
        cf3_clean = maybe_fix_mojibake(unescape_quotes(cf3_raw))
        selected_clean = maybe_fix_mojibake(unescape_quotes(selected_cf))

        # JSON record (keeps cf1..cf3 too for convenience)
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

        # CSV row (now includes cf1, cf2, cf3)
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

        time.sleep(0.5)  # gentle rate limit

    # Save JSON
    json_path = os.path.join(output_dir, "filtered_counterfactuals.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json_records, f, indent=2, ensure_ascii=False)

    # Save CSV
    csv_path = os.path.join(output_dir, "filtered_counterfactuals.csv")
    pd.DataFrame(out_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved:\n- {json_path}\n- {csv_path}")

# =========================
# CLI
# =========================
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no está definido en el entorno/.env")

    # OpenAI client
    client = OpenAI(api_key=api_key)

    # Paths (ajústalos si quieres otros)
    base_path = r"C:\Users\abdel\Downloads\concentracionIA"
    src_json = os.path.join(base_path, "results_llms3", "counterfactual_results.json")
    output_dir = os.path.join(base_path, "results_llms4")

    data = load_counterfactual_json(src_json)
    filter_counterfactuals(data, client, output_dir)

if __name__ == "__main__":
    main()
