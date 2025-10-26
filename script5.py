import os
import re
import json
import time
import pandas as pd
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

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


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_eval_json(raw: str) -> Dict[str, Any]:
    """
    Parse model response to dict with keys:
    faithfulness, contextual_appropriateness, logical_coherence,
    clarity_and_completeness, annotator_comment.
    """
    s = strip_code_fences(raw)

    # Try strict JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        blk = m.group(0)
        try:
            obj = json.loads(blk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            s = blk  # continue with regex below

    # Regex fallback for the integer flags and optional comment
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
    # Fill any missing ints with 0 to keep schema consistent
    for k in ["faithfulness", "contextual_appropriateness", "logical_coherence", "clarity_and_completeness"]:
        if parsed[k] is None:
            parsed[k] = 0
    return parsed

def maybe_fix_mojibake(s: str) -> str:
    """Light fix for strings like 'transiciÃ³n' → 'transición'."""
    if not isinstance(s, str):
        return s
    if "Ã" in s or "Â" in s:
        try:
            return bytes(s, "latin1").decode("utf-8")
        except Exception:
            return s
    return s

def load_filtered_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            # In case someone saved a dict with a top-level key
            data = data.get("items") or data.get("data") or []
        if not isinstance(data, list):
            raise ValueError("filtered_counterfactuals.json must be a JSON list")
        return data

def evaluate_explanations(data: List[Dict[str, Any]], client: OpenAI, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    json_records = []
    csv_rows = []

    for i, item in enumerate(data, 1):
        original_message = maybe_fix_mojibake(str(item.get("original_message", "")))
        selected_cf = maybe_fix_mojibake(str(item.get("selected_cf", "")))
        predicted = str(item.get("predicted_sentiment", ""))
        justification = str(item.get("justification", ""))

        # Build prompt with the SELECTED CF as the "Message"
        prompt = EVAL_PROMPT.format(
            message=selected_cf.replace('"', '\\"'),
            prediction=predicted,
            explanation=justification.replace('"', '\\"')
        )

        print(f"--- Evaluating explanation {i}/{len(data)} ---")
        try:
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

        parsed = parse_eval_json(content)

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

        time.sleep(0.3)  # gentle rate limit

    # Save JSON
    json_path = os.path.join(output_dir, "explanation_evals.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2, ensure_ascii=False)

    # Save CSV
    csv_path = os.path.join(output_dir, "explanation_evals.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved:\n- {json_path}\n- {csv_path}")

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no está definido en el entorno/.env")

    client = OpenAI(api_key=api_key)

    base_path = r"C:\Users\abdel\Downloads\concentracionIA"
    src_json = os.path.join(base_path, "results_llms4", "filtered_counterfactuals.json")
    output_dir = os.path.join(base_path, "results_llms5")

    data = load_filtered_json(src_json)
    evaluate_explanations(data, client, output_dir)

if __name__ == "__main__":
    main()
