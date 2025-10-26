from typing import Iterable, List, Optional
import math

import pandas as pd

# ---- Optional NLTK tokenization (with auto-download & safe fallback) ----
_USE_NLTK = True
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.metrics import edit_distance as _nltk_edit_distance

    # Ensure required resources exist; download once if missing
    if _USE_NLTK:
        for _pkg in ["punkt", "punkt_tab"]:
            try:
                nltk.data.find(f"tokenizers/{_pkg}")
            except LookupError:
                nltk.download(_pkg, quiet=True)
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False
    # simple whitespace fallback
    def word_tokenize(text: str) -> List[str]:
        return str(text).split()

    def _nltk_edit_distance(a: List[str], b: List[str]) -> int:
        # basic dynamic-programming edit distance on tokens
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,       # deletion
                    dp[i][j-1] + 1,       # insertion
                    dp[i-1][j-1] + cost,  # substitution
                )
        return dp[m][n]

# ---- Transformers (Perplexity) ----
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class CounterfactualEvaluator:
    """
    Evaluate counterfactual text pairs with:
      - Flip Rate (FR)
      - Token Distance (Dis)  [token-level Levenshtein, normalized]
      - Perplexity (PP)       [causal LM loss with proper padding mask]
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: Optional[str] = None,
        max_length: int = 512,
        use_nltk: bool = True,
    ):
        # Tokenizer / Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

        # Ensure a defined pad token for causal LMs like GPT-2-family
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Quiet the new config warning (harmless if not present)
        try:
            self.model.config.loss_type = "ForCausalLMLoss"
        except Exception:
            pass

        # NLTK toggle (fallback is automatic if unavailable)
        self.use_nltk = use_nltk and _HAS_NLTK

    # ---------- 1) Flip Rate ----------
    @staticmethod
    def flip_rate(y_pred_original: Iterable, y_pred_cf: Iterable) -> float:
        """
        FR = (1/N) * Σ 1[f(x_i) != f(x'_i)]
        """
        y0 = pd.Series(list(y_pred_original))
        y1 = pd.Series(list(y_pred_cf))
        if len(y0) == 0:
            return 0.0
        return (y0 != y1).mean()

    # ---------- 2) Token Distance ----------
    def token_distance(
        self,
        texts_original: Iterable[str],
        texts_cf: Iterable[str],
        normalize: str = "orig",  # "orig" or "max"
    ) -> float:
        """
        Dis = (1/N) * Σ d(x_i, x'_i), where d = edit_distance(tokens)
        Normalization:
          - "orig": divide by len(tokens_orig) (>=1)
          - "max": divide by max(len(tokens_orig), len(tokens_cf)) (>=1)
        """
        xs = list(texts_original)
        cs = list(texts_cf)
        assert len(xs) == len(cs), "texts_original and texts_cf must have same length"

        distances: List[float] = []
        for x, c in zip(xs, cs):
            t_x = word_tokenize(str(x))
            t_c = word_tokenize(str(c))
            raw = _nltk_edit_distance(t_x, t_c)

            if normalize == "max":
                denom = max(len(t_x), len(t_c), 1)
            else:  # "orig"
                denom = max(len(t_x), 1)

            distances.append(raw / denom)

        return float(sum(distances) / len(distances)) if distances else 0.0

    # ---------- 3) Perplexity ----------
    def perplexity(self, texts: Iterable[str]) -> float:
        """
        PP(x) = exp( -1/n Σ log pθ(z_i | z_<i) )
        Uses model NLL on input_ids, masking out padding with -100 in labels.
        Returns the mean perplexity across given texts.
        """
        texts = list(texts)
        if not texts:
            return float("nan")

        ppl: List[float] = []
        for text in texts:
            enc = self.tokenizer(
                str(text),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            labels = enc["input_ids"].clone()
            if "attention_mask" in enc:
                labels[enc["attention_mask"] == 0] = -100

            with torch.no_grad():
                out = self.model(**enc, labels=labels)
                loss = out.loss  # scalar
                ppl.append(math.exp(loss.item()))

        return float(sum(ppl) / len(ppl))

    # ---------- Combined ----------
    def evaluate(
        self,
        df: pd.DataFrame,
        orig_col: str = "original_text",
        cf_col: str = "cf_text",
        orig_pred: str = "orig_pred",
        cf_pred: str = "cf_pred",
    ) -> dict:
        """
        Compute all metrics on a DataFrame with:
          original_text, cf_text, orig_pred, cf_pred
        """
        # Basic checks
        for col in [orig_col, cf_col, orig_pred, cf_pred]:
            if col not in df.columns:
                raise KeyError(f"Missing required column: '{col}'")

        return {
            "flip_rate": self.flip_rate(df[orig_pred], df[cf_pred]),
            "token_distance": self.token_distance(df[orig_col], df[cf_col]),
            "perplexity_original": self.perplexity(df[orig_col].tolist()),
            "perplexity_cf": self.perplexity(df[cf_col].tolist()),
        }
