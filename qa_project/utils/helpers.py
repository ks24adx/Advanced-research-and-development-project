"""
utils/helpers.py
─────────────────────────────────────────────────────────────
Shared utility functions used across the project.
"""

import re
import string
import unicodedata


# ── Text normalisation (standard NLP-QA practice) ───────────────────────────

def normalize_answer(text: str) -> str:
    """
    Normalise an answer string for Exact-Match evaluation.
    Steps: unicode → lowercase → remove articles → remove punctuation
           → collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    text = _remove_articles(text)
    text = _remove_punctuation(text)
    text = _fix_whitespace(text)
    return text.strip()


def _remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def _remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def _fix_whitespace(text: str) -> str:
    return " ".join(text.split())


# ── Token-level F1 ──────────────────────────────────────────────────────────

def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute the token-level F1 score between prediction and ground truth.
    Consistent with the SQuAD evaluation script.
    """
    pred_tokens  = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Exact Match ─────────────────────────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> bool:
    """Return True if normalised prediction equals normalised ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ── Logging helpers ──────────────────────────────────────────────────────────

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    def log_info(msg: str)    -> None: print(Fore.CYAN    + f"[INFO]  {msg}")
    def log_ok(msg: str)      -> None: print(Fore.GREEN   + f"[OK]    {msg}")
    def log_warn(msg: str)    -> None: print(Fore.YELLOW  + f"[WARN]  {msg}")
    def log_error(msg: str)   -> None: print(Fore.RED     + f"[ERROR] {msg}")
    def log_header(msg: str)  -> None:
        print(Fore.BLUE + Style.BRIGHT + "\n" + "═"*60)
        print(Fore.BLUE + Style.BRIGHT + f"  {msg}")
        print(Fore.BLUE + Style.BRIGHT + "═"*60)
except ImportError:
    def log_info(msg):    print(f"[INFO]  {msg}")
    def log_ok(msg):      print(f"[OK]    {msg}")
    def log_warn(msg):    print(f"[WARN]  {msg}")
    def log_error(msg):   print(f"[ERROR] {msg}")
    def log_header(msg):
        print("\n" + "="*60)
        print(f"  {msg}")
        print("="*60)
