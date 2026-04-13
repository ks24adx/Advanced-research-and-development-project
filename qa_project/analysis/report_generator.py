"""
analysis/report_generator.py
─────────────────────────────────────────────────────────────
Generates a Markdown results report from pipeline outputs,
ready to paste into a document or render as HTML.
"""

from __future__ import annotations
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import OUTPUT_DIR, CONDITION_LABELS


def generate_markdown_report(split_name: str,
                               df: pd.DataFrame,
                               condition_results: dict[str, pd.DataFrame],
                               chart_paths: dict[str, Path] | None = None) -> Path:
    """
    Write a full Markdown report to OUTPUT_DIR/<split_name>/report.md
    and return the file path.
    """
    lines = []
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# QA Error Analysis: Knowledge vs Reasoning Failures",
        f"**Split:** {split_name.upper()}  |  **Generated:** {ts}",
        "",
        "---",
        "",
    ]

    # ── 1. Dataset Summary ────────────────────────────────────────────────────
    lines += ["## 1. Dataset Summary", ""]
    total    = len(df)
    k_cnt    = (df["error_type"] == "knowledge").sum()
    r_cnt    = (df["error_type"] == "reasoning").sum()
    correct  = (df["error_type"] == "none").sum()
    ambig    = total - k_cnt - r_cnt - correct

    lines += [
        f"| Metric | Count | % |",
        f"|--------|-------|---|",
        f"| Total questions | {total} | 100% |",
        f"| Correct (no error) | {correct} | {correct/total*100:.1f}% |",
        f"| Knowledge errors | {k_cnt} | {k_cnt/total*100:.1f}% |",
        f"| Reasoning errors | {r_cnt} | {r_cnt/total*100:.1f}% |",
        f"| Ambiguous | {ambig} | {ambig/total*100:.1f}% |",
        "",
    ]

    # ── 2. Dataset Composition ────────────────────────────────────────────────
    lines += ["## 2. Dataset Composition", ""]
    comp = df.groupby("source_dataset")["question_id"].count().reset_index()
    comp.columns = ["Dataset", "Questions"]
    lines += _df_to_md(comp)
    lines += [""]

    # ── 3. Accuracy Table ─────────────────────────────────────────────────────
    lines += ["## 3. Accuracy by Condition (Exact Match %)", ""]
    rows = []
    for cond, res in condition_results.items():
        row = {"Condition": CONDITION_LABELS.get(cond, cond)}
        for ds in df["source_dataset"].unique():
            sub = res[res["source_dataset"] == ds]
            row[ds] = f"{sub['em'].mean()*100:.1f}%" if len(sub) else "N/A"
        all_em = res["em"].mean() * 100
        row["Average"] = f"{all_em:.1f}%"
        rows.append(row)
    acc_df = pd.DataFrame(rows)
    lines += _df_to_md(acc_df)
    lines += [""]

    # ── 4. Error distribution per dataset ────────────────────────────────────
    lines += ["## 4. Error Distribution per Dataset", ""]
    dist_rows = []
    for ds in df["source_dataset"].unique():
        sub = df[df["source_dataset"] == ds]
        total_ds = len(sub)
        for et in ["knowledge", "reasoning", "none", "ambiguous"]:
            cnt = (sub["error_type"] == et).sum()
            if cnt > 0:
                dist_rows.append({
                    "Dataset"    : ds,
                    "Error Type" : et,
                    "Count"      : cnt,
                    "Percentage" : f"{cnt/total_ds*100:.1f}%",
                })
    lines += _df_to_md(pd.DataFrame(dist_rows))
    lines += [""]

    # ── 5. CoT vs Standard ────────────────────────────────────────────────────
    lines += ["## 5. CoT vs Standard Prompting by Error Type", ""]
    cot_rows = []
    std_df = condition_results.get("closed_book", pd.DataFrame())
    cot_df = condition_results.get("cot_only",    pd.DataFrame())

    for et in ["knowledge", "reasoning", "none"]:
        df_et   = df[df["error_type"] == et]
        qids    = set(df_et["question_id"])

        std_sub = std_df[std_df["question_id"].isin(qids)] if len(std_df) else pd.DataFrame()
        cot_sub = cot_df[cot_df["question_id"].isin(qids)] if len(cot_df) else pd.DataFrame()

        std_em  = f"{std_sub['em'].mean()*100:.1f}%" if len(std_sub) else "N/A"
        cot_em  = f"{cot_sub['em'].mean()*100:.1f}%" if len(cot_sub) else "N/A"

        if len(std_sub) and len(cot_sub):
            gain = cot_sub["em"].mean()*100 - std_sub["em"].mean()*100
            gain_str = f"+{gain:.1f}pp" if gain >= 0 else f"{gain:.1f}pp"
        else:
            gain_str = "N/A"

        cot_rows.append({
            "Error Type"       : et,
            "Standard EM"      : std_em,
            "CoT EM"           : cot_em,
            "CoT Gain"         : gain_str,
        })
    lines += _df_to_md(pd.DataFrame(cot_rows))
    lines += [""]

    # ── 6. RAG vs Closed-Book per error type ─────────────────────────────────
    lines += ["## 6. RAG vs Closed-Book by Error Type", ""]
    rag_df  = condition_results.get("rag_only", pd.DataFrame())
    rag_rows = []
    for et in ["knowledge", "reasoning", "none"]:
        df_et   = df[df["error_type"] == et]
        qids    = set(df_et["question_id"])
        cb_sub  = std_df[std_df["question_id"].isin(qids)] if len(std_df) else pd.DataFrame()
        rag_sub = rag_df[rag_df["question_id"].isin(qids)] if len(rag_df) else pd.DataFrame()

        cb_em  = f"{cb_sub['em'].mean()*100:.1f}%" if len(cb_sub) else "N/A"
        rag_em = f"{rag_sub['em'].mean()*100:.1f}%" if len(rag_sub) else "N/A"

        if len(cb_sub) and len(rag_sub):
            gain = rag_sub["em"].mean()*100 - cb_sub["em"].mean()*100
            gain_str = f"+{gain:.1f}pp" if gain >= 0 else f"{gain:.1f}pp"
        else:
            gain_str = "N/A"

        rag_rows.append({
            "Error Type"   : et,
            "Closed-Book EM": cb_em,
            "RAG EM"       : rag_em,
            "RAG Gain"     : gain_str,
        })
    lines += _df_to_md(pd.DataFrame(rag_rows))
    lines += [""]

    # ── 7. Charts ─────────────────────────────────────────────────────────────
    if chart_paths:
        lines += ["## 7. Generated Charts", ""]
        for name, path in chart_paths.items():
            lines.append(f"- `{path.name}` — {name.replace('_', ' ').title()}")
        lines += [""]

    # ── 8. Methodology notes ─────────────────────────────────────────────────
    lines += [
        "## 8. Methodology Notes",
        "",
        "- **Evaluation metric**: Exact Match (EM) after normalisation "
        "(lowercase, strip articles and punctuation)",
        "- **Error classification**: Rule-based using closed/open-book EM labels "
        "and retrieval success flag",
        "- **Model**: GPT-4 Turbo (temperature = 0)",
        "- **Retrieval**: DPR top-5 passages from Wikipedia",
        "- **CoT**: Chain-of-thought prompting with 'Final Answer:' extraction",
        "",
        "---",
        "_Report generated automatically by analysis/report_generator.py_",
    ]

    # ── Write file ────────────────────────────────────────────────────────────
    out_dir  = OUTPUT_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ── Helper ────────────────────────────────────────────────────────────────────

def _df_to_md(df: pd.DataFrame) -> list[str]:
    """Convert a DataFrame to a Markdown table (list of lines)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return [header, sep] + rows
