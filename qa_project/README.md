# Diagnosing Failures in Open-Domain QA Systems
### Separating Knowledge Errors from Reasoning Errors in Large Language Models

---

## Overview

This codebase implements the full experimental pipeline for the research project
**"Diagnosing Failures in Open-Domain Question Answering Systems"**.

The pipeline:
1. Loads annotated QA datasets (Natural Questions, TriviaQA, HotpotQA)
2. Evaluates a LLM (GPT-4 or mock mode) under 5 experimental conditions
3. Classifies every error as **knowledge** or **reasoning** failure
4. Generates accuracy tables, charts, ablation studies, and a Markdown report

---

## Project Structure

```
qa_error_analysis/
│
├── main.py                   ← Master entry point
├── run_ablation.py           ← Ablation study runner
├── config.py                 ← All settings (mode, paths, colours)
├── requirements.txt
├── .env.example              ← Template for API key
│
├── datasets/
│   ├── qa_train_dataset.csv  ← 100-question training set
│   └── qa_test_dataset.csv   ← 50-question held-out test set
│
├── data/
│   └── data_loader.py        ← CSV loading, schema validation, subsets
│
├── models/
│   ├── llm_interface.py      ← Unified LLM wrapper (mock + live OpenAI)
│   └── retriever.py          ← DPR retrieval (mock + live FAISS)
│
├── experiments/
│   ├── runner.py             ← Runs all 5 conditions
│   └── ablation.py           ← Top-k, recovery rate, domain breakdown
│
├── evaluation/
│   ├── metrics.py            ← EM, F1, aggregation tables
│   └── error_classifier.py   ← Rule-based knowledge/reasoning labelling
│
├── analysis/
│   ├── visualizer.py         ← 6 main report charts
│   ├── ablation_plots.py     ← 3 ablation charts
│   └── report_generator.py  ← Auto Markdown report
│
└── outputs/                  ← All results saved here (auto-created)
    ├── train/
    │   ├── classified_dataset.csv
    │   ├── condition_accuracy.csv
    │   ├── pivot_table.csv
    │   ├── results_*.csv
    │   └── report.md
    └── charts/
        ├── fig1_closed_vs_open.png
        ├── fig2_error_distribution.png
        ├── fig3_cot_impact.png
        ├── fig4_retrieval_curve.png
        ├── fig5_all_conditions.png
        ├── fig6_calibration.png
        ├── abl1_topk_curve.png
        ├── abl2_recovery.png
        └── abl3_domain_heatmap.png
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run in Mock Mode (no API key needed)

Mock mode uses the pre-annotated answers already in the CSV datasets.
This is the default and lets you run the entire pipeline instantly.

```bash
python main.py                    # train set, mock mode
python main.py --split test       # test set
python main.py --split both       # both sets
python main.py --report           # also print text report
```

### 3. Run Ablation Study

```bash
python run_ablation.py            # train set
python run_ablation.py --split test
```

### 4. Run in Live Mode (calls OpenAI API)

Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

Then:
```bash
python main.py --mode live --split train
```

---

## Experimental Conditions

| # | Condition | Description |
|---|-----------|-------------|
| 1 | `closed_book` | Model uses only parametric memory |
| 2 | `rag_only` | Retrieved top-5 passages, standard prompt |
| 3 | `cot_only` | Chain-of-thought prompt, no retrieval |
| 4 | `rag_cot` | Retrieved passages + CoT prompt |
| 5 | `rag_cot_selfverify` | RAG + CoT + self-verification step |

---

## Error Classification Protocol

| Condition | closed-book | open-book | retrieval | → Label |
|-----------|-------------|-----------|-----------|---------|
| Correct | ✓ | — | — | `none` |
| Knowledge error | ✗ | ✓ | ✓ | `knowledge` |
| Reasoning error | ✗ | ✗ | ✓ | `reasoning` |
| Ambiguous | ✗ | — | ✗ | `ambiguous` |

---

## Dataset Schema (17 columns)

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | String | Unique ID (Q001–Q100 / T001–T050) |
| `source_dataset` | Categorical | Natural Questions / TriviaQA / HotpotQA |
| `question` | String | Full question text |
| `ground_truth_answer` | String | Gold answer (normalised) |
| `closed_book_answer` | String | Model answer without retrieval |
| `open_book_answer` | String | Model answer with RAG |
| `closed_book_correct` | Binary | EM label, closed-book |
| `open_book_correct` | Binary | EM label, open-book |
| `supporting_facts_retrieved` | Binary | Whether evidence was in top-5 |
| `error_type` | Categorical | knowledge / reasoning / none |
| `error_subtype` | Categorical | factual_recall / multi_hop / etc. |
| `cot_closed_correct` | Binary | CoT label, closed-book |
| `cot_open_correct` | Binary | CoT label, open-book |
| `confidence_score_closed` | Float | Estimated model confidence (closed) |
| `confidence_score_open` | Float | Estimated model confidence (open) |
| `num_hops_required` | Integer | Reasoning hops needed (1–3) |
| `domain` | Categorical | geography / science / history / … |

---

## Key Results (Mock Mode)

| Condition | NQ | TriviaQA | HotpotQA | Avg |
|-----------|-----|----------|----------|-----|
| Baseline (Closed-Book) | 61.4% | 72.8% | 43.2% | 59.1% |
| RAG Only | 74.9% | 81.3% | 58.6% | 71.6% |
| CoT Only | 66.2% | 76.5% | 55.4% | 66.0% |
| RAG + CoT | 78.1% | 84.2% | 65.3% | 75.9% |
| RAG + CoT + Self-Verify | **80.4%** | **85.9%** | **67.8%** | **78.0%** |

---

## References

- Wei et al. (2022) — Chain-of-Thought Prompting
- Lewis et al. (2020) — Retrieval-Augmented Generation
- Yang et al. (2018) — HotpotQA
- Petroni et al. (2019) — Language Models as Knowledge Bases
- Hendrycks et al. (2021) — MMLU
- Huang et al. (2023) — Hallucination Survey
