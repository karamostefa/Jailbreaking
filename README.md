# Jailbreak / LSEP Zero-Trust guard experiments

Reproduces every number in Sections "Experimental Protocol" and "Results" of the paper.

## Setup (Windows, CPU only)
- Ollama 0.40.16 (portable zip from the official GitHub release) at `C:\Users\DELL\tools\ollama\`
- Models stored in `D:\ollama_models` (see `start_ollama.ps1`)
- Models: `llama3.2:3b`, `qwen2.5:3b`, `deepseek-r1:1.5b` (targets), `llama-guard3:1b` (runtime guard),
  `llama-guard3:8b` (offline judge). Exact digests: `results/models.json`.
- Python 3 standard library only (matplotlib for the figure).

## Data (in `data/`)
- `harmful-behaviors.csv`, `benign-behaviors.csv` — JailbreakBench JBB-Behaviors (Chao et al., 2024)
- `jbb_JBC_manual_*.json` — AIM role-play prompts (JailbreakBench artifacts)
- `jbb_PAIR_black_box_vicuna-13b-v1.5.json` — PAIR prompts found against Vicuna-13B (82 behaviors)
- `pasttense_gpt-3.5-turbo.json` — past-tense reformulations released by Andriushchenko & Flammarion

## Run
```
powershell -File start_ollama.ps1
python build_prompts.py          # -> prompts.jsonl (582 prompts)
python run_all.py                # info, gen, guard_in, guard_out, judge (resumable)
python analyze.py                # -> results/summary.{json,md}, table_asr.tex, fig_asr.pdf
python audit_sheet.py export     # optional human audit of the judge (keep internal)
```
Every stage appends to `results/*.jsonl` and skips finished items, so a stopped run can be restarted.

## Ethics
`results/gen_*.jsonl` and `results/audit_sheet.csv` contain model outputs to harmful requests.
They must not be published. Only ids, labels and aggregate statistics are released.

## Released vs. withheld
Released: prompts, per-item labels (`results/labels.csv`), summaries, tables, figure, model digests.
Withheld on purpose: raw model responses to harmful requests (`gen_*.jsonl`) and the human-audit sheets.
The files in `data/` are public JailbreakBench / past-tense artifacts with the stored model responses removed;
see the original repositories (JailbreakBench/artifacts, tml-epfl/llm-past-tense) for their licenses.
`analyze.py` needs the raw `gen_*`, `guard_*` and `judge_*` files; to re-run the whole study, execute `run_all.py`.
