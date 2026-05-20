"""
In bảng điểm ROUGE-L F1 của tất cả checkpoint theo từng thực nghiệm.
Mỗi bảng có cột Avg tính trung bình các dataset.

Chạy từ thư mục gốc dự án:
    python eval_mta/scripts/visualize_checkpoints.py
    python eval_mta/scripts/visualize_checkpoints.py --eval_root eval_mta/eval_outputs
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATASETS = ["Dolly", "Self-Instruct", "Vicuna", "S-NI"]
STEP_PAT = re.compile(r"^(\d+)$|^epoch\d+_step(\d+)")


def parse_log(log_path: Path) -> dict[str, float] | None:
    text = log_path.read_text(errors="ignore")
    scores = {}
    for dataset in DATASETS:
        pat = re.compile(rf"^{re.escape(dataset)} ROUGE-L F1: ([\d.]+)%", re.MULTILINE)
        matches = pat.findall(text)
        if matches:
            scores[dataset] = float(matches[-1])
    return scores if scores else None


def extract_step(name: str) -> int | None:
    m = STEP_PAT.match(name)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def collect_results(eval_root: Path) -> dict[str, pd.DataFrame]:
    experiments: dict[str, list] = defaultdict(list)

    for log_path in sorted(eval_root.rglob("eval.log")):
        ckpt_dir = log_path.parent
        parent_dir = ckpt_dir.parent
        scores = parse_log(log_path)
        if not scores:
            continue
        step = extract_step(ckpt_dir.name)
        exp_key = str(parent_dir.relative_to(eval_root))
        label = step if step is not None else ckpt_dir.name
        experiments[exp_key].append((label, scores))

    dataframes = {}
    for exp_key, entries in experiments.items():
        rows = [{"step": label, **scores} for label, scores in entries]
        df = pd.DataFrame(rows)
        if all(isinstance(v, int) for v in df["step"]):
            df = df.sort_values("step").reset_index(drop=True)
            df["step"] = df["step"].astype(int)
        score_cols = [d for d in DATASETS if d in df.columns]
        df["Avg"] = df[score_cols].mean(axis=1).round(2)
        dataframes[exp_key] = df

    return dataframes


def short_label(exp_key: str) -> str:
    parts = Path(exp_key).parts
    meaningful = [p for p in parts if p not in ("results", "train", "teamspace",
                                                  "studios", "this_studio")]
    return " / ".join(meaningful[-3:]) if len(meaningful) >= 3 else " / ".join(meaningful)


def fmt_val(col: str, val) -> str:
    if col == "step":
        return str(int(val)) if pd.notna(val) else "-"
    return f"{val:.2f}" if pd.notna(val) else "-"


def print_table(exp_key: str, df: pd.DataFrame):
    label = short_label(exp_key)
    score_cols = [d for d in DATASETS if d in df.columns] + ["Avg"]
    cols = ["step"] + score_cols

    # Format tất cả giá trị thành string trước
    formatted = [[fmt_val(c, row[c]) for c in cols] for _, row in df.iterrows()]
    col_widths = {c: max(len(c), max(len(r[i]) for r in formatted))
                  for i, c in enumerate(cols)}

    sep = "+-" + "-+-".join("-" * col_widths[c] for c in cols) + "-+"
    header = "| " + " | ".join(c.ljust(col_widths[c]) for c in cols) + " |"

    print(f"\n{'=' * len(sep)}")
    print(label)
    print(sep)
    print(header)
    print(sep)
    for row_vals in formatted:
        line = "| " + " | ".join(v.ljust(col_widths[c]) for v, c in zip(row_vals, cols)) + " |"
        print(line)
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", default="eval_mta/eval_outputs/results")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    if not eval_root.exists():
        print(f"[LỖI] Không tìm thấy thư mục: {eval_root}")
        return

    all_data = collect_results(eval_root)
    if not all_data:
        print("Không tìm thấy eval.log nào có kết quả hợp lệ.")
        return

    for exp_key, df in all_data.items():
        print_table(exp_key, df)

    print()


if __name__ == "__main__":
    main()
