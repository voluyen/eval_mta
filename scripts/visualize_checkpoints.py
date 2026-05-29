"""
In bảng điểm ROUGE-L F1 của tất cả checkpoint theo từng thực nghiệm.
Mỗi bảng có cột Avg tính trung bình các dataset.

Chạy từ thư mục gốc dự án:
    python eval_mta/scripts/visualize_checkpoints.py
    python eval_mta/scripts/visualize_checkpoints.py --eval_root eval_mta/eval_outputs
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DATASETS = ["Dolly", "Self-Instruct", "Vicuna", "S-NI"]
DATASET_ALIASES = {
    "dolly": "Dolly",
    "self_instruct": "Self-Instruct",
    "self-instruct": "Self-Instruct",
    "vicuna": "Vicuna",
    "sni": "S-NI",
    "s-ni": "S-NI",
}
STEP_PAT = re.compile(r"^(\d+)$|^epoch\d+_step(\d+)")


def parse_log(log_path: Path) -> dict[str, float] | None:
    text = log_path.read_text(errors="ignore")
    scores = {}
    for dataset in DATASETS:
        pat = re.compile(rf"^{re.escape(dataset)} ROUGE-L F1: ([\d.]+)%", re.MULTILINE)
        matches = pat.findall(text)
        if matches:
            scores[dataset] = float(matches[-1])

    # eval_generate.py format:
    #   dolly            ROUGE-L: 24.04
    for key, dataset in DATASET_ALIASES.items():
        pat = re.compile(rf"^\s*{re.escape(key)}\s+ROUGE-L:\s+([\d.]+)", re.MULTILINE)
        matches = pat.findall(text)
        if matches:
            scores[dataset] = float(matches[-1])
    return scores if scores else None


def parse_json_scores(json_path: Path) -> dict[str, float] | None:
    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return None

    scores = {}
    for key, value in data.items():
        dataset = DATASET_ALIASES.get(key.lower())
        if not dataset or not isinstance(value, dict):
            continue

        score = value.get("rouge_l_avg", value.get("rouge_l_f1"))
        if isinstance(score, (int, float)):
            scores[dataset] = float(score)

    return scores if scores else None


def extract_step(name: str) -> int | None:
    m = STEP_PAT.match(name)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def collect_results(eval_root: Path) -> dict[str, list[dict]]:
    experiments: dict[str, list] = defaultdict(list)
    seen_ckpt_dirs = set()

    result_paths = (
        sorted(eval_root.rglob("scores.json"))
        + sorted(eval_root.rglob("eval.json"))
        + sorted(eval_root.rglob("eval.log"))
    )

    for result_path in result_paths:
        ckpt_dir = result_path.parent
        if ckpt_dir in seen_ckpt_dirs:
            continue
        parent_dir = ckpt_dir.parent
        if result_path.suffix == ".json":
            scores = parse_json_scores(result_path)
        else:
            scores = parse_log(result_path)
        if not scores:
            continue
        seen_ckpt_dirs.add(ckpt_dir)
        step = extract_step(ckpt_dir.name)
        try:
            exp_key = str(parent_dir.relative_to(eval_root))
        except ValueError:
            # Bỏ qua file tổng hợp nằm ngay tại eval_root, không thuộc checkpoint nào.
            continue
        label = step if step is not None else ckpt_dir.name
        experiments[exp_key].append((label, scores))

    tables = {}
    for exp_key, entries in experiments.items():
        rows = []
        for label, scores in entries:
            row = {"step": label, **scores}
            score_cols = [d for d in DATASETS if d in row]
            row["Avg"] = round(sum(row[d] for d in score_cols) / len(score_cols), 2)
            rows.append(row)

        if all(isinstance(row["step"], int) for row in rows):
            rows = sorted(rows, key=lambda row: row["step"])
        else:
            rows = sorted(rows, key=lambda row: str(row["step"]))

        tables[exp_key] = rows

    return tables


def short_label(exp_key: str) -> str:
    parts = Path(exp_key).parts
    meaningful = [p for p in parts if p not in ("results", "train", "teamspace",
                                                  "studios", "this_studio")]
    return " / ".join(meaningful[-3:]) if len(meaningful) >= 3 else " / ".join(meaningful)


def fmt_val(col: str, val) -> str:
    if col == "step":
        return str(val) if val is not None else "-"
    if isinstance(val, (int, float)):
        return f"{val:.2f}"
    return "-"


def print_table(exp_key: str, rows: list[dict]):
    label = short_label(exp_key)
    score_cols = [d for d in DATASETS if any(d in row for row in rows)] + ["Avg"]
    cols = ["step"] + score_cols

    # Format tất cả giá trị thành string trước
    formatted = [[fmt_val(c, row.get(c)) for c in cols] for row in rows]
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
    parser.add_argument("--eval_root", default="./eval_outputs/checkpoints/csd/qwen1.5/csd")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    if not eval_root.exists():
        print(f"[LỖI] Không tìm thấy thư mục: {eval_root}")
        return

    all_data = collect_results(eval_root)
    if not all_data:
        print("Không tìm thấy scores.json/eval.json/eval.log nào có kết quả hợp lệ.")
        return

    for exp_key, rows in all_data.items():
        print_table(exp_key, rows)

    print()


if __name__ == "__main__":
    main()
