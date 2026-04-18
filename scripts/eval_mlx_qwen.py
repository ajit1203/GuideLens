import os
import sys
import json
import re
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.mlx_qwen_model import MLXQwenVLM

MODEL_PATH = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

# Use None to evaluate the base model only.
# If you want to evaluate the fine-tuned adapter, change this to:
# ADAPTER_PATH = "checkpoints/mlx_qwen25vl_adapter"
ADAPTER_PATH = None

VAL_JSONL = "data/processed/mlx_val.jsonl"

OUT_CSV = "results/mlx_qwen/predictions/mlx_val_predictions.csv"
OUT_JSON = "results/mlx_qwen/metrics/mlx_eval_metrics.json"

MAX_SAMPLES = 100

TRUST_PREFIX = (
    "You are a trustworthy assistive visual question answering system. "
    "Answer the user's question about the image briefly and clearly. "
    "If the image is too unclear or the question cannot be answered from the image, "
    "respond exactly with: unanswerable.\n\nQuestion: "
)


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"<\|.*?\|>", " ", text)      # remove chat tokens
    text = re.sub(r"[^\w\s]", " ", text)        # remove punctuation
    text = " ".join(text.split())               # normalize spaces
    return text


def strip_training_prefix(question: str) -> str:
    question = str(question).strip()
    if question.startswith(TRUST_PREFIX):
        return question[len(TRUST_PREFIX):].strip()
    return question


def is_unanswerable(text: str) -> bool:
    text = normalize_text(text)
    return text in {
        "unanswerable",
        "cannot determine",
        "cant determine",
        "unclear",
        "not answerable",
    }


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    adapter_path = ADAPTER_PATH if (ADAPTER_PATH and os.path.exists(ADAPTER_PATH)) else None

    model = MLXQwenVLM(
        model_path=MODEL_PATH,
        adapter_path=adapter_path,
    )

    rows = []
    total = 0
    exact_match = 0
    unanswerable_total = 0
    unanswerable_correct = 0

    with open(VAL_JSONL, "r") as f:
        for i, line in enumerate(f):
            if i >= MAX_SAMPLES:
                break

            sample = json.loads(line)
            image_path = sample["image"]
            stored_question = sample["question"]
            raw_question = strip_training_prefix(stored_question)
            ground_truth = sample["answer"]

            prediction = model.answer_question(
                image_path=image_path,
                question=raw_question,
                max_tokens=24,
            )

            gt_norm = normalize_text(ground_truth)
            pred_norm = normalize_text(prediction)

            total += 1
            match = int(gt_norm == pred_norm)
            exact_match += match

            if is_unanswerable(ground_truth):
                unanswerable_total += 1
                if is_unanswerable(prediction):
                    unanswerable_correct += 1

            rows.append({
                "image": image_path,
                "question": raw_question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "ground_truth_normalized": gt_norm,
                "prediction_normalized": pred_norm,
                "exact_match": match,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    metrics = {
        "num_samples": total,
        "exact_match_accuracy": exact_match / total if total > 0 else 0.0,
        "unanswerable_recall": unanswerable_correct / unanswerable_total if unanswerable_total > 0 else 0.0,
        "adapter_used": adapter_path if adapter_path else "base_model_only",
    }

    with open(OUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved predictions to:", OUT_CSV)
    print("Saved metrics to:", OUT_JSON)
    print(metrics)


if __name__ == "__main__":
    main()