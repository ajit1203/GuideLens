import os
import json
import pandas as pd


TRUST_PROMPT_PREFIX = (
    "You are a trustworthy assistive visual question answering system. "
    "Answer the user's question about the image briefly and clearly. "
    "If the image is too unclear or the question cannot be answered from the image, "
    "respond exactly with: unanswerable.\n\nQuestion: "
)


def normalize_answer(final_answer: str, answerable: int) -> str:
    final_answer = str(final_answer).strip().lower()
    if answerable == 0 or final_answer == "unanswerable":
        return "unanswerable"
    return final_answer


def convert_csv_to_jsonl(input_csv: str, output_jsonl: str, limit: int | None = None):
    df = pd.read_csv(input_csv)
    rows_written = 0

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, "w") as f:
        for _, row in df.iterrows():
            image_path = row["image_rel_path"]
            question = str(row["question"]).strip()
            answerable = int(row["answerable"])
            final_answer = str(row["final_answer"]).strip()

            if not os.path.exists(image_path):
                continue
            if not question:
                continue

            record = {
                "image": image_path,
                "question": TRUST_PROMPT_PREFIX + question,
                "answer": normalize_answer(final_answer, answerable),
            }

            f.write(json.dumps(record) + "\n")
            rows_written += 1

            if limit is not None and rows_written >= limit:
                break

    print(f"Saved {rows_written} examples to {output_jsonl}")


def main():
    # Start small on Mac
    convert_csv_to_jsonl(
        input_csv="data/processed/train_full.csv",
        output_jsonl="data/processed/mlx_train.jsonl",
        limit=1500,
    )

    convert_csv_to_jsonl(
        input_csv="data/processed/val_full.csv",
        output_jsonl="data/processed/mlx_val.jsonl",
        limit=300,
    )


if __name__ == "__main__":
    main()