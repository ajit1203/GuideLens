import os
import json
import random
from collections import Counter
import pandas as pd


def clean_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.lower().strip().split())


def extract_answers(answer_list):
    extracted = []
    for ans in answer_list:
        if isinstance(ans, dict) and "answer" in ans:
            extracted.append(clean_text(ans["answer"]))
        elif isinstance(ans, str):
            extracted.append(clean_text(ans))
    return extracted


def majority_answer(answer_list):
    answers = extract_answers(answer_list)
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_rows(data, split_name):
    rows = []

    for item in data:
        image_name = item.get("image")
        question = clean_text(item.get("question", ""))
        final_answer = majority_answer(item.get("answers", []))
        answerable = int(item.get("answerable", 0))

        if image_name is None or question == "" or final_answer is None:
            continue

        # Important consistency fix:
        # if the majority answer is "unanswerable", force answerable to 0
        if final_answer == "unanswerable":
            answerable = 0

        image_rel_path = f"data/raw/vizwiz/{split_name}/{image_name}"

        rows.append({
            "image": image_name,
            "image_rel_path": image_rel_path,
            "question": question,
            "final_answer": final_answer,
            "answerable": answerable,
            "split": split_name
        })

    return rows


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    random.seed(42)

    train_json = "data/raw/vizwiz/annotations/train.json"
    val_json = "data/raw/vizwiz/annotations/val.json"

    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Load raw annotation files
    train_data = load_json(train_json)
    val_data = load_json(val_json)

    # Build cleaned rows
    train_rows = build_rows(train_data, "train")
    val_rows = build_rows(val_data, "val")

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    print("Initial cleaned train rows:", len(train_df))
    print("Initial cleaned val rows:", len(val_df))

    # Build top-K answer vocabulary using full training data only
    top_k = 300
    answer_counts = Counter(train_df["final_answer"].tolist())
    top_answers = [ans for ans, _ in answer_counts.most_common(top_k)]

    answer_to_idx = {ans: idx for idx, ans in enumerate(top_answers)}
    idx_to_answer = {str(idx): ans for ans, idx in answer_to_idx.items()}

    # Keep only rows whose final answer is in top-K
    train_df = train_df[train_df["final_answer"].isin(top_answers)].copy()
    val_df = val_df[val_df["final_answer"].isin(top_answers)].copy()

    # Map answer label
    train_df["answer_label"] = train_df["final_answer"].map(answer_to_idx)
    val_df["answer_label"] = val_df["final_answer"].map(answer_to_idx)

    # Save full processed files
    train_full_path = os.path.join(processed_dir, "train_full.csv")
    val_full_path = os.path.join(processed_dir, "val_full.csv")

    train_df.to_csv(train_full_path, index=False)
    val_df.to_csv(val_full_path, index=False)

    # Create smaller subsets for local training
    train_limit = 3000
    val_limit = 600

    train_subset_df = train_df.copy()
    val_subset_df = val_df.copy()

    if len(train_subset_df) > train_limit:
        train_subset_df = train_subset_df.sample(train_limit, random_state=42).reset_index(drop=True)

    if len(val_subset_df) > val_limit:
        val_subset_df = val_subset_df.sample(val_limit, random_state=42).reset_index(drop=True)

    # Save subset files
    train_subset_path = os.path.join(processed_dir, "train_subset.csv")
    val_subset_path = os.path.join(processed_dir, "val_subset.csv")

    train_subset_df.to_csv(train_subset_path, index=False)
    val_subset_df.to_csv(val_subset_path, index=False)

    # Save vocabulary mappings
    answer_to_idx_path = os.path.join(processed_dir, "answer_to_idx.json")
    idx_to_answer_path = os.path.join(processed_dir, "idx_to_answer.json")

    save_json(answer_to_idx, answer_to_idx_path)
    save_json(idx_to_answer, idx_to_answer_path)

    # Print summary
    print("\nProcessed files created successfully.")
    print(f"Top-K answer vocabulary size: {len(answer_to_idx)}")

    print("\nSaved full processed files:")
    print(f"  {train_full_path} -> {len(train_df)} rows")
    print(f"  {val_full_path}   -> {len(val_df)} rows")

    print("\nSaved subset files:")
    print(f"  {train_subset_path} -> {len(train_subset_df)} rows")
    print(f"  {val_subset_path}   -> {len(val_subset_df)} rows")

    print("\nSample answer distribution:")
    for ans, count in answer_counts.most_common(10):
        print(f"  {ans}: {count}")

    print("\nAnswerable distribution in subset train:")
    print(train_subset_df["answerable"].value_counts().to_dict())

    print("\nAnswerable distribution in subset val:")
    print(val_subset_df["answerable"].value_counts().to_dict())


if __name__ == "__main__":
    main()