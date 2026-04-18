import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "data/processed/train_full.csv"
OUT_PATH = "figures/top_answers.png"
TOP_K = 15

df = pd.read_csv(INPUT_CSV)

answer_counts = df["final_answer"].value_counts().head(TOP_K).sort_values()

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

plt.figure(figsize=(10, 6))
plt.barh(answer_counts.index, answer_counts.values)
plt.xlabel("Count")
plt.ylabel("Answer")
plt.title(f"Top {TOP_K} Most Common Answers in Training Set")
plt.tight_layout()
plt.savefig(OUT_PATH)
plt.close()

print(f"Saved plot to: {OUT_PATH}")