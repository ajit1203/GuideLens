import re
import os
import matplotlib.pyplot as plt

LOG_PATH = "results/mlx_qwen/logs/mlx_train_log.txt"
OUT_PATH = "results/mlx_qwen/plots/mlx_train_loss.png"

pattern = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([0-9.]+)")

iters = []
losses = []

with open(LOG_PATH, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            iters.append(int(match.group(1)))
            losses.append(float(match.group(2)))

if not iters:
    raise ValueError("No training loss values found in the log file.")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(iters, losses, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Train Loss")
plt.title("MLX Qwen LoRA Training Loss")
plt.tight_layout()
plt.savefig(OUT_PATH)
plt.close()

print(f"Saved loss plot to: {OUT_PATH}")