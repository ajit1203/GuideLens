import os
import json
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        answerability_loss_weight=0.5
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.answerability_loss_weight = answerability_loss_weight
        self.answer_loss_fn = torch.nn.CrossEntropyLoss()
        self.answerability_loss_fn = torch.nn.CrossEntropyLoss()

    def _step(self, batch, train=True):
        images = batch["image"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        answerable = batch["answerable"].to(self.device)

        if train:
            self.optimizer.zero_grad()

        outputs = self.model(images, input_ids, attention_mask)

        answer_loss = self.answer_loss_fn(outputs["answer_logits"], labels)
        answerability_loss = self.answerability_loss_fn(outputs["answerability_logits"], answerable)
        total_loss = answer_loss + self.answerability_loss_weight * answerability_loss

        if train:
            total_loss.backward()
            self.optimizer.step()

        answer_preds = torch.argmax(outputs["answer_logits"], dim=1)
        answerability_preds = torch.argmax(outputs["answerability_logits"], dim=1)

        answer_acc = (answer_preds == labels).float().mean().item()
        answerability_acc = (answerability_preds == answerable).float().mean().item()

        return total_loss.item(), answer_acc, answerability_acc

    def run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        total_answer_acc = 0.0
        total_answerability_acc = 0.0

        if train:
            iterator = tqdm(loader, desc="Training")
        else:
            iterator = tqdm(loader, desc="Validation")

        with torch.set_grad_enabled(train):
            for batch in iterator:
                loss, answer_acc, answerability_acc = self._step(batch, train=train)
                total_loss += loss
                total_answer_acc += answer_acc
                total_answerability_acc += answerability_acc

        n = len(loader)
        return {
            "loss": total_loss / n,
            "answer_acc": total_answer_acc / n,
            "answerability_acc": total_answerability_acc / n
        }

    def fit(self, train_loader, val_loader, epochs, checkpoint_path, metrics_path, loss_plot_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_answer_acc": [],
            "val_answer_acc": [],
            "train_answerability_acc": [],
            "val_answerability_acc": []
        }

        best_val_loss = float("inf")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_metrics = self.run_epoch(train_loader, train=True)
            val_metrics = self.run_epoch(val_loader, train=False)

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_answer_acc"].append(train_metrics["answer_acc"])
            history["val_answer_acc"].append(val_metrics["answer_acc"])
            history["train_answerability_acc"].append(train_metrics["answerability_acc"])
            history["val_answerability_acc"].append(val_metrics["answerability_acc"])

            print("Train:", train_metrics)
            print("Val  :", val_metrics)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(self.model.state_dict(), checkpoint_path)

        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()

        return history