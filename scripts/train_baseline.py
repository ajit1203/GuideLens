import os
import json
import yaml
import torch
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim import Adam

from src.data.dataset import VizWizDataset
from src.models.vqa_model import TrustworthyVQAModel
from src.training.trainer import Trainer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_sample_predictions(model, val_loader, idx_to_answer, device, output_csv):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(images, input_ids, attention_mask)

            answer_preds = torch.argmax(outputs["answer_logits"], dim=1).cpu().tolist()
            answerability_preds = torch.argmax(outputs["answerability_logits"], dim=1).cpu().tolist()

            for i in range(len(answer_preds)):
                pred_idx = answer_preds[i]
                predicted_answer = idx_to_answer.get(str(pred_idx), "unknown")

                rows.append({
                    "image_name": batch["image_name"][i],
                    "question": batch["question"][i],
                    "ground_truth_answer": batch["final_answer"][i],
                    "predicted_answer": predicted_answer,
                    "predicted_answerability": int(answerability_preds[i])
                })

            if len(rows) >= 20:
                break

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(rows[:20]).to_csv(output_csv, index=False)


def main():
    config_path = "configs/base.yml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)
    print("Current working directory:", os.getcwd())

    train_csv = config["data"]["train_csv"]
    val_csv = config["data"]["val_csv"]

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    train_dataset = VizWizDataset(
        csv_path=train_csv,
        tokenizer_name=config["model"]["text_model_name"],
        image_size=config["data"]["image_size"],
        max_question_length=config["data"]["max_question_length"],
        project_root="."
    )

    val_dataset = VizWizDataset(
        csv_path=val_csv,
        tokenizer_name=config["model"]["text_model_name"],
        image_size=config["data"]["image_size"],
        max_question_length=config["data"]["max_question_length"],
        project_root="."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False
    )

    answer_to_idx_path = config["paths"]["answer_to_idx"]
    idx_to_answer_path = config["paths"]["idx_to_answer"]

    if not os.path.exists(answer_to_idx_path):
        raise FileNotFoundError(f"answer_to_idx.json not found: {answer_to_idx_path}")
    if not os.path.exists(idx_to_answer_path):
        raise FileNotFoundError(f"idx_to_answer.json not found: {idx_to_answer_path}")

    with open(answer_to_idx_path, "r") as f:
        answer_to_idx = json.load(f)

    with open(idx_to_answer_path, "r") as f:
        idx_to_answer = json.load(f)

    model = TrustworthyVQAModel(
        num_answers=len(answer_to_idx),
        text_model_name=config["model"]["text_model_name"],
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"],
        freeze_vision=config["model"]["freeze_vision"],
        freeze_text=config["model"]["freeze_text"]
    ).to(device)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["train"]["lr"]
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        answerability_loss_weight=config["train"]["answerability_loss_weight"]
    )

    checkpoint_path = config["paths"]["checkpoint_path"]
    metrics_path = config["paths"]["metrics_path"]
    loss_plot_path = config["paths"]["loss_plot_path"]
    predictions_path = config["paths"]["predictions_path"]

    print("Checkpoint path:", checkpoint_path)
    print("Metrics path:", metrics_path)
    print("Loss plot path:", loss_plot_path)
    print("Predictions path:", predictions_path)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["train"]["epochs"],
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        loss_plot_path=loss_plot_path
    )

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    else:
        raise FileNotFoundError(f"Checkpoint was not created: {checkpoint_path}")

    save_sample_predictions(
        model=model,
        val_loader=val_loader,
        idx_to_answer=idx_to_answer,
        device=device,
        output_csv=predictions_path
    )

    print("\nTraining completed successfully.")
    print("Saved files:")
    print(" - Checkpoint :", checkpoint_path)
    print(" - Metrics    :", metrics_path)
    print(" - Loss plot  :", loss_plot_path)
    print(" - Predictions:", predictions_path)


if __name__ == "__main__":
    main()