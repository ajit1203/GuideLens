import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class VizWizDataset(Dataset):
    def __init__(
        self,
        csv_path,
        tokenizer_name="distilbert-base-uncased",
        image_size=224,
        max_question_length=24,
        project_root="."
    ):
        self.df = pd.read_csv(csv_path)
        self.project_root = project_root
        self.max_question_length = max_question_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.project_root, row["image_rel_path"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        question = str(row["question"])
        encoded = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt"
        )

        answer_label = int(row["answer_label"])
        answerable = int(row["answerable"])

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(answer_label, dtype=torch.long),
            "answerable": torch.tensor(answerable, dtype=torch.long),
            "question": question,
            "image_name": row["image"],
            "final_answer": row["final_answer"]
        }