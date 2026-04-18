import os
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


class MLXQwenVLM:
    def __init__(self, model_path: str, adapter_path: str | None = None):
        self.model_path = model_path
        self.adapter_path = adapter_path

        if adapter_path and os.path.exists(adapter_path):
            self.model, self.processor = load(model_path, adapter_path=adapter_path)
        else:
            self.model, self.processor = load(model_path)

        try:
            self.config = self.model.config
        except Exception:
            self.config = load_config(model_path)

    def _clean_output(self, text: str) -> str:
        text = str(text).strip()

        # remove common chat artifacts
        text = text.replace("<|im_start|>", " ")
        text = text.replace("<|im_end|>", " ")
        text = text.replace("<image>", " ")

        # collapse whitespace
        text = " ".join(text.split())

        # remove common leading role labels
        for prefix in ["assistant", "Assistant", "user", "User", "system", "System"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip(" :,-")

        return text.strip()

    def answer_question(self, image_path: str, question: str, max_tokens: int = 24) -> str:
        prompt = (
            "You are a trustworthy assistive visual question answering system. "
            "Answer the user's question about the image briefly and clearly. "
            "If the image is too unclear or the question cannot be answered from the image, "
            "respond exactly with: unanswerable.\n\n"
            f"Question: {question}"
        )

        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=1,
        )

        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            [image_path],
            max_tokens=max_tokens,
            verbose=False,
        )

        if hasattr(result, "text"):
            return self._clean_output(result.text)

        return self._clean_output(str(result))