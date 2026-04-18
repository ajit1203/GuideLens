# import os
# import sys
# import json
# import yaml

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import torch
# torch.set_num_threads(1)

# import streamlit as st
# from PIL import Image
# from torchvision import transforms
# from transformers import AutoTokenizer

# from src.models.vqa_model import TrustworthyVQAModel

# st.set_page_config(page_title="Trustworthy Assistive VQA Chat", layout="wide")


# def load_resources():
#     config_path = os.path.join(PROJECT_ROOT, "configs", "base.yml")
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)

#     answer_to_idx_path = os.path.join(PROJECT_ROOT, config["paths"]["answer_to_idx"])
#     idx_to_answer_path = os.path.join(PROJECT_ROOT, config["paths"]["idx_to_answer"])
#     checkpoint_path = os.path.join(PROJECT_ROOT, config["paths"]["checkpoint_path"])

#     if not os.path.exists(answer_to_idx_path):
#         raise FileNotFoundError(f"Missing file: {answer_to_idx_path}")
#     if not os.path.exists(idx_to_answer_path):
#         raise FileNotFoundError(f"Missing file: {idx_to_answer_path}")
#     if not os.path.exists(checkpoint_path):
#         raise FileNotFoundError(f"Missing file: {checkpoint_path}")

#     with open(answer_to_idx_path, "r") as f:
#         answer_to_idx = json.load(f)

#     with open(idx_to_answer_path, "r") as f:
#         idx_to_answer = json.load(f)

#     device = "cpu"

#     st.write("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(
#         config["model"]["text_model_name"],
#         use_fast=False
#     )

#     st.write("Building model...")
#     model = TrustworthyVQAModel(
#         num_answers=len(answer_to_idx),
#         text_model_name=config["model"]["text_model_name"],
#         hidden_dim=config["model"]["hidden_dim"],
#         dropout=config["model"]["dropout"],
#         freeze_vision=config["model"]["freeze_vision"],
#         freeze_text=config["model"]["freeze_text"],
#         use_pretrained_vision=False,
#         local_files_only=False,
#     )

#     st.write("Loading checkpoint...")
#     state_dict = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(state_dict, strict=False)
#     model.to(device)
#     model.eval()

#     transform = transforms.Compose([
#         transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#         ),
#     ])

#     return model, tokenizer, transform, idx_to_answer, device, config


# def ensure_resources():
#     if "resources_loaded" not in st.session_state:
#         st.session_state["resources_loaded"] = False

#     if not st.session_state["resources_loaded"]:
#         model, tokenizer, transform, idx_to_answer, device, config = load_resources()
#         st.session_state["model"] = model
#         st.session_state["tokenizer"] = tokenizer
#         st.session_state["transform"] = transform
#         st.session_state["idx_to_answer"] = idx_to_answer
#         st.session_state["device"] = device
#         st.session_state["config"] = config
#         st.session_state["resources_loaded"] = True


# def predict(image, question):
#     model = st.session_state["model"]
#     tokenizer = st.session_state["tokenizer"]
#     transform = st.session_state["transform"]
#     idx_to_answer = st.session_state["idx_to_answer"]
#     device = st.session_state["device"]
#     config = st.session_state["config"]

#     image = image.convert("RGB")
#     image_tensor = transform(image).unsqueeze(0).to(device)

#     encoded = tokenizer(
#         question,
#         padding="max_length",
#         truncation=True,
#         max_length=config["data"]["max_question_length"],
#         return_tensors="pt",
#     )

#     input_ids = encoded["input_ids"].to(device)
#     attention_mask = encoded["attention_mask"].to(device)

#     with torch.no_grad():
#         outputs = model(image_tensor, input_ids, attention_mask)

#         answer_probs = torch.softmax(outputs["answer_logits"], dim=1)
#         answerability_probs = torch.softmax(outputs["answerability_logits"], dim=1)

#         answer_idx = torch.argmax(answer_probs, dim=1).item()
#         answerability_idx = torch.argmax(answerability_probs, dim=1).item()

#         answer = idx_to_answer.get(str(answer_idx), "unknown")
#         answer_conf = answer_probs[0, answer_idx].item()
#         answerability_conf = answerability_probs[0, answerability_idx].item()

#     return answer, answer_conf, answerability_idx, answerability_conf


# st.title("Trustworthy Assistive VQA Chat")
# st.caption("Upload an image and ask a question about it.")

# left_col, right_col = st.columns([1, 1.2])

# with left_col:
#     st.subheader("Upload Image")
#     uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

#     image = None
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, width="stretch", caption="Uploaded Image")

# with right_col:
#     st.subheader("Chat")
#     question = st.text_input(
#         "Ask a question about the image",
#         placeholder="What is in this image?",
#     )

#     if st.button("Load Model"):
#         try:
#             ensure_resources()
#             st.success("Model loaded successfully.")
#         except Exception as e:
#             st.error(f"Failed to load model: {e}")

#     if st.button("Get Answer"):
#         if image is None:
#             st.warning("Please upload an image first.")
#         elif question.strip() == "":
#             st.warning("Please enter a question.")
#         else:
#             try:
#                 if "resources_loaded" not in st.session_state or not st.session_state["resources_loaded"]:
#                     ensure_resources()

#                 answer, answer_conf, answerability_idx, answerability_conf = predict(image, question)

#                 st.markdown("### Result")
#                 st.write(f"**Predicted answer:** {answer}")
#                 st.write(f"**Answer confidence:** {answer_conf:.3f}")
#                 st.write(f"**Brief description:** The image likely contains **{answer}** or something visually similar.")

#                 if answerability_idx == 0:
#                     st.error(
#                         f"The system suggests this image or question may be difficult to answer reliably "
#                         f"(score: {answerability_conf:.3f})."
#                     )
#                 else:
#                     st.success(
#                         f"The system considers this question reasonably answerable "
#                         f"(score: {answerability_conf:.3f})."
#                     )

#             except Exception as e:
#                 st.error(f"Prediction failed: {e}")



import os
import sys
import tempfile
import yaml
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.qwen_vl_model import QwenVLModel

st.set_page_config(page_title="Trustworthy Assistive VQA Chat", layout="wide")

if "model" not in st.session_state:
    st.session_state["model"] = None


def load_model():
    if st.session_state["model"] is None:
        with open(os.path.join(PROJECT_ROOT, "configs", "qwen.yml"), "r") as f:
            config = yaml.safe_load(f)

        adapter_dir = os.path.join(PROJECT_ROOT, config["train"]["adapter_dir"])
        if not os.path.exists(adapter_dir):
            adapter_dir = None

        st.session_state["model"] = QwenVLModel(
            model_name=config["model"]["model_name"],
            adapter_path=adapter_dir,
            device="cpu",
            local_files_only=False,
        )
        st.session_state["config"] = config


st.title("Trustworthy Assistive VQA Chat")
st.caption("Upload an image and ask a question about it.")

left_col, right_col = st.columns([1, 1.2])

with left_col:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    image = None
    temp_image_path = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width="stretch", caption="Uploaded Image")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_image_path = tmp.name

with right_col:
    question = st.text_input(
        "Ask a question about the image",
        placeholder="What is in this image?"
    )

    if st.button("Load Qwen Model"):
        with st.spinner("Loading Qwen2.5-VL..."):
            load_model()
        st.success("Model loaded successfully.")

    if st.button("Get Answer"):
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            if st.session_state["model"] is None:
                with st.spinner("Loading Qwen2.5-VL..."):
                    load_model()

            with st.spinner("Generating answer..."):
                answer = st.session_state["model"].answer_question(
                    image_path=temp_image_path,
                    question=question,
                    max_new_tokens=st.session_state["config"]["inference"]["max_new_tokens"],
                )

            st.markdown("### Result")
            st.write(answer)

            answer_lower = answer.lower()
            if "unanswerable" in answer_lower or "cannot determine" in answer_lower or "unclear" in answer_lower:
                st.error("The system is not confident enough to answer reliably.")
            else:
                st.success("Answer generated successfully.")