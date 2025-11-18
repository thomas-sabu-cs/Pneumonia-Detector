import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from pathlib import Path

import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class CenterLungCrop(object):
    """
    Remove fixed margins at top/bottom/sides where borders/text often live.
    Tune fractions after visual inspection.
    """
    def __init__(self, top_frac=0.08, bottom_frac=0.08, side_frac=0.05):
        self.top_frac = top_frac
        self.bottom_frac = bottom_frac
        self.side_frac = side_frac

    def __call__(self, img: Image.Image):
        w, h = img.size
        left   = int(self.side_frac * w)
        right  = int((1 - self.side_frac) * w)
        top    = int(self.top_frac * h)
        bottom = int((1 - self.bottom_frac) * h)
        return img.crop((left, top, right, bottom))

# ------------------------------------------------
# Config
# ------------------------------------------------
CKPT_PATH = Path("checkpoints/densenet121_kaggle_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ------------------------------------------------
# Model + transforms loader
# ------------------------------------------------
def load_model_and_transforms(ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    img_size = ckpt.get("img_size", 384)
    norm_cfg = ckpt.get("normalization", {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    })
    mean = norm_cfg["mean"]
    std = norm_cfg["std"]

    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(DEVICE)
    model.eval()

    infer_tfms = transforms.Compose([
        CenterLungCrop(top_frac=0.08, bottom_frac=0.08, side_frac=0.05),
        transforms.Resize(int(img_size * 1.2)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, infer_tfms, img_size


model, infer_tfms, IMG_SIZE = load_model_and_transforms(CKPT_PATH)

# Grad-CAM setup
target_layer = model.features[-2]
cam = GradCAM(model=model, target_layers=[target_layer])


# ------------------------------------------------
# Core prediction + Grad-CAM
# ------------------------------------------------
def predict_pneumonia(img: Image.Image, threshold: float = 0.9):
    if img is None:
        return "No image", 0.0, "Please upload an image.", None

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Prepare tensor WITH grad for Grad-CAM
    x = infer_tfms(img).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    # Forward pass (no torch.no_grad here!)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    normal_prob = float(probs[0].item())
    pneu_prob = float(probs[1].item())

    if pneu_prob >= threshold:
        pred_class = "PNEUMONIA"
        confidence = pneu_prob
        target_class = 1
    else:
        pred_class = "NORMAL"
        confidence = normal_prob
        target_class = 0

    # ---- Grad-CAM ----
    orig_resized = img.resize((IMG_SIZE, IMG_SIZE))
    rgb_float = np.array(orig_resized).astype(np.float32) / 255.0

    grayscale_cam = cam(
        input_tensor=x,
        targets=[ClassifierOutputTarget(target_class)]
    )[0]  # [H, W]

    # Keep only positive contributions
    grayscale_cam = np.maximum(grayscale_cam, 0)
    grayscale_cam = grayscale_cam / (grayscale_cam.max() + 1e-8)

    # Threshold to suppress weak, diffuse activations (tune 0.3 as needed)
    threshold_cam = np.where(grayscale_cam > 0.3, grayscale_cam, 0.0)

    cam_image = show_cam_on_image(rgb_float, threshold_cam, use_rgb=True)
    cam_pil = Image.fromarray(cam_image)

    if pred_class == "PNEUMONIA":
        message = (
            f"Prediction: PNEUMONIA (threshold={threshold:.2f}).\n\n"
            f"Estimated probability of pneumonia: {pneu_prob:.3f}\n"
            f"Estimated probability of normal:   {normal_prob:.3f}\n\n"
            "Note: This model is a research/educational tool only and "
            "must not be used for real medical diagnosis."
        )
    else:
        message = (
            f"Prediction: NORMAL (threshold={threshold:.2f}).\n\n"
            f"Estimated probability of normal:   {normal_prob:.3f}\n"
            f"Estimated probability of pneumonia: {pneu_prob:.3f}\n\n"
            "Note: This model is a research/educational tool only and "
            "must not be used for real medical diagnosis."
        )

    return pred_class, confidence, message, cam_pil


# ------------------------------------------------
# Gradio UI
# ------------------------------------------------
title = "Chest X-Ray Pneumonia Detector (DenseNet121 + Grad-CAM)"
description = (
    "Upload a chest X-ray image. The model will classify it as NORMAL or PNEUMONIA "
    "and show a Grad-CAM heatmap of the regions influencing its decision.\n\n"
    "**Important:** This is a research / educational demo only and must not be used "
    "for clinical decisions."
)

demo = gr.Interface(
    fn=predict_pneumonia,
    inputs=[
        gr.Image(type="pil", label="Chest X-ray"),
        gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Pneumonia threshold"),
    ],
    outputs=[
        gr.Label(label="Predicted Class"),
        gr.Number(label="Confidence (probability of predicted class)"),
        gr.Textbox(label="Explanation", lines=6),
        gr.Image(type="pil", label="Grad-CAM Heatmap"),
    ],
    title=title,
    description=description,
)


if __name__ == "__main__":
    demo.launch()