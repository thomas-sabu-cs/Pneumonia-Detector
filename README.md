Pneumonia Detector (DenseNet121, PyTorch)
A end-to-end project to train and deploy a pneumonia detector from chest X-ray images using PyTorch and DenseNet121. Includes class-imbalance handling, evaluation (AUC/accuracy), optional Grad-CAM explainability, and a simple Gradio app for web-style upload-and-predict. Train locally with data on an external HDD; keep code/checkpoints on SSD for speed. Deploy later to Hugging Face Spaces or Render.

Features
Transfer learning with DenseNet121 (ImageNet-pretrained)
Handles class imbalance (weighted loss)
Patient-safe splits for Kaggle dataset structure
Metrics: AUC-ROC, accuracy, confusion matrix, classification report
Grad-CAM visualization of model attention
Gradio demo app for quick UI testing
Ready path to deploy on Hugging Face Spaces (free-tier)
Not for Clinical Use
This project is for research and educational purposes only and is not intended for clinical use.

Repository Structure
notebooks/
train_kaggle.ipynb — Jupyter notebook to train/evaluate on Kaggle Pneumonia dataset
scripts/
train_kaggle_pneumonia.py — CLI training script equivalent to the notebook
gradcam_vis.py — Generate Grad-CAM heatmaps for a given image and checkpoint
inference.py — Lightweight inference module (PneumoModel) used by the app
app_gradio.py — Local Gradio app for upload-and-predict
checkpoints/ — Saved model weights (git-ignored by default)
requirements.txt — Minimal runtime dependencies
README.md — This file
You can keep dataset(s) on an external HDD. Code, env, and checkpoints should be on your SSD.

Quickstart
1) Environment setup (Windows/macOS/Linux)
It’s okay to have multiple Python versions. We recommend Python 3.11 for best PyTorch compatibility.

Using venv (Windows PowerShell):

If you have Python 3.11 installed:
Create and activate an env:
python3.11 -m venv pneumo311
.\pneumo311\Scripts\activate
If python3.11 is not found, install it from the official site and use its full path:
"C:\Users\YOUR_USER\AppData\Local\Programs\Python\Python311\python.exe" -m venv pneumo311
.\pneumo311\Scripts\activate
Install dependencies:

GPU (CUDA 12.1):
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
CPU-only:
pip install --upgrade pip
pip install torch torchvision torchaudio
Common libs:

pip install scikit-learn pandas numpy matplotlib seaborn tqdm albumentations opencv-python
pip install jupyterlab ipywidgets timm grad-cam gradio
Verify:

python -c "import torch; print(torch.__version__); print('CUDA?', torch.cuda.is_available())"
2) Dataset
Start with Kaggle: “Chest X-Ray Images (Pneumonia)”.

Download and extract to your external HDD. Expected structure:
.../chest_xray/train/NORMAL, .../train/PNEUMONIA
.../chest_xray/val/NORMAL, .../val/PNEUMONIA
.../chest_xray/test/NORMAL, .../test/PNEUMONIA
Example paths:

Windows HDD: E:/medical_data/kaggle_pneumonia/chest_xray/
macOS: /Volumes/External/medical_data/kaggle_pneumonia/chest_xray/
Linux: /media/username/HDD/medical_data/kaggle_pneumonia/chest_xray/
We recommend leaving data on HDD and keeping code/checkpoints on SSD.

Train
Option A: Jupyter notebook

Launch: jupyter lab
Open notebooks/train_kaggle.ipynb
Set DATA_DIR to your HDD path (see above) and run cells top to bottom.
Best checkpoint will be saved to ./checkpoints/densenet121_kaggle_best.pth
Option B: CLI script

Example:
python scripts/train_kaggle_pneumonia.py --data_dir "E:/medical_data/kaggle_pneumonia/chest_xray" --output_dir "./checkpoints" --batch_size 32 --epochs 12 --num_workers 0
Notes:

On Windows, num_workers=0 or 2 is often safer.
Class imbalance is handled via weighted CrossEntropyLoss.
Evaluate
The training notebook/script reports:

Validation AUC-ROC and accuracy per epoch
Final test AUC/accuracy, confusion matrix, and classification report
Save the best checkpoint for deployment.

Grad-CAM (Explainability)
Generate a heatmap to check the model’s focus areas:

python scripts/gradcam_vis.py --checkpoint "./checkpoints/densenet121_kaggle_best.pth" --image "E:/medical_data/kaggle_pneumonia/chest_xray/test/PNEUMONIA/your_image.jpeg" --out "./cam.png"
Inspect that attention is on lung fields rather than borders or markers.

Local Demo (Gradio)
Spin up a simple UI to upload an X-ray and get a prediction:

python scripts/app_gradio.py
Open http://localhost:7860 in your browser
This uses the saved checkpoint and runs locally (CPU or GPU).

Deployment Options
Hugging Face Spaces (free-tier):
Use Gradio Space
Include files:
scripts/inference.py (rename to inference.py at repo root or adjust imports)
scripts/app_gradio.py (rename to app.py)
requirements.txt
Your checkpoint in checkpoints/ (via Git LFS), or modify app to download from a URL at startup
Spaces will auto-build and give you a public URL. You can embed it on your website.
Render / Railway (free-tier CPU):
Switch to FastAPI backend if desired; add server.py, Dockerfile, and a static frontend.
Useful for custom domains; will sleep on idle on free plans.
Tip: For fast CPU inference, you can reduce image size to 320 or use a lighter backbone (MobileNetV3) later.

Scaling Up (NIH ChestX-ray)
Once the pipeline works:

Download NIH ChestX-ray (ChestX-ray14). Data is large (tens of GB).
Switch to a multi-label setup with BCEWithLogitsLoss(pos_weight=...)
Use patient-wise train/val/test splits to avoid leakage
Report per-class AUC and specifically pneumonia AUC
We can add a scripts/train_nih.py when you’re ready.

Requirements
Minimal requirements.txt:

torch
torchvision
pillow
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
albumentations
opencv-python
timm
grad-cam
gradio
If deploying to HF Spaces, keep it minimal for faster builds. You may drop albumentations/opencv if not needed in inference.

Troubleshooting
pip can’t find torch:
Ensure Python 3.11–3.12. Python 3.13 often lacks stable wheels.
For GPU, use the correct index URL (e.g., cu121). For CPU-only, omit the index URL.
Windows slow DataLoader:
Use num_workers=0 or 2.
Low AUC:
Reduce augmentations, check image size, verify class weights, review data quality.
Grad-CAM blank:
Ensure correct target layer for DenseNet (model.features[-1]) and that the model is in eval mode.
License and Attribution
Check the license of the datasets (Kaggle Pneumonia, NIH ChestX-ray) before redistribution.
This code is provided for educational and research purposes. No warranty or clinical guarantees.
Roadmap
 Notebook and script parity for training
 NIH multi-label training script
 FastAPI endpoint with optional Grad-CAM return (base64 PNG)
 HF Spaces deployment guide with auto-download checkpoint
 Model card and dataset notes
