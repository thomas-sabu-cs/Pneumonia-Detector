# Pneumonia Detector (DenseNet121 · PyTorch)

Build, train, and demo a pneumonia detector from chest X-ray images using PyTorch and DenseNet121. The project includes class-imbalance handling, robust evaluation, Grad-CAM explainability, and a Gradio upload-and-predict app. Train locally with data on an external HDD; keep code/checkpoints on SSD for speed. Deploy later to Hugging Face Spaces or Render.

> Disclaimer: This project is for research and educational purposes only and is not intended for clinical use.

## Features
- Transfer learning with DenseNet121 (ImageNet-pretrained)
- Class imbalance handling (weighted loss)
- Evaluation: AUC-ROC, accuracy, confusion matrix, classification report
- Grad-CAM visualization for model explainability
- Gradio demo app for quick local UI
- Ready-to-deploy path for Hugging Face Spaces (free tier)

## Repository Structure
```
.
├─ notebooks/
│  └─ train_kaggle.ipynb            # Jupyter notebook to train/evaluate on Kaggle Pneumonia
├─ scripts/
│  ├─ train_kaggle_pneumonia.py     # CLI training script
│  ├─ gradcam_vis.py                # Generate Grad-CAM heatmaps
│  ├─ inference.py                  # Reusable inference module (PneumoModel)
│  └─ app_gradio.py                 # Local Gradio app (upload-and-predict)
├─ checkpoints/                     # Saved model weights (git-ignored)
├─ requirements.txt                 # Runtime dependencies
└─ README.md
```

Tip: Keep datasets on an external HDD; keep code, venv, and checkpoints on SSD.

## Quickstart

### 1) Environment setup (Windows/macOS/Linux)
It’s fine to have multiple Python versions. PyTorch is best supported on Python 3.11–3.12.

- Create a virtual environment (Windows PowerShell):
  - If Python 3.11 is installed:
    ```
    python3.11 -m venv pneumo311
    .\pneumo311\Scripts\activate
    ```
  - If not found, install Python 3.11 and use its full path:
    ```
    "C:\Users\YOUR_USER\AppData\Local\Programs\Python\Python311\python.exe" -m venv pneumo311
    .\pneumo311\Scripts\activate
    ```

- Install PyTorch:
  - GPU (CUDA 12.1):
    ```
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
  - CPU-only:
    ```
    pip install --upgrade pip
    pip install torch torchvision torchaudio
    ```

- Common libraries:
  ```
  pip install scikit-learn pandas numpy matplotlib seaborn tqdm albumentations opencv-python
  pip install jupyterlab ipywidgets timm grad-cam gradio
  ```

- Verify:
  ```
  python -c "import torch; print(torch.__version__); print('CUDA?', torch.cuda.is_available())"
  ```

### 2) Dataset (Kaggle Pneumonia)
Download “Chest X-Ray Images (Pneumonia)” from Kaggle and extract to your external HDD.

Expected structure:
```
chest_xray/
├─ train/
│  ├─ NORMAL/
│  └─ PNEUMONIA/
├─ val/
│  ├─ NORMAL/
│  └─ PNEUMONIA/
└─ test/
   ├─ NORMAL/
   └─ PNEUMONIA/
```

Example paths:
- Windows: `E:/medical_data/kaggle_pneumonia/chest_xray/`
- macOS: `/Volumes/External/medical_data/kaggle_pneumonia/chest_xray/`
- Linux: `/media/username/HDD/medical_data/kaggle_pneumonia/chest_xray/`

## Training

### Option A: Jupyter notebook
1) Launch:
```
jupyter lab
```
2) Open `notebooks/train_kaggle.ipynb`
3) Set `DATA_DIR` to your HDD path and run cells top to bottom.
4) Best checkpoint is saved to:
```
./checkpoints/densenet121_kaggle_best.pth
```

### Option B: CLI script
Run:
```
python scripts/train_kaggle_pneumonia.py --data_dir "E:/medical_data/kaggle_pneumonia/chest_xray" --output_dir "./checkpoints" --batch_size 32 --epochs 12 --num_workers 0
```
Notes:
- On Windows, `num_workers=0` or `2` is often safer.
- Class imbalance is handled via weighted CrossEntropyLoss.

## Evaluation
The notebook/script reports:
- Validation AUC-ROC and accuracy per epoch
- Final test AUC/accuracy
- Confusion matrix and classification report

Save the best checkpoint for deployment.

## Grad-CAM (Explainability)
Generate a heatmap to check the model’s focus:
```
python scripts/gradcam_vis.py --checkpoint "./checkpoints/densenet121_kaggle_best.pth" --image "E:/medical_data/kaggle_pneumonia/chest_xray/test/PNEUMONIA/your_image.jpeg" --out "./cam.png"
```
Inspect that attention covers the lungs rather than borders or text markers.

## Local Demo (Gradio)
Run a simple UI to upload an X-ray and get a prediction:
```
python scripts/app_gradio.py
```
Open http://localhost:7860 in your browser.

## Deployment

### Hugging Face Spaces (free-tier)
- Create a Gradio Space.
- Include:
  - `inference.py` (at repo root or adjust imports)
  - `app_gradio.py` renamed to `app.py`
  - `requirements.txt`
  - Your checkpoint in `checkpoints/` (via Git LFS), or modify the app to download from a URL on startup
- Spaces will auto-build and give you a public URL (can be embedded on your site).

### Render / Railway (free-tier CPU)
- Optionally use a FastAPI backend (`server.py`) + static frontend.
- Good for custom domains; free plans may sleep on idle.

Tips for faster CPU inference:
- Reduce image size to 320
- Consider a lighter backbone (e.g., MobileNetV3) later

## Scaling Up (NIH ChestX-ray)
After Kaggle works:
- Download NIH ChestX-ray (ChestX-ray14) to HDD (large: tens of GB)
- Switch to multi-label training with `BCEWithLogitsLoss(pos_weight=...)`
- Use patient-wise splits to avoid leakage
- Report per-class AUC, especially pneumonia

A `scripts/train_nih.py` can be added later.

## Requirements
Minimal `requirements.txt`:
```
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
```
For HF Spaces, keep requirements minimal for faster builds (you can drop albumentations/opencv if not used in inference).

## Troubleshooting
- pip can’t find torch:
  - Ensure Python 3.11–3.12. Python 3.13 may lack stable wheels.
  - For GPU, use the correct index URL (e.g., `cu121`). For CPU-only, omit it.
- Windows slow DataLoader:
  - Use `num_workers=0` or `2`.
- Low AUC:
  - Tweak augmentations, verify class weights, try image size 320–384, review data quality.
- Grad-CAM blank:
  - Ensure the correct target layer for DenseNet (`model.features[-1]`) and model in eval mode.

## License and Attribution
- Check dataset licenses (Kaggle Pneumonia, NIH ChestX-ray) before redistribution.
- Code provided for educational and research purposes without warranty; not for clinical use.

## Roadmap
- [ ] Notebook and script parity for training
- [ ] NIH multi-label training script
- [ ] FastAPI endpoint with optional Grad-CAM (base64 PNG) return
- [ ] HF Spaces deployment guide with auto-download checkpoint
- [ ] Model card and dataset notes
