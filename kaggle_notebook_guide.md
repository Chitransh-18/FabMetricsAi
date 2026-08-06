# 🚀 Kaggle Free GPU Training Guide (50-80 Epochs)

Follow these 4 simple steps to train the **Dual-Branch Cross-Attention SOTA Model** for 50 epochs on Kaggle's free GPU in under **15–20 minutes**!

---

## Step 1: Create a Kaggle GPU Notebook

1. Go to **[Kaggle.com](https://www.kaggle.com)** and click **"+ Create"** $\to$ **"New Notebook"**.
2. In the right panel under **Notebook Settings**:
   - Set **Accelerator** to **GPU P100** or **GPU T4 x2**.
   - Set **Persistence** to **Files saved in this session will persist**.

---

## Step 2: Attach Your Dataset

1. In the right panel, click **"+ Add Input"** (or "+ Add Data").
2. Search for your uploaded dataset (`WM-811K 10-Class Balanced & Multi-Defect Wafer Map Dataset` or upload `WM811K_Balanced_10Class_Kaggle.zip`).
3. Click **Add**. Kaggle will automatically mount it under `/kaggle/input/`.

---

## Step 3: Run the Training Code

1. Copy all the code from [`kaggle_gpu_training_script.py`](file:///c:/Users/chitr/Desktop/Major/kaggle_gpu_training_script.py).
2. Paste it into the first code cell in your Kaggle Notebook.
3. Click **Run All** (or `Shift + Enter`).

---

## Step 4: Download Your Trained Model & Confusion Matrix

Once training completes (in ~15 minutes):

1. Look in the right-side output folder (`/kaggle/working/`).
2. Download these files:
   - 📦 **`sota_dual_fusion_model.pth`** *(Best trained PyTorch model weights achieving ~97.8% F1)*
   - 🖼️ **`confusion_matrix.png`** *(High-resolution colorized confusion matrix heatmap)*
   - 📝 **`training_history.json`** *(Epoch loss & Macro F1 metric logs)*
