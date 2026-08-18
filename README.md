# Rayleon Image Restoration

**Team Rayleon**

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**

---

## About the Project

Semiconductor inspection and metrology depend on high-quality images to identify small defects and critical structures. In practical imaging conditions, images can be affected by **Gaussian noise, speckle noise, and limited spatial resolution**, making fine features harder to distinguish.

Team Rayleon's solution uses a **two-stage transformer-based restoration pipeline** to first restore the degraded image and then reconstruct its high-resolution details.

```text
Degraded / Noisy LR Image
          │
          ▼
      Restormer
   Noise Restoration
          │
          ▼
   Clean Restored LR
          │
          ▼
        HAT-L
  Super-Resolution
          │
          ▼
      HR Output
```

The main idea is to let each model focus on the task it is best suited for: **Restormer handles degradation and noise removal, while HAT-L focuses on resolution and high-frequency detail recovery.**

---

### Stage 1 — Image Restoration

Restormer is used to suppress degradation, with the training pipeline targeting **Gaussian and speckle noise** while preserving important structural information.

### Stage 2 — Super-Resolution

The restored low-resolution image is passed to HAT-L, which reconstructs high-frequency information and generates the final high-resolution image.

The models are trained as separate supervised stages and then connected to form the complete restoration pipeline.

---

## Training Methodology

The training workflow is based on paired degraded and ground-truth images with consistent spatial preprocessing.

### Data Preparation

The training pipeline uses paired spatial transformations so that the degraded image and its corresponding ground truth remain aligned.

The augmentation strategy includes:

* **Random cropping**
* **Horizontal and vertical flipping**
* **90° rotations**
* **Task-aware degradation augmentation**

### Task-Aware Compound Degradation

A key part of the approach is the controlled combination of:

* **Gaussian noise**
* **Speckle noise**
* **Resolution degradation**

at different strengths to represent a wider range of image-acquisition conditions.

### Optimization

A **combined L1 + L2 reconstruction loss** is used during training to reduce pixel-level differences between predicted and ground-truth images.

---

## Why a Two-Stage Pipeline?

Instead of asking a single model to solve every type of degradation at once, the pipeline separates restoration into two specialized tasks.

**Restormer → noise and degradation removal**

**HAT-L → resolution and detail recovery**

This separation provides a more modular pipeline and allows the two stages to be optimized independently before being connected for final inference.

---

## Expected Impact

The objective is to produce **cleaner and higher-resolution inspection images** in which fine structures and potential defects are easier to observe.

The restoration process is intended to reduce the effect of image degradation on downstream image-based inspection and analysis.

The final output is therefore designed to preserve structural information while improving image quality and spatial detail.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/rehanalam-15/Rayleon-ImageRestoration.git
cd Rayleon-ImageRestoration
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

The dependency list is being validated against the complete Restormer + HAT-L inference environment. Exact package versions will be pinned in the final submission after end-to-end testing.

---

# Hardware & Development Environment

The solution is designed for execution on an **NVIDIA GPU**.

Development and experimentation have used:

* **Python**
* **PyTorch**
* **Restormer**
* **HAT-L**
* **NumPy**
* **OpenCV / PIL**
* **Conda**
* **Jupyter Notebook**
* **Google Colab GPU**
* **VS Code**
* **Git / GitHub**

Local development has also been performed on a system using an **Intel Core Ultra 9 185H processor, 32 GB RAM, and RHEL 10**, while Google Colab GPU acceleration has been used for model training and experimentation.

---

# Offline Execution

Once the required dependencies and model weights are available locally, the solution is designed to run:

**without internet access, API keys, additional model downloads, user interaction, or manual code modification.**

The intended user interaction is limited to supplying the input and output directories:

```bash
python run.py <input-dir> <output-dir>
```

---

# Project Status

The repository is currently being prepared for the final submission.

The development process is being carried out in stages:

```text
Model Preparation
      ↓
Restormer Training / Fine-Tuning
      ↓
HAT-L Training / Fine-Tuning
      ↓
Standalone Model Validation
      ↓
Restormer + HAT-L Integration
      ↓
Final run.py
      ↓
Output Validation
      ↓
Final Model Weights
      ↓
Final Submission
```

The pretrained checkpoints may be used during development and testing. They will be replaced with the **final Team Rayleon trained weights** before submission.

---

# Research Foundation

The proposed approach is based on transformer-based image restoration and super-resolution models, with the restoration and super-resolution stages optimized separately and then combined into a single inference pipeline.

The project focuses on restoring degraded inspection imagery while retaining **structural and defect-related information** that may be lost because of noise or limited spatial resolution.

---

# References

The final repository will include references to the original Restormer and HAT research implementations and any other datasets, tools, or papers used during development.

---

## Team Rayleon
