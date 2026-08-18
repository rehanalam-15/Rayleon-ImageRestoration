"""
Team Rayleon
AI-Based Restoration of Degraded Images

Official submission entry point.

Usage:
    python run.py <input-dir> <output-dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

MODELS_DIR = ROOT_DIR / "models"

RESTORMER_DIR = MODELS_DIR / "restormer"
HAT_DIR = MODELS_DIR / "hat"

RESTORMER_WEIGHTS = RESTORMER_DIR / "Rayleon_Restormer.pth"
HAT_WEIGHTS = HAT_DIR / "Rayleon_HAT-L.pth"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Team Rayleon image restoration pipeline."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing input .npy files."
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where restored outputs will be written."
    )

    return parser.parse_args()


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(array: np.ndarray, filename: str) -> np.ndarray:
    """
    Validate and normalize an input grayscale image.

    The exact preprocessing required by Restormer will be
    finalized after we inspect your trained/pretrained model.
    """

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{filename}: input is not a NumPy array.")

    if array.ndim not in (2, 3):
        raise ValueError(
            f"{filename}: expected shape (H,W) or (H,W,1), "
            f"got {array.shape}"
        )

    if array.ndim == 3 and array.shape[-1] != 1:
        raise ValueError(
            f"{filename}: expected grayscale input with shape "
            f"(H,W) or (H,W,1), got {array.shape}"
        )

    array = np.asarray(array, dtype=np.float32)

    if not np.isfinite(array).all():
        raise ValueError(f"{filename}: input contains NaN or Inf.")

    # Temporary safety normalization.
    # We will replace this with the exact competition/model
    # preprocessing after inspecting your pipeline.
    array = np.clip(array, 0.0, 1.0)

    if array.ndim == 3:
        array = array[..., 0]

    return array


# ============================================================
# MODEL LOADING
# ============================================================

def load_restormer():
    """
    Load Restormer architecture + checkpoint.

    This function will be completed using your exact
    Restormer implementation.
    """

    if not RESTORMER_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Restormer checkpoint not found:\n"
            f"{RESTORMER_WEIGHTS}"
        )

    # TODO:
    # 1. Import your exact Restormer architecture
    # 2. Construct the model
    # 3. Load Rayleon_Restormer.pth
    # 4. Move to DEVICE
    # 5. model.eval()

    raise NotImplementedError(
        "Restormer loader will be implemented after "
        "the exact Restormer model files/checkpoint format "
        "are inspected."
    )


def load_hat():
    """
    Load HAT-L architecture + checkpoint.

    This function will be completed using your exact
    HAT-L implementation.
    """

    if not HAT_WEIGHTS.exists():
        raise FileNotFoundError(
            f"HAT-L checkpoint not found:\n"
            f"{HAT_WEIGHTS}"
        )

    # TODO:
    # 1. Import the exact HAT-L architecture
    # 2. Construct the model
    # 3. Load Rayleon_HAT-L.pth
    # 4. Move to DEVICE
    # 5. model.eval()

    raise NotImplementedError(
        "HAT-L loader will be implemented after "
        "the exact HAT-L model files/checkpoint format "
        "are inspected."
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

@torch.inference_mode()
def run_restormer(model, image: np.ndarray) -> np.ndarray:
    """
    Run Restormer on one image.

    Exact tensor conversion and preprocessing will be
    finalized after inspecting your current inference code.
    """

    tensor = torch.from_numpy(image).float()

    # H,W -> 1,1,H,W
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    output = output.squeeze().detach().cpu().numpy()

    return output.astype(np.float32)


@torch.inference_mode()
def run_hat(model, image: np.ndarray) -> np.ndarray:
    """
    Run HAT-L on one Restormer output.
    """

    tensor = torch.from_numpy(image).float()

    # H,W -> 1,1,H,W
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    output = output.squeeze().detach().cpu().numpy()

    return output.astype(np.float32)


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_output(
    array: np.ndarray,
    filename: str,
) -> np.ndarray:

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{filename}: output is not a NumPy array.")

    if array.ndim not in (2, 3):
        raise ValueError(
            f"{filename}: output must have shape (H,W) "
            f"or (H,W,1), got {array.shape}"
        )

    if array.ndim == 3 and array.shape[-1] != 1:
        raise ValueError(
            f"{filename}: output is not grayscale: {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{filename}: output contains NaN or Inf."
        )

    array = np.asarray(array, dtype=np.float32)

    # Required competition range.
    array = np.clip(array, 0.0, 1.0)

    if array.ndim == 3:
        array = array[..., 0]

    return array


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    image: np.ndarray,
    npy_path: Path,
    png_path: Path,
):
    """
    Save normalized .npy output and visualization PNG.
    """

    image = np.asarray(image, dtype=np.float32)

    # Numerical output required by the competition.
    np.save(npy_path, image)

    # Visualization output.
    image_uint8 = np.round(image * 255.0).astype(np.uint8)

    Image.fromarray(image_uint8, mode="L").save(png_path)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist:\n{input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"ERROR: Input path is not a directory:\n{input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate PNG visualization folder.
    png_dir = output_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.npy"))

    if not input_files:
        print(f"ERROR: No .npy files found in:\n{input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("TEAM RAYLEON IMAGE RESTORATION")
    print("=" * 60)
    print(f"Device       : {DEVICE}")
    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Input files  : {len(input_files)}")
    print("=" * 60)

    print("\nLoading models...")

    restormer = load_restormer()
    hat = load_hat()

    print("Models loaded successfully.\n")

    for index, input_path in enumerate(input_files, start=1):

        print(
            f"[{index}/{len(input_files)}] "
            f"Processing {input_path.name}"
        )

        try:
            # ------------------------------------------------
            # Load input
            # ------------------------------------------------

            noisy = np.load(input_path)

            noisy = validate_input(
                noisy,
                input_path.name,
            )

            # ------------------------------------------------
            # Restormer
            # ------------------------------------------------

            restored = run_restormer(
                restormer,
                noisy,
            )

            restored = np.clip(
                restored,
                0.0,
                1.0,
            )

            # ------------------------------------------------
            # HAT-L
            # ------------------------------------------------

            hr = run_hat(
                hat,
                restored,
            )

            # ------------------------------------------------
            # Final validation
            # ------------------------------------------------

            hr = validate_output(
                hr,
                input_path.name,
            )

            # ------------------------------------------------
            # Output paths
            # ------------------------------------------------

            npy_output = output_dir / input_path.name

            png_output = png_dir / (
                input_path.stem + ".png"
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            save_outputs(
                hr,
                npy_output,
                png_output,
            )

        except Exception as exc:

            print(
                f"\nERROR while processing "
                f"{input_path.name}:"
            )

            print(exc)

            raise

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Restored .npy files : {output_dir}")
    print(f"PNG visualizations  : {png_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
