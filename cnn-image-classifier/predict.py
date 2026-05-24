"""
Inference script — run predictions on a single image or a folder of images.

Usage:
    python predict.py --model checkpoints/best_model.keras --image path/to/img.jpg
    python predict.py --model checkpoints/best_model.keras --folder path/to/images/
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def preprocess(image_path: str, target_size=(32, 32)) -> np.ndarray:
    """Load and preprocess a single image for CIFAR-10 inference."""
    img = Image.open(image_path).convert("RGB").resize(target_size)
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = (x - MEAN) / STD
    return np.expand_dims(x, 0)  # (1, H, W, 3)


def predict_single(model, image_path: str, top_k=3):
    x      = preprocess(image_path)
    probs  = model(x, training=False).numpy()[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    print(f"\nPredictions for: {image_path}")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. {CIFAR10_CLASSES[idx]:<12} {probs[idx]*100:.2f}%")
    return CIFAR10_CLASSES[top_idx[0]], probs[top_idx[0]]


def predict_folder(model, folder: str):
    folder = Path(folder)
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    if not images:
        print("No images found.")
        return
    results = []
    for img_path in images:
        label, conf = predict_single(model, str(img_path))
        results.append({"file": img_path.name, "label": label, "confidence": conf})
    return results


def main():
    parser = argparse.ArgumentParser(description="CNN Inference")
    parser.add_argument("--model",  required=True, help="Path to saved .keras model")
    parser.add_argument("--image",  default=None,  help="Single image path")
    parser.add_argument("--folder", default=None,  help="Folder of images")
    parser.add_argument("--top_k",  type=int, default=3)
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = tf.keras.models.load_model(args.model)

    if args.image:
        predict_single(model, args.image, top_k=args.top_k)
    elif args.folder:
        predict_folder(model, args.folder)
    else:
        print("Provide --image or --folder.")


if __name__ == "__main__":
    main()
