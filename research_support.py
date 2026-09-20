"""Dataset validation and environment metadata for reproducible experiments."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def paired_files(root, split):
    """Match images to masks by ID, accepting ISIC's _segmentation suffix."""
    root = Path(root)
    indexed = []
    for folder in ("images", "masks"):
        directory = root / split / folder
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Missing dataset directory: {directory}. "
                "Pass --data with the directory containing train/ and val/."
            )
        files = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            sample_id = path.stem
            if folder == "masks" and sample_id.endswith("_segmentation"):
                sample_id = sample_id[:-len("_segmentation")]
            if sample_id in files:
                raise ValueError(f"Duplicate sample ID {sample_id!r} in {directory}")
            files[sample_id] = path
        if not files:
            raise ValueError(f"No supported images found in {directory}")
        indexed.append(files)
    images, masks = indexed
    if images.keys() != masks.keys():
        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        raise ValueError(
            f"Unpaired files in {root / split}: "
            f"missing masks={missing_masks[:5]}, missing images={missing_images[:5]}"
        )
    return [(sample_id, images[sample_id], masks[sample_id]) for sample_id in sorted(images)]


def package_versions():
    packages = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "numpy", "Pillow", "scipy", "scikit-learn",
                 "matplotlib", "tqdm", "timm", "einops", "thop", "tensorboardX",
                 "pytorch-wavelets", "PyWavelets"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return packages
