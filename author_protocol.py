"""Run configuration for the original train/val and soft-mask protocol."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from configs.config_setting import setting_config
from research_support import paired_files, package_versions
from utils import set_seed


PROTOCOL = "author_train_val_soft_masks_v1"
ABLATIONS = {
    "A0": ("off", "mask"),
    "A1": ("spatial", "mask"),
    "A2": ("low", "none"),
    "A3": ("high", "none"),
    "A4": ("both", "none"),
    "A5": ("both", "feature"),
    "A6": ("both", "mask"),
}


def make_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("isic17", "isic18"))
    parser.add_argument("--data", help="Override the original dataset directory")
    parser.add_argument("--experiment", choices=tuple(ABLATIONS))
    parser.add_argument("--mode", choices=("off", "low", "high", "both", "spatial"))
    parser.add_argument("--gate", choices=("none", "feature", "mask"))
    parser.add_argument("--stages", type=int, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--size", type=int, help="Square input size; formal runs use 256")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--out", help="Run directory; defaults to results/author_DATASET_EXPERIMENT_sSEED")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--resume", action="store_true", help="Restore settings and state from --out")
    actions.add_argument("--evaluate", action="store_true", help="Evaluate the saved best checkpoint on the same val")
    parser.add_argument("--visualize-ids", nargs="*", default=[], help="Val image stems for frequency panels")
    parser.add_argument("--visualize-stage", type=int, choices=range(1, 6), default=1)
    args = parser.parse_args(argv)
    if args.visualize_ids:
        parser.error("Frequency panel export is not implemented in train.py yet")
    config = SimpleNamespace(**{k: copy.deepcopy(v) for k, v in vars(setting_config).items()
                                if not k.startswith("_") and not isinstance(v, (staticmethod, classmethod))})
    saved = None
    if args.resume or args.evaluate:
        if not args.out:
            parser.error("--resume/--evaluate requires --out")
        for key in ("dataset", "data", "experiment", "mode", "gate", "stages", "seed", "epochs", "batch_size", "size"):
            if getattr(args, key) is not None:
                parser.error(f"--resume/--evaluate restores saved settings; omit --{key.replace('_', '-')}")
        saved = json.loads((Path(args.out) / "config.json").read_text(encoding="utf-8"))
        if saved["protocol"] != PROTOCOL:
            raise ValueError("This run does not use the author train/val protocol")
        vars(config).update(saved["settings"])
    else:
        if args.experiment and any(value is not None for value in (args.mode, args.gate, args.stages)):
            parser.error("Use either an A0-A6 preset or custom --mode/--gate/--stages")
        if args.dataset:
            config.datasets = args.dataset
            config.data_path = "data/" + {"isic17": "isic2017", "isic18": "isic2018"}[args.dataset]
        if args.data:
            config.data_path = args.data
        for key in ("seed", "epochs", "batch_size"):
            if getattr(args, key) is not None:
                setattr(config, key, getattr(args, key))
        if args.size is not None:
            config.input_size_h = config.input_size_w = args.size
        if args.experiment:
            mode, gate = ABLATIONS[args.experiment]
            config.model_config.update(freq_mode=mode, freq_gate=gate, freq_stages=[1, 2, 3])
        else:
            for arg, key in ((args.mode, "freq_mode"), (args.gate, "freq_gate"), (args.stages, "freq_stages")):
                if arg is not None:
                    config.model_config[key] = arg
        config.experiment = args.experiment or next((name for name, (mode, gate) in ABLATIONS.items()
            if config.model_config["freq_mode"] == mode and config.model_config["freq_gate"] == gate
            and list(config.model_config["freq_stages"]) == [1, 2, 3]), "custom")
    config.device = args.device or getattr(config, "device", "cuda")
    config.threads = args.threads if args.threads is not None else getattr(config, "threads", 1)
    for key in ("epochs", "batch_size", "threads", "print_interval", "val_interval", "save_interval"):
        if getattr(config, key) <= 0:
            parser.error(f"{key} must be positive")
    for size in (config.input_size_h, config.input_size_w):
        if size < 32 or size % 32:
            parser.error("Input dimensions must be positive multiples of 32")
    stages = config.model_config["freq_stages"]
    if not stages or len(set(stages)) != len(stages) or any(stage not in range(1, 6) for stage in stages):
        parser.error("--stages must contain distinct values from 1 to 5")
    if config.network != "egeunet" or not config.model_config["gt_ds"] or not config.model_config["bridge"]:
        parser.error("This entry uses EGEUNet with the original bridge and deep supervision")
    config.data_path = Path(config.data_path).resolve().as_posix() + "/"
    run = args.out or f"results/author_{config.datasets}_{config.experiment}_s{config.seed}"
    config.work_dir = Path(run).resolve().as_posix() + "/"
    config.resume, config.evaluate = args.resume, args.evaluate
    config.visualize_ids, config.visualize_stage = args.visualize_ids, args.visualize_stage
    # Keep the original one-angle-per-transform behavior, but seed its construction.
    set_seed(config.seed)
    config.train_transformer, config.test_transformer = setting_config.make_transforms(
        config.datasets, config.input_size_h, config.input_size_w)
    config.rotation_angle = config.train_transformer.transforms[4].angle
    if saved is not None:
        config.rotation_angle = saved["settings"]["rotation_angle"]
        config.train_transformer.transforms[4].angle = config.rotation_angle
    return config


def experiment_spec(config, train_dataset, val_dataset):
    excluded = {"work_dir", "resume", "evaluate", "visualize_ids", "visualize_stage"}
    settings = {}
    for key, value in vars(config).items():
        if key in excluded:
            continue
        try:
            settings[key] = json.loads(json.dumps(value))
        except TypeError:
            pass
    root = Path(config.data_path)
    manifest = {}
    for split, dataset in (("train", train_dataset), ("val", val_dataset)):
        expected = {(str(image.resolve()), str(mask.resolve())) for _, image, mask in paired_files(root, split)}
        actual = {(str(Path(image).resolve()), str(Path(mask).resolve())) for image, mask in dataset.data}
        if expected != actual:
            raise ValueError(f"Image/mask pairing is inconsistent in {split}")
        manifest[split] = [{"image": Path(image).relative_to(root).as_posix(),
                            "mask": Path(mask).relative_to(root).as_posix(),
                            "image_sha256": hashlib.sha256(Path(image).read_bytes()).hexdigest(),
                            "mask_sha256": hashlib.sha256(Path(mask).read_bytes()).hexdigest()}
                           for image, mask in dataset.data]
    source_root = Path(__file__).parent
    sources = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in (
        "train.py", "engine.py", "author_protocol.py", "configs/config_setting.py", "datasets/dataset.py",
        "models/egeunet.py", "models/frequency_bridge.py", "utils.py", "research_support.py")}
    spec = {"protocol": PROTOCOL, "selection": "min_val_loss", "report_split": "val",
            "independent_test": False, "settings": settings, "sources": sources,
            "packages": package_versions(),
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}
    return spec, manifest


def check_saved_spec(saved, current, evaluate=False):
    expected, actual = copy.deepcopy(saved), copy.deepcopy(current)
    if evaluate:
        for spec in (expected, actual):
            for key in ("device", "threads"):
                spec["settings"].pop(key, None)
    if expected != actual:
        raise ValueError("Saved and current config/data/code/packages differ; use the original run environment")
