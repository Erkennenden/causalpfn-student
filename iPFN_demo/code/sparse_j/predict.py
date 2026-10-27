"""Five-array inference for the globally selected frozen iPFN J checkpoint."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import torch
from pfn_social.phase5_meta_bayes import load_model
from pfn_social.sparse_frozen_gaussian import SparseFrozenGaussian

CHECKPOINT_SHA256 = "a0dde8e928d7d48c0ef8581b81cf38fecd9525408bff7179b4df7ee761fed7a6"
INPUT_KEYS = {"x", "adjacency", "treatment", "observed_indices", "outcomes_observed"}


def load_selected_model(checkpoint=None):
    checkpoint = Path(checkpoint) if checkpoint is not None else Path(__file__).parent / "models/iPFN_J.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Frozen J checkpoint hash mismatch")
    parent, _ = load_model(checkpoint)
    if parent.config.architecture != "J":
        raise ValueError("This entry point requires the selected J architecture")
    return SparseFrozenGaussian(parent, identity={"kind": "target", "sha256": CHECKPOINT_SHA256, "frozen_no_refit": True})


def read_factual_input(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != INPUT_KEYS:
            raise ValueError("Exactly five observed arrays are required; no oracle or hidden outcomes")
        return {key: data[key].copy() for key in INPUT_KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval-level", type=float, default=None,
                        help="Also return marginal posterior intervals, e.g. 0.95")
    args = parser.parse_args()
    if args.interval_level is not None and not 0 < args.interval_level < 1:
        parser.error('--interval-level must be between 0 and 1')
    torch.set_num_threads(1)
    inputs = read_factual_input(args.input)
    model = load_selected_model()
    means = model.predict_arm_means(**inputs)
    effects = np.stack((means[:, 2] - means[:, 0], means[:, 3] - means[:, 2], means[:, 3] - means[:, 0]), -1)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    intervals = {}
    if args.interval_level is not None:
        from pfn_social.intervals import predict_intervals
        intervals = predict_intervals(model, **inputs, level=args.interval_level)
    np.savez_compressed(destination, arm_means=means, ite=effects, ate=effects.mean(0), **intervals)
    print(f"N={len(means)} m={len(inputs['outcomes_observed'])}; frozen J SHA verified; four arms returned")


if __name__ == "__main__":
    main()
