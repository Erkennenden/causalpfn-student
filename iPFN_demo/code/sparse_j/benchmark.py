"""Evaluate the current frozen J, supported classics, two kernels and deep adapters."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import hashlib
import json
import os
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from predict import load_selected_model
from pfn_social.phase6_limited_outcomes import generate_mechanism, make_observed_draw
from pfn_social.phase6_sparse_classical import DesignBased, FrozenMixture, CausalPFNReference, pack
from pfn_social.sparse_classical_calibration import load_model as load_calibrated_kernel
from scripts.phase4_strong_controls import design as old_design
from scripts.phase5_frozen_controls import load_prior

ROOT = Path(__file__).resolve().parent
CLASSICS = {"ht": "HT", "hajek": "Hajek", "f": "F (raw20)", "lin": "Lin (raw20)",
            "f-z2": "F-z2", "lin-z2": "Lin-z2", "non-interference": "Non-interference OLS"}
DEFAULT_METHODS = ("j", *CLASSICS, "kernel-old", "kernel-new")
ALL_METHODS = (*DEFAULT_METHODS, "causalpfn", "hinet", "tnet", "hypersci")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def source_directory():
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def configuration():
    config = json.loads((ROOT / "experiment.json").read_text())
    for relative, digest in config["frozen_assets_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Frozen asset hash mismatch: {relative}")
    return config


def load_methods(names, config):
    """Load only requested current methods; never load a historical candidate grid."""
    result = {}
    with np.load(ROOT / config["transform"], allow_pickle=False) as saved:
        transform = {k: saved[k].copy() for k in ("x_mean", "x_scale", "projection")}
    for name in names:
        if name == "j":
            result[name] = load_selected_model()
        elif name in CLASSICS:
            result[name] = DesignBased(CLASSICS[name], transform)
        elif name == "kernel-old":
            protocol = json.loads((ROOT / config["old_kernel_protocol"]).read_text())
            components = []
            for component in protocol["components"]:
                row = component["selection"]
                spec, prior = load_prior(ROOT / row["prior_file"])
                components.append((spec, prior, row["value"], old_design))
            result[name] = FrozenMixture(components, json.loads((ROOT / config["old_kernel_selection"]).read_text()))
        elif name == "kernel-new":
            result[name], _ = load_calibrated_kernel(ROOT / config["new_kernel_checkpoint"])
        elif name == "causalpfn":
            from pfn_social.estimation import load_causalpfn_checkpoint
            identity = config["CausalPFN_external_checkpoint"]
            checkpoint = ROOT / identity["path"]
            if not checkpoint.exists():
                raise FileNotFoundError("Copy the original official CausalPFN checkpoint to " + str(checkpoint))
            if sha(checkpoint) != identity["sha256"]:
                raise ValueError("Official CausalPFN checkpoint hash mismatch")
            result[name] = CausalPFNReference(load_causalpfn_checkpoint(checkpoint, torch.device("cpu")), True)
        elif name == "hinet":
            from pfn_social.phase6_sparse_deep import fit_hinet_masked
            result[name] = fit_hinet_masked
        elif name == "tnet":
            from pfn_social.phase6_sparse_deep import fit_tnet_masked
            result[name] = fit_tnet_masked
        elif name == "hypersci":
            from pfn_social.phase6_masked_hypersci import fit_predict_masked_hypersci
            result[name] = fit_predict_masked_hypersci
        else:
            raise ValueError(name)
    return result


def predict_method(name, method, inputs, seed):
    if name in ("j", "kernel-new"):
        return pack(method.predict_arm_means(**inputs), n=len(inputs["x"]))
    if name == "hypersci":
        means, _ = method(seed=seed, **inputs)
        return pack(means, n=len(inputs["x"]))
    if name in ("hinet", "tnet"):
        return method(seed=seed, **inputs)
    return method.predict(**inputs)


def cases(args):
    if args.cohort == "full":
        return [(f, i, range(12), (20, 50, 100)) for f in range(7) for i in range(8)]
    if args.cohort == "pilot84":
        return [(f, i, range(2), (20, 50, 100)) for f in range(7) for i in range(2)]
    return [(args.family, args.mechanism, (args.repeat,), (args.observed,))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("one", "full", "pilot84"), default="one")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--family", type=int, choices=range(7), default=0)
    parser.add_argument("--mechanism", type=int, choices=range(8), default=0)
    parser.add_argument("--repeat", type=int, choices=range(12), default=0)
    parser.add_argument("--observed", type=int, choices=(20, 50, 100), default=50)
    parser.add_argument("--output", type=Path, required=True, help="A fresh output directory")
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Duplicate methods")
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError("Choose a fresh output directory")
    torch.set_num_threads(1)
    config = configuration()
    destination.mkdir(parents=True)
    for directory in ("factual", "truth", "predictions"):
        (destination / directory).mkdir()
    started = time.perf_counter()
    with source_directory(), threadpool_limits(limits=1):
        methods = load_methods(args.methods, config)
        load_seconds = time.perf_counter() - started
        receipts = []
        for family, mechanism_id, repeats, masks in cases(args):
            mechanism = generate_mechanism(family, mechanism_id)
            # Truth is serialized separately and never supplied to any predictor.
            np.savez_compressed(destination / "truth" / f"f{family}_m{mechanism_id}.npz", arm_means=mechanism.parent.truth)
            for repeat in repeats:
                rows, _ = make_observed_draw(mechanism, repeat)
                for row in rows:
                    if row["m_observed"] not in masks:
                        continue
                    identity = row["context_id"]
                    inputs = {key: value.numpy().copy() for key, value in row["batch"].items()}
                    np.savez_compressed(destination / "factual" / (identity + ".npz"), **inputs)
                    for name, model in methods.items():
                        seed = 651013 + 1000 * family + 100 * mechanism_id + repeat
                        began = time.perf_counter()
                        failure = None
                        try:
                            prediction = predict_method(name, model, inputs, seed)
                        except Exception as error:
                            prediction = pack(ite=np.full((200, 3), np.nan), ate=np.full(3, np.nan), n=200)
                            failure = {"type": type(error).__name__, "message": str(error)}
                        elapsed = time.perf_counter() - began
                        np.savez_compressed(destination / "predictions" / (identity + "__" + name + ".npz"),
                            **{key: prediction[key] for key in ("arm_means", "ite_effects", "ate_effects", "arm_supported", "ite_supported", "ate_supported")})
                        receipts.append(dict(context_id=identity, method=name, fit_seed=seed, online_seconds=elapsed,
                            arm_supported=prediction["arm_supported"].tolist(), ite_supported=prediction["ite_supported"].tolist(),
                            ate_supported=prediction["ate_supported"].tolist(), failure=failure))
    (destination / "execution.json").write_text(json.dumps(dict(config=config, methods=args.methods, cohort=args.cohort,
        model_load_seconds=load_seconds, torch_threads=1, BLAS_threads=1, receipts=receipts,
        no_training_of_pretrained_J_or_kernel_parameters=True, no_hidden_outcomes_supplied=True), indent=2))
    print(json.dumps(dict(output=str(destination), method_contexts=len(receipts), cohort=args.cohort), indent=2))


if __name__ == "__main__":
    main()
