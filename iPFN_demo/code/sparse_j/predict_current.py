"""Current five-array inference: frozen J plus predeclared regional calibration.

Original frozen-J and kernel loaders stay component loaders. This public entry
applies the one declared gate, leaving all N>=50 and partial-outcome cases exact.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import torch
from predict import load_selected_model as load_frozen_J,read_factual_input
from regional_inference import RegionalArmEnsemble,regional_gate


def load_current_model(root=None):
    from benchmark import configuration,load_methods
    root=Path(root) if root is not None else Path(__file__).resolve().parent
    selected=json.loads((root/'regional_selected.json').read_text())
    digest=hashlib.sha256((root/'regional_inference.py').read_bytes()).hexdigest()
    if digest!=selected['inference_sha256']:raise ValueError('Regional inference source hash mismatch')
    frozen=load_frozen_J(root/'models/iPFN_J.pt')
    kernels=load_methods(['kernel-old','kernel-new'],configuration())
    return RegionalArmEnsemble({'j':frozen,**kernels},selected['settings'],selected['primary_weights']['j'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    inputs=read_factual_input(args.input);model=load_current_model()
    means=model.predict_arm_means(**inputs)
    effects=np.stack((means[:,2]-means[:,0],means[:,3]-means[:,2],means[:,3]-means[:,0]),-1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,arm_means=means,ite=effects,ate=effects.mean(0))
    route="regional calibrated ensemble" if regional_gate(inputs) else "frozen J"
    print(f"Current iPFN: N={len(means)} m={len(inputs['observed_indices'])}; route={route}")


if __name__=='__main__':main()
