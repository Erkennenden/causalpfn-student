"""Truth-assisted explanation of oracle shrinkage, never a benchmark predictor.

For fixed design and coefficients, analytically integrate over outcome noise.
True beta appears ONLY here to compute conditional bias; it is not a fit input
to the deployable J or the observed-Y oracle used in performance comparisons.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr, ndtri
from threadpoolctl import threadpool_limits

from make_table import write_csv, EFFECTS
from pfn_social.e12_configuration import generate_mechanism
from pfn_social.phase3_hierarchical import coefficient_prior
from scripts.phase5_strong_controls import design_from_h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    prior,covariance=coefficient_prior()
    records=[]
    with threadpool_limits(limits=1):
        m=generate_mechanism(103,2).parent
        for repeat in range(12):
            context=f'f103_m2_r{repeat}_observed50'
            with np.load(args.input/'factual'/f'{context}.npz') as saved:
                t,ids=(saved[k] for k in ('treatment','observed_indices'))
                np.testing.assert_array_equal(saved['adjacency'],m.adjacency)
                np.testing.assert_array_equal(saved['x'],m.x)
            factual,query=design_from_h(m.h,m.adjacency,t)
            f=factual[ids]
            noise=m.setting.noise_sd**2
            factor=cho_factor(f@covariance@f.T+noise*np.eye(len(f)),lower=True)
            gain=cho_solve(factor,f@covariance).T
            contrasts=np.stack((query[:,2]-query[:,0],query[:,3]-query[:,2],query[:,3]-query[:,0]),1).mean(0)
            # True coefficients are diagnostic targets, not available to an estimator.
            expected_beta=prior+gain@(f@(m.coefficients-prior))
            bias=contrasts@(expected_beta-m.coefficients)
            sampling_cov=noise*(contrasts@gain)@(contrasts@gain).T
            posterior_cov=contrasts@(covariance-gain@f@covariance)@contrasts.T
            sampling_sd=np.sqrt(np.maximum(np.diag(sampling_cov),0))
            half=ndtri(.975)*np.sqrt(np.maximum(np.diag(posterior_cov),0))
            for i,effect in enumerate(EFFECTS):
                coverage=ndtr((half[i]-bias[i])/sampling_sd[i])-ndtr((-half[i]-bias[i])/sampling_sd[i])
                records.append(dict(family=103,mechanism=2,repeat=repeat,m_observed=50,effect=effect,
                    expected_bias_over_noise=bias[i],sampling_sd_over_noise=sampling_sd[i],
                    posterior_half_width=half[i],conditional_coverage_over_noise=coverage,
                    diagnostic_uses_true_coefficients=True))
    write_csv(args.output,records)
    for effect in EFFECTS:
        rows=[r for r in records if r['effect']==effect]
        print(effect, {key:float(np.mean([r[key] for r in rows])) for key in (
            'expected_bias_over_noise','sampling_sd_over_noise','posterior_half_width','conditional_coverage_over_noise')})


if __name__=='__main__':
    main()
