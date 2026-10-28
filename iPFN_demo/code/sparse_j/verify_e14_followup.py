"""Audit complete saved samples, pairing, geometry, source and frozen assets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark import configuration,ROOT
from make_table import read_results
from pfn_social.e14_rgg import radius_graph

def audit_run(directory,families,count):
    rows,digest=read_results(directory)
    manifest=json.loads((directory/'execution.json').read_text())
    expected={(f'f{f}_m{m}_r{r}_observed{b}',method) for f in families for m in range(count)
        for r in range(12) for b in (20,50,100) for method in ('j','ht','hajek','kernel-old','kernel-new')}
    observed=[(r['context_id'],r['method']) for r in manifest['receipts']]
    assert len(observed)==len(set(observed)) and set(observed)==expected
    for identity,method in observed:
        assert (directory/'predictions'/(identity+'__'+method+'.npz')).is_file()
    for identity in {i for i,m in observed}:assert (directory/'factual'/(identity+'.npz')).is_file()
    runtime=json.loads((directory/'runtime.json').read_text())
    for name,digest_saved in runtime['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest_saved,name
    assert manifest['config']['frozen_assets_sha256']==configuration()['frozen_assets_sha256']
    return dict(contexts=len(expected)//5,method_contexts=len(expected),ate_rows=len(rows),
        prediction_failures=sum(bool(r['failure']) for r in manifest['receipts']),
        interval_failures=sum(bool(r['interval_failure']) for r in manifest['receipts']),
        context_metrics_sha256=digest,source_and_asset_hashes='passed')

def audit_pairing(directory):
    pairs=0
    for mid in range(50):
        graphs=[]
        for family in (13,14):
            with np.load(directory/'graphs'/f'f{family}_m{mid}.npz') as z:
                a,pos,mask=z['adjacency'],z['positions'],z['observed_permutation']
                assert a.sum()==3600 and np.all(a.sum(1)>0)
                assert np.array_equal(a,a.T) and not a.diagonal().any()
                distances=np.linalg.norm(pos[:,None]-pos[None,:],axis=-1)
                hi=distances[a==1].max();lo=distances[(a==0)&~np.eye(200,dtype=bool)].min()
                assert hi<lo
                np.testing.assert_array_equal(a,radius_graph(pos,(hi+lo)/2))
                graphs.append((a.copy(),mask.copy()))
        np.testing.assert_array_equal(graphs[0][1],graphs[1][1])
        assert not np.array_equal(graphs[0][0],graphs[1][0])
        for rep in range(12):
            for budget in (20,50,100):
                paths=[directory/'factual'/f'f{family}_m{mid}_r{rep}_observed{budget}.npz' for family in (13,14)]
                with np.load(paths[0]) as u,np.load(paths[1]) as c:
                    for key in ('x','treatment','observed_indices'):np.testing.assert_array_equal(u[key],c[key])
                    for z,graph in ((u,graphs[0]),(c,graphs[1])):
                        np.testing.assert_array_equal(z['adjacency'],graph[0])
                        assert z['outcomes_observed'].shape==(budget,)
                        assert set(z.files)=={'x','adjacency','treatment','observed_indices','outcomes_observed'}
                pairs+=1
    return dict(graphs=100,paired_contexts=pairs,all_checks='passed')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('replication','matched','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=dict(replication=audit_run(a.replication,[9],100),matched=audit_run(a.matched,[13,14],50),pairing=audit_pairing(a.matched))
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
