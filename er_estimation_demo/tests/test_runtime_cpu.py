"""Run with stdlib unittest; no accelerator or torch is needed."""
import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pfn_interference.runtime import MetricAccumulator, read_csv_arrays, validation_indices


class RuntimeCPUTests(unittest.TestCase):
    def test_validation_partition_has_no_duplicates_or_padding(self):
        for count in (1, 7, 8, 11, 5120):
            parts = [validation_indices(count, r, 8) for r in range(8)]
            self.assertEqual(sorted(i for part in parts for i in part), list(range(count)))
            self.assertLessEqual(max(map(len, parts)) - min(map(len, parts)), 1)

    def test_global_rmse_and_normalization_not_average_of_batch_rmse(self):
        acc = MetricAccumulator()
        for count, mse, mae, scale in [(1, 1., 1., 2.), (3, 9., 3., 4.)]:
            metrics = {"validation_gmm_nll": 2.0}
            for effect in ("direct", "spillover", "total"):
                metrics.update({effect+"_effect_mse": mse, effect+"_effect_mae": mae,
                                effect+"_effect_rmse": math.sqrt(mse)})
            acc.add(metrics, count, [scale]*3)
        result = acc.result()
        self.assertAlmostEqual(result["direct_effect_rmse"], math.sqrt(7))
        self.assertAlmostEqual(result["direct_effect_nmae_pct"], 2.5/3.5*100)
        self.assertAlmostEqual(result["effect_rmse_macro"], math.sqrt(7))
        merged = MetricAccumulator()
        merged.merge(acc.payload())
        self.assertEqual(merged.result(), result)

    def make_csv(self, path, duplicate=False):
        rows=[]
        for unit in range(2):
            row={"schema_version":"test-v1", "dataset_id":"17", "n_units":"2",
                 "unit_id":str(0 if duplicate else unit), "graph_type":"2", "star_center":"-1",
                 "adjacency_bits": "2" if unit==0 else "1"}
            for field in ("x", "observed_treatment", "observed_exposure", "degree", "majority_arm",
                          "low_sampled_exposure", "high_sampled_exposure", "y_obs", "structural_baseline",
                          "tau", "gamma", "eta", "oracle_direct_ite", "oracle_spillover_ite", "oracle_total_ite"):
                row[field]=str(unit+1)
            for i in range(5): row[f"token_{i}"]=str(unit+i/10)
            for effect in ("direct","spillover","total"):
                for i in range(4): row[f"{effect}_query_{i}"]=str(i)
                for suffix in ("outcome_a","outcome_b","effect"): row[f"{effect}_{suffix}"]=str(unit+.25)
            rows.append(row)
        with path.open("w", newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

    def test_csv_float32_labels_adjacency_and_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"one.csv"; self.make_csv(path)
            arrays, ids=read_csv_arrays(path, "test-v1", 5, 4, ("direct","spillover","total"), 1)
            self.assertEqual(ids,[17])
            np.testing.assert_array_equal(arrays["adjacency"], [[[0,1],[1,0]]])
            np.testing.assert_array_equal(arrays["query_effect"], [[[.25]*3,[1.25]*3]])
            self.assertEqual(arrays["tokens"].dtype,np.float32)
            self.assertEqual(arrays["majority_arm"].dtype,np.int64)
            arrays2, ids2=read_csv_arrays(path, "test-v1", 5, 4, ("direct","spillover","total"))
            self.assertEqual(ids, ids2)
            for key in arrays: np.testing.assert_array_equal(arrays[key],arrays2[key])

    def test_duplicate_rows_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"one.csv"; self.make_csv(path,duplicate=True)
            with self.assertRaises(ValueError):
                read_csv_arrays(path,"test-v1",5,4,("direct","spillover","total"),1)

    def test_incorrect_expected_count_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"one.csv"; self.make_csv(path)
            with self.assertRaises(ValueError):
                read_csv_arrays(path,"test-v1",5,4,("direct","spillover","total"),2)

    def test_random_partition_metric_merge_matches_direct_calculation(self):
        rng=np.random.default_rng(713)
        truth=rng.normal(size=(11,7,3))
        pred=truth+rng.normal(size=truth.shape)
        combined=MetricAccumulator()
        for rank in range(8):
            indices=validation_indices(11,rank,8)
            y=truth[indices];p=pred[indices];error=p-y
            acc=MetricAccumulator();metrics={}
            for j,effect in enumerate(("direct","spillover","total")):
                metrics.update({effect+"_effect_mse":np.mean(error[...,j]**2),
                                effect+"_effect_mae":np.mean(abs(error[...,j])),
                                effect+"_effect_bias":np.mean(error[...,j]),
                                effect+"_effect_rmse":np.sqrt(np.mean(error[...,j]**2))})
            acc.add(metrics,len(indices)*7,np.mean(abs(y),axis=(0,1)))
            combined.merge(acc.payload())
        out=combined.result()
        for j,effect in enumerate(("direct","spillover","total")):
            error=pred[...,j]-truth[...,j]
            self.assertAlmostEqual(out[effect+"_effect_rmse"],np.sqrt(np.mean(error**2)))
            self.assertAlmostEqual(out[effect+"_effect_mae"],np.mean(abs(error)))
            self.assertAlmostEqual(out[effect+"_effect_bias"],np.mean(error))

    def test_invalid_adjacency_bits_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"one.csv";self.make_csv(path)
            with path.open(newline="") as f:
                reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
            rows[0]["adjacency_bits"]="4" # bit outside the two-node graph
            with path.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            with self.assertRaises(ValueError):
                read_csv_arrays(path,"test-v1",5,4,("direct","spillover","total"),1)


if __name__ == "__main__": unittest.main()
