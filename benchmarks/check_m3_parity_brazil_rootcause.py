"""Root-cause ablation for Brazilian_houses: histree T7 Q16r_int with (a) its own axis bin edges
(np.unique(np.quantile(...))) vs (b) miter's AxisBinner edges injected into histree.
Everything else identical (same Q16r codes from the prototype, same seeds, same inner split).
Usage: python check_m3_parity_brazil_rootcause.py <dataset> [n_seeds]  -> benchmarks/results/raw/m3_rootcause_<dataset>.json
"""
import json, math, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, "/Users/kacperpiotrowski/Studia/Projekt/histree/src")
import numpy as np
import miter
from sklearn.model_selection import train_test_split
import check_m3_parity_brazil_pol as c
from histree.gb.oblique_int import ObliqueGBInt

name = sys.argv[1]; n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
task, X, y = c.load(c.DATASETS[name]); p = X.shape[1]; m0 = math.ceil(math.sqrt(p))
rows = []
for fold in range(5):
    tr, te = task.get_train_test_split_indices(repeat=0, fold=fold)
    a, b = train_test_split(tr, test_size=0.2, random_state=0); sd = float(np.std(y[te]))
    Q = c.q16r_proto(X[a], X)
    ds = miter.Dataset(X[a]); medges = [ds.bin_edges(j).astype(np.float64) for j in range(p)]
    class HM(ObliqueGBInt):
        def _fit_binning(self, XX):
            ObliqueGBInt._fit_binning(self, XX)
            self.edges_ = medges; self.nbins_f_ = np.array([len(e) + 1 for e in medges], dtype=np.int64)
    for s in range(n_seeds):
        seed = fold + 100 * s; rec = dict(fold=fold, seed=seed)
        for lab, cls in (("own_edges", ObliqueGBInt), ("miter_edges", HM)):
            h = cls(selector="R", m=2 * m0, bin_sample=-3, random_state=seed, **c.CFG).fit(Q[a], y[a], Q[b], y[b])
            rec[lab] = float(np.sqrt(np.mean((h.predict(Q[te]) - y[te]) ** 2)) / sd); rec[lab + "_trees"] = int(h.best_iteration_)
        rows.append(rec); print(rec, flush=True)
(c.RAW / f"m3_rootcause_{name}.json").write_text(json.dumps(rows, indent=1))
