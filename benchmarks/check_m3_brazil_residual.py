"""Causal test of the residual miter-vs-histree gap on Brazilian_houses (OpenML 44141).

Each variant swaps ONE component of the histree T7 prototype (ObliqueGBInt) toward miter's behaviour.
The prototype source is read, patched as text and exec'd into a private module; nothing on disk
in histree is modified. Same outer folds, same inner split, seed = fold + 100*s as in
check_m3_parity_brazil_pol.py.

Variants:
  orig    unpatched prototype
  f32     gradients rounded to float32 (miter: float g/h, double accumulation)   [swap a]
  rng1    oblique bin range = smax - smin + 1 (miter oblique_bin_index range)     [swap b]
  q16m    Q16 codes as miter: scale 65534, floor(v+0.5), clip to [0, 65534]        [swap c]
  combo   all swaps that moved the needle (set by --combo f32,rng1,...)
  miter   the C++ package (raw X)

Usage: python check_m3_brazil_residual.py run <variant> <s_lo> <s_hi> [--combo a,b]
       python check_m3_brazil_residual.py agg
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
HISTREE_SRC = Path("/Users/kacperpiotrowski/Studia/Projekt/histree/src")
sys.path.insert(0, str(HISTREE_SRC))

import check_m3_parity_brazil_pol as c  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

RAW = c.RAW
NAME = "Brazilian_houses"
PATCH_F32 = ("g = F - y; h = np.ones(n)", "g = (F - y).astype(np.float32).astype(np.float64); h = np.ones(n)")
PATCH_RNG1 = [("    rng = smax - smin\n", "    rng = smax - smin + 1\n"), ("    if rng <= 0:\n        return\n", "    if rng <= 1:\n        return\n")]


def make_cls(swaps: set[str]):
    import histree.gb.oblique_int as oi
    src = Path(oi.__file__).read_text()
    if "f32" in swaps:
        assert PATCH_F32[0] in src
        src = src.replace(*PATCH_F32)
    if "rng1" in swaps:
        for a, b in PATCH_RNG1:
            assert a in src, a
            src = src.replace(a, b)
    mod = types.ModuleType("histree.gb.oblique_int_variant")
    mod.__package__ = "histree.gb"
    mod.__file__ = oi.__file__
    sys.modules[mod.__name__] = mod
    exec(compile(src, oi.__file__, "exec"), mod.__dict__)
    return mod.ObliqueGBInt


def q16_variant(XA, X_all, miter_style: bool):
    if not miter_style:
        return c.q16r_proto(XA, X_all)
    lo = np.quantile(XA, 0.0001, axis=0)
    hi = np.quantile(XA, 0.9999, axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return np.clip(np.floor((X_all - lo) * (65534.0 / span) + 0.5), 0, 65534)


def run(variant: str, s_lo: int, s_hi: int, combo: set[str], folds: list[int]) -> None:
    task, X, y = c.load(c.DATASETS[NAME])
    p = X.shape[1]
    m0 = int(math.ceil(math.sqrt(p)))
    swaps = {"orig": set(), "f32": {"f32"}, "rng1": {"rng1"}, "q16m": {"q16m"}, "combo": combo, "miter": set()}[variant]
    if variant != "miter":
        cls = make_cls(swaps)
    else:
        import miter
    rows, t_start = [], time.perf_counter()
    for fold in folds:
        tr, te = task.get_train_test_split_indices(repeat=0, fold=fold)
        a, b = train_test_split(tr, test_size=0.2, random_state=0)
        sd = float(np.std(y[te]))
        if variant != "miter":
            Q = q16_variant(X[a], X, "q16m" in swaps)
        for s in range(s_lo, s_hi):
            seed = fold + 100 * s
            t0 = time.perf_counter()
            if variant == "miter":
                m = miter.MiterRegressor(n_estimators=5000, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
                                         lambda_l2=1.0, min_gain_to_split=0.0, max_bin=255, m_mult=2, k_proj=0,
                                         early_stopping_rounds=50, n_jobs=1, random_state=seed
                                         ).fit(X[a], y[a], eval_set=[(X[b], y[b])])
                pred, trees = m.predict(X[te]), len(m.train_loss_)
            else:
                m = cls(selector="R", m=2 * m0, bin_sample=-3, random_state=seed, **c.CFG).fit(Q[a], y[a], Q[b], y[b])
                pred, trees = m.predict(Q[te]), int(m.best_iteration_)
            e = float(np.sqrt(np.mean((pred - y[te]) ** 2)) / sd)
            rows.append(dict(fold=fold, seed=seed, nrmse=e, trees=trees, secs=time.perf_counter() - t0))
            print(f"{variant} fold={fold} seed={seed} nrmse={e:.6f} trees={trees} {rows[-1]['secs']:.1f}s", flush=True)
    tag = variant + ("_" + "+".join(sorted(combo)) if variant == "combo" else "")
    ftag = "" if len(folds) == 5 else "_pilot"
    out = dict(variant=tag, elapsed=time.perf_counter() - t_start, rows=rows)
    (RAW / f"m3_residual_{tag}_s{s_lo}-{s_hi}{ftag}.json").write_text(json.dumps(out, indent=1))


def load_rows(tag: str, rngs=((0, 5), (5, 10))):
    rows, el = [], 0.0
    for lo, hi in rngs:
        pth = RAW / f"m3_residual_{tag}_s{lo}-{hi}.json"
        if pth.exists():
            d = json.loads(pth.read_text())
            rows += d["rows"]
            el += d["elapsed"]
    return rows, el


def stats(a, b):
    from scipy.stats import wilcoxon
    rel = 100 * (a - b) / b
    return rel.mean(), np.median(rel), int((a < b).sum()), len(a), wilcoxon(a, b).pvalue, rel.std(ddof=1)


def agg() -> None:
    import platform
    import subprocess

    import miter
    import numba
    import scipy
    import sklearn
    d0 = json.loads((RAW / "m3_parity_Brazilian_houses_histree.json").read_text())["rows"]
    d1 = json.loads((RAW / "m3_parity_Brazilian_houses_miter.json").read_text())["rows"]
    data = {"orig": d0, "miter": d1}
    elapsed = {}
    for pth in sorted(RAW.glob("m3_residual_*_s*.json")):
        if "pilot" in pth.name:
            continue
        d = json.loads(pth.read_text())
        data.setdefault(d["variant"], [])
        data[d["variant"]] = data[d["variant"]] + d["rows"] if d["variant"] not in ("orig", "miter") or pth.name.endswith("s5-10.json") else data[d["variant"]]
        elapsed[d["variant"]] = elapsed.get(d["variant"], 0) + d["elapsed"]
    key = lambda r: (r["fold"], r["seed"])  # noqa: E731
    for k in data:
        data[k] = sorted(data[k], key=key)
    out = ["# Brazilian_houses residual gap: one-component swaps in histree T7 toward miter", "", "## Provenance"]
    out.append(f"- hardware/OS: Apple M4 (10 cores), {platform.platform()}, python {platform.python_version()}")
    out.append(f"- numpy {np.__version__}, scipy {scipy.__version__}, sklearn {sklearn.__version__}, numba {numba.__version__}, miter {miter.__version__}")
    out.append(f"- miter commit ff9d80f (with axis-binning fix 876f133); histree commit "
               f"{subprocess.run(['git', '-C', str(HISTREE_SRC.parent), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}")
    out.append("- 1 thread per process, <=2 processes in parallel, nice -n 10, another benchmark running concurrently (timings approximate)")
    out.append("- elapsed (sum of raw files, s): " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(elapsed.items())))
    out.append("- orig/miter s=0..4 are the raw files of m3_parity_brazil_pol; s=5..9 were run by this script (m3_residual_*_s5-10.json)")
    out.append("")
    for label, sel in (("s=0..4 (25 pairs)", lambda r: r["seed"] // 100 < 5), ("s=0..9 (50 pairs)", lambda r: True)):
        out.append(f"## {label}")
        base = {k: [r for r in v if sel(r)] for k, v in data.items()}
        need = len(base["miter"])
        names = [k for k in base if len(base[k]) == need]
        if "orig" not in names:
            out.append("(orig incomplete)")
            continue
        for k in names:
            assert [key(r) for r in base[k]] == [key(r) for r in base["orig"]]
        arr = {k: np.array([r["nrmse"] for r in base[k]]) for k in names}
        out.append("")
        out.append("| variant | mean nRMSE | median nRMSE | mean trees | vs orig: mean% / median% / wins / p | miter vs variant: mean% / median% / miter wins / p |")
        out.append("|---|---|---|---|---|---|")
        for k in names:
            tr = np.mean([r["trees"] for r in base[k]])
            if k == "orig":
                vo = "-"
            elif k == "miter":
                s_ = stats(arr[k], arr["orig"])
                vo = f"{s_[0]:+.2f} / {s_[1]:+.2f} / {s_[2]}/{s_[3]} / {s_[4]:.3g}"
            else:
                s_ = stats(arr[k], arr["orig"])
                vo = f"{s_[0]:+.2f} / {s_[1]:+.2f} / {s_[2]}/{s_[3]} / {s_[4]:.3g}"
            if k == "miter":
                vm = "-"
            else:
                s_ = stats(arr["miter"], arr[k])
                vm = f"{s_[0]:+.2f} / {s_[1]:+.2f} / {s_[2]}/{s_[3]} / {s_[4]:.3g}"
            out.append(f"| {k} | {arr[k].mean():.6f} | {np.median(arr[k]):.6f} | {tr:.0f} | {vo} | {vm} |")
        out.append("")
        out.append("(wins in 'vs orig' = variant has lower nRMSE than orig; wins in 'miter vs variant' = miter lower. Rel diff = (a-b)/b with a = first-named.)")
        fo = np.array([r["fold"] for r in base["orig"]])
        out.append("")
        out.append("| fold | miter mean | orig mean | rel diff of means % | paired rel diff mean % +- se |")
        out.append("|---|---|---|---|---|")
        for f in range(5):
            mm, oo = arr["miter"][fo == f], arr["orig"][fo == f]
            rr = 100 * (mm - oo) / oo
            out.append(f"| {f} | {mm.mean():.6f} | {oo.mean():.6f} | {100*(mm.mean()-oo.mean())/oo.mean():+.2f} | {rr.mean():+.2f} +- {rr.std(ddof=1)/math.sqrt(len(rr)):.2f} |")
        s_ = stats(arr["miter"], arr["orig"])
        out.append(f"- miter vs orig seed-to-seed std of rel diff: {s_[5]:.2f} pct points; standard error of mean rel diff: {s_[5]/math.sqrt(s_[3]):.2f}")
        out.append("")
    (Path(__file__).resolve().parent / "results" / "m3_brazil_residual_summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    if sys.argv[1] == "agg":
        agg()
    else:
        combo = set()
        folds = list(range(5))
        for i, a in enumerate(sys.argv):
            if a == "--combo":
                combo = set(sys.argv[i + 1].split(","))
            if a == "--folds":
                folds = [int(x) for x in sys.argv[i + 1].split(",")]
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), combo, folds)
