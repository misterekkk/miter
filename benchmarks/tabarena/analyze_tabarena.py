"""Compare miter C++ with cached GBMs and the numba prototype from per-split CSVs. Usage: analyze_tabarena.py <miter_eval_per_split.csv> <out.json>"""

import json
import sys

import pandas as pd
from scipy.stats import wilcoxon

H = "/Users/kacperpiotrowski/Studia/Projekt/histree/research/results/t2/"
a = pd.read_csv(sys.argv[1])
a = a[a.method != "x"]
proto = pd.concat(
    [
        pd.read_csv(H + f)[lambda d: d.method.str.contains("MITER")]
        for f in (
            "T2A_full_eval_per_split.csv",
            "T2A_tid11_eval_per_split.csv",
            "T2K1_eval_per_split.csv",
        )
    ]
)
proto["method"] = "[Proto] numba (RUNI on 6 no-cat sets, K1 on 7 cat sets)"
proto = proto[["dataset", "fold", "method", "metric_error", "time_train_s"]]
cpp = a[a.method.str.contains("MITER_CPP")]
base = a[~a.method.str.contains("MITER")]
allm = pd.concat(
    [
        cpp[["dataset", "fold", "method", "metric_error", "time_train_s"]],
        base[["dataset", "fold", "method", "metric_error", "time_train_s"]],
        proto,
    ]
)
# prototype: use K1 for datasets with categories (T2K1), RUNI for A datasets; K1 file also includes miami/wine (no cats) -> identical to RUNI-B
allm = allm.drop_duplicates(["dataset", "fold", "method"], keep="first")
pv = allm.pivot_table(
    index=["dataset", "fold"], columns="method", values="metric_error"
)
tt = allm.pivot_table(
    index="dataset", columns="method", values="time_train_s", aggfunc="median"
)
mean_ds = pv.groupby("dataset").mean()
out = {
    "mean_err": mean_ds.round(5).to_dict(),
    "median_train_s": tt.round(2).to_dict(),
    "nsplits": pv.groupby("dataset").size().to_dict(),
}


def cmp(m, b):
    ok = pv[[m, b]].dropna()
    d = ok.groupby("dataset").mean()
    g = (d[b] - d[m]) / d[b] * 100  # + = m better
    p = wilcoxon(g).pvalue if len(g) > 5 and (g != 0).any() else float("nan")
    return dict(
        n=len(g),
        mean=round(g.mean(), 3),
        median=round(g.median(), 3),
        wins=int((g > 0).sum()),
        p=round(float(p), 4),
        per_ds=g.round(3).to_dict(),
    )


M = sys.argv[3] if len(sys.argv) > 3 else "[New] MITER_CPP (default)"
res = {}
for b in [c for c in pv.columns if c != M]:
    if M in pv and pv[[M, b]].dropna().shape[0]:
        res[b] = cmp(M, b)
out["cmp"] = res
json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
for b, r in res.items():
    print(
        f"{b:34s} n={r['n']:2d} mean={r['mean']:+.2f}% med={r['median']:+.2f}% wins={r['wins']}/{r['n']} p={r['p']}"
    )
