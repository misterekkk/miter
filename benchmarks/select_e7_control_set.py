"""Lists the E7 control-set candidates (exp_e7_ctr_default_prereg.md, rules 1-4; rule 5 is manual).

Usage (from benchmarks/): ../.venv/bin/python select_e7_control_set.py candidates.json
"""

from __future__ import annotations

import glob
import json
import re
import sys

import openml
from bench_exp_external_holdout import EXCLUDED_NAMES, _norm_name

SUITES = {
    271: "AMLB-clf",
    269: "AMLB-reg",
    99: "OpenML-CC18",
    353: "OpenML-CTR23",
    334: "Grinsztajn-clf-cat",
    335: "Grinsztajn-reg-cat",
}
DATASET_TUPLE = re.compile(r'\(\s*"(?:reg|clf|mc|bin|multi)"\s*,\s*"([^"]+)"\s*,\s*\d+\s*\)')


def seen_names() -> set[str]:
    """TabArena plus every dataset named in an earlier miter benchmark script."""
    seen = {_norm_name(x) for x in EXCLUDED_NAMES}
    for path in glob.glob("bench_*.py") + glob.glob("check_*.py") + glob.glob("diag_*.py"):
        with open(path) as f:
            seen.update(_norm_name(m.group(1)) for m in DATASET_TUPLE.finditer(f.read()))
    return seen


def main(out_path: str) -> None:
    seen = seen_names()
    rows: dict[int, dict] = {}
    for sid, sname in SUITES.items():
        suite = openml.study.get_suite(sid)
        tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
        for _, t in tasks.iterrows():
            did = int(t["did"])
            row = rows.setdefault(did, {
                "did": did,
                "name": t["name"],
                "suites": [],
                "ttype": t.get("task_type"),
                "n": int(t.get("NumberOfInstances", 0) or 0),
                "p": int(t.get("NumberOfFeatures", 0) or 0),
                "nsym": int(t.get("NumberOfSymbolicFeatures", 0) or 0),
                "ncls": int(t.get("NumberOfClasses", 0) or 0),
            })
            row["suites"].append(sname)
    for r in rows.values():
        is_clf = "Classification" in str(r["ttype"])
        r["ncat"] = r["nsym"] - (1 if is_clf else 0)  # OpenML counts a nominal target as symbolic
        r["kind"] = "reg" if not is_clf else ("clf" if r["ncls"] == 2 else "mc")
        r["excluded"] = _norm_name(r["name"]) in seen
    out = sorted(rows.values(), key=lambda r: r["did"])
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    ok = [r for r in out if not r["excluded"] and r["ncat"] >= 1 and 500 <= r["n"] <= 200_000 and r["p"] <= 500]
    print(f"{len(rows)} suite datasets; {len(ok)} eligible")
    for r in ok:
        print(r["did"], r["name"], r["kind"], r["n"], r["p"], "cat", r["ncat"], r["suites"])


if __name__ == "__main__":
    main(sys.argv[1])
