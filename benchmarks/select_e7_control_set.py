"""Lists the E7 control-set candidates (exp_e7_ctr_default_prereg.md, rules 1-4; rule 5 is manual).

Usage (from benchmarks/): ../.venv/bin/python select_e7_control_set.py candidates.json
"""
import re, glob, sys, json
sys.path.insert(0, ".")
import openml
from bench_exp_external_holdout import EXCLUDED_NAMES, _norm_name
seen = {_norm_name(x) for x in EXCLUDED_NAMES}
for f in glob.glob("bench_*.py") + glob.glob("check_*.py") + glob.glob("diag_*.py"):
    for m in re.finditer(r'\(\s*"(?:reg|clf|mc|bin|multi)"\s*,\s*"([^"]+)"\s*,\s*\d+\s*\)', open(f).read()):
        seen.add(_norm_name(m.group(1)))
SUITES = {271: "AMLB-clf", 269: "AMLB-reg", 99: "OpenML-CC18", 353: "OpenML-CTR23", 334: "Grinsztajn-clf-cat", 335: "Grinsztajn-reg-cat"}
rows = {}
for sid, sname in SUITES.items():
    s = openml.study.get_suite(sid)
    tasks = openml.tasks.list_tasks(task_id=s.tasks, output_format="dataframe")
    for _, t in tasks.iterrows():
        did = int(t["did"]); name = t["name"]
        rows.setdefault(did, {"did": did, "name": name, "suites": [], "ttype": t.get("task_type"),
            "n": int(t.get("NumberOfInstances", 0) or 0), "p": int(t.get("NumberOfFeatures", 0) or 0),
            "nsym": int(t.get("NumberOfSymbolicFeatures", 0) or 0), "ncls": int(t.get("NumberOfClasses", 0) or 0)})
        rows[did]["suites"].append(sname)
out = []
for r in rows.values():
    is_clf = "Classification" in str(r["ttype"])
    ncat = r["nsym"] - (1 if is_clf else 0)   # OpenML counts a nominal target as symbolic
    r["ncat"] = ncat; r["kind"] = ("reg" if not is_clf else ("clf" if r["ncls"] == 2 else "mc"))
    r["excluded"] = _norm_name(r["name"]) in seen
    out.append(r)
json.dump(sorted(out, key=lambda r: r["did"]), open(sys.argv[1], "w"), indent=1)
ok = [r for r in out if not r["excluded"] and r["ncat"] >= 1 and 500 <= r["n"] <= 200_000 and r["p"] <= 500]
print(len(rows), "suite datasets;", len(ok), "eligible")
for r in sorted(ok, key=lambda r: r["did"]):
    print(r["did"], r["name"], r["kind"], r["n"], r["p"], "cat", r["ncat"], r["suites"])
