"""Download cached results of ALL methods of TabArenaContext (official downloader, normal channel) into TABARENA_CACHE."""

import os
import time

os.environ.setdefault(
    "TABARENA_CACHE",
    "/Users/kacperpiotrowski/Studia/Projekt/histree/research/results/tabarena_cache",
)
from tabarena.contexts import TabArenaContext

ctx = TabArenaContext()
print(len(ctx.methods), "methods:", ctx.methods, flush=True)
t0 = time.time()
df = ctx.load_results(download_results="auto")
print(
    "DONE",
    f"{time.time() - t0:.0f}s",
    df.shape,
    df.method.nunique(),
    "variants",
    flush=True,
)
df.to_pickle(os.environ.get("OUT", "/tmp/_all_results.pkl")) if False else None
