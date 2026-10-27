"""Stage 12: re-score every stored safety response with the current refusal classifier (CPU only, no model needed).

The first classifier missed refusals written with a typographic apostrophe ("I can’t"), which Llama-3.2 uses; this
re-scores the saved XSTest responses in every result file, updates the safety entries of each summary, and recomputes
the retention in results/<model>/diagnosis/damage.json. Planning artifacts (results/<model>/fit/) are left as they were,
because the allocations that were actually trained were planned from them.

  python scripts/12_rescore_safety.py
"""
import _bootstrap  # noqa: F401
from collections import Counter

import numpy as np

from rah.config import REPO, load_config
from rah.evaluation.scorers import score_safe
from rah.utils import read_json, write_json

MODELS = ["llama", "qwen", "smollm", "olmo"]


def rescore(path) -> int:
    d = read_json(path)
    safe = [r for r in d["records"] if r.get("cap") == "safe"]
    flips = 0
    for r in safe:
        new = score_safe(r["response"], {"unsafe": r["unsafe"]})
        flips += new["refused"] != r["refused"]
        r["refused"], r["score"] = new["refused"], new["score"]
    if flips:
        s = d["summary"]
        s["safe"] = float(np.mean([r["score"] for r in safe]))
        s["safe_harmful_refusal"] = float(np.mean([r["refused"] for r in safe if r["unsafe"]]))
        s["safe_over_refusal"] = float(np.mean([r["refused"] for r in safe if not r["unsafe"]]))
        write_json(path, d)
    return flips


def update_damage(name):
    diag = REPO / "results" / name / "diagnosis"
    caps = load_config(name)["capabilities"]
    dmg = read_json(diag / "damage.json")
    for lvl, split in (("main", "test"), ("main", "dev"), ("light", "test")):
        pruned = read_json(diag / f"eval_{lvl}_{split}.json")["summary"]
        base = read_json(diag / f"eval_base_{split}.json")["summary"]
        r = {c: pruned[c] / base[c] if base.get(c, 0) > 1e-9 else float("nan") for c in caps}  # as runner.retention
        dmg[f"{lvl}_{split}"] = {"retention": r, "damage": {c: 1 - r[c] for c in caps}}
    write_json(diag / "damage.json", dmg)


def main():
    changed = Counter()
    for name in MODELS:
        for path in sorted((REPO / "results" / name).rglob("*.json")):
            d = read_json(path)
            if isinstance(d, dict) and any(r.get("cap") == "safe" for r in d.get("records", [])):
                n = rescore(path)
                if n:
                    changed[name] += n
                    print(f"{path.relative_to(REPO)}: {n} safety decisions changed")
        update_damage(name)
    print("changed decisions per model:", dict(changed))


if __name__ == "__main__":
    main()
