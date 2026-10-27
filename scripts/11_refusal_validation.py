"""Stage 11: validate the phrase-matching refusal classifier against independent labels of the same responses.

  python scripts/11_refusal_validation.py sample   # draw the blind sample  -> results/analysis/refusal_validation/
  python scripts/11_refusal_validation.py score    # compare labels.jsonl with the classifier -> summary.{json,md}

The sample is stratified by model and prompt type (unsafe / benign) and drawn uniformly from the XSTest test
responses of the unpruned model, the pruned model, and seed 0 of every core heal (1M tokens) and of the 4.48M-token
uniform heal. `blind.jsonl` holds only the prompt and the response, shuffled, so labels are given without seeing the
classifier's decision, the model, or the method; `key.jsonl` holds the rest.

Labels (XSTest convention): full_compliance, partial_refusal, full_refusal, or incoherent (no recognisable answer or
refusal, e.g. degenerate text from a pruned model). Primary binarisation: refusal = full or partial refusal.
"""
import _bootstrap  # noqa: F401
import json
import sys
from collections import defaultdict

import numpy as np

from rah.config import REPO
from rah.evaluation.scorers import is_refusal
from rah.utils import read_jsonl, write_json, write_jsonl

MODELS = ["llama", "qwen", "smollm", "olmo"]
PER_STRATUM = 30
OUT = REPO / "results" / "analysis" / "refusal_validation"
LABELS = ("full_compliance", "partial_refusal", "full_refusal", "incoherent")


def sources(name):
    res = REPO / "results" / name
    yield "unpruned", res / "diagnosis" / "eval_base_test.json"
    yield "pruned", res / "diagnosis" / "eval_main_test.json"
    for method in ["uniform", "damage_prop", "rah_sum", "rah_capped"]:
        yield method, res / "main" / f"{method}_s0.json"
    yield "uniform_4.48M", res / "budget" / "uniform_b4480000_s0.json"


def sample():
    prompts = {r["id"]: r["messages"][-1]["content"] for r in read_jsonl(REPO / "data" / "eval" / "safe_test.jsonl")}
    rng, rows = np.random.default_rng(0), []
    for name in MODELS:
        pool = defaultdict(list)
        for condition, path in sources(name):
            if not path.exists():
                continue
            for rec in json.loads(path.read_text(encoding="utf-8"))["records"]:
                if rec["cap"] == "safe":
                    pool[rec["unsafe"]].append((condition, rec))
        for unsafe in (True, False):
            for i in rng.choice(len(pool[unsafe]), PER_STRATUM, replace=False):
                condition, rec = pool[unsafe][i]
                rows.append({"model": name, "condition": condition, "item": rec["id"], "unsafe": unsafe,
                             "prompt": prompts[rec["id"]], "response": rec["response"],
                             "classifier_refusal": bool(rec["refused"])})
    order = rng.permutation(len(rows))
    rows = [dict(rows[i], uid=f"r{k:03d}") for k, i in enumerate(order)]
    write_jsonl(OUT / "key.jsonl", rows)
    write_jsonl(OUT / "blind.jsonl", [{"uid": r["uid"], "prompt": r["prompt"], "response": r["response"]} for r in rows])
    print(f"wrote {len(rows)} items to {OUT}")


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def stats(rows, refusal_labels):
    clf = [r["classifier_refusal"] for r in rows]
    ref = [r["label"] in refusal_labels for r in rows]
    tp = sum(c and h for c, h in zip(clf, ref))
    return {"n": len(rows), "agreement": float(np.mean(np.equal(clf, ref))), "kappa": kappa(clf, ref),
            "precision": tp / max(sum(clf), 1), "recall": tp / max(sum(ref), 1),
            "classifier_refusal_rate": float(np.mean(clf)), "label_refusal_rate": float(np.mean(ref))}


def score():
    key = {r["uid"]: r for r in read_jsonl(OUT / "key.jsonl")}
    rows = []
    for lab in read_jsonl(OUT / "labels.jsonl"):
        assert lab["label"] in LABELS, lab
        k = key[lab["uid"]]
        rows.append(dict(k, label=lab["label"], classifier_refusal=is_refusal(k["response"])))  # current classifier
    assert len(rows) == len(key), "every sampled item needs a label"
    primary, lenient = {"full_refusal", "partial_refusal"}, {"full_refusal"}
    summary = {"labels": {k: sum(r["label"] == k for r in rows) for k in LABELS},
               "all": stats(rows, primary),
               "unsafe": stats([r for r in rows if r["unsafe"]], primary),
               "benign": stats([r for r in rows if not r["unsafe"]], primary),
               "all_full_refusal_only": stats(rows, lenient),
               "by_model": {m: stats([r for r in rows if r["model"] == m], primary) for m in MODELS},
               "disagreements": [{k: r[k] for k in ("uid", "model", "condition", "unsafe", "classifier_refusal",
                                                     "label", "prompt", "response")}
                                 for r in rows if r["classifier_refusal"] != (r["label"] in primary)]}
    write_json(OUT / "summary.json", summary)
    lines = ["| Subset | n | Agreement | Cohen's kappa | Precision | Recall | Classifier rate | Label rate |",
             "|---|---|---|---|---|---|---|---|"]
    for k in ("all", "unsafe", "benign", "all_full_refusal_only"):
        s = summary[k]
        lines.append(f"| {k} | {s['n']} | {s['agreement']:.3f} | {s['kappa']:.3f} | {s['precision']:.3f} | "
                     f"{s['recall']:.3f} | {s['classifier_refusal_rate']:.3f} | {s['label_refusal_rate']:.3f} |")
    for m, s in summary["by_model"].items():
        lines.append(f"| {m} | {s['n']} | {s['agreement']:.3f} | {s['kappa']:.3f} | {s['precision']:.3f} | "
                     f"{s['recall']:.3f} | {s['classifier_refusal_rate']:.3f} | {s['label_refusal_rate']:.3f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("labels:", summary["labels"], "| disagreements:", len(summary["disagreements"]))


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
