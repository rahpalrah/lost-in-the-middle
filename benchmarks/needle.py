"""Needle-in-a-haystack test for long-document question answering (standard library only).

Hides facts at known depths in synthetic contracts of growing length, full of near-duplicate
distractor clauses that share the question's words, then checks whether the answer engine finds
them. Two question kinds:

* ``single``: one fact ("What is the notice period for Service Q7?").
* ``multi``:  two facts in different places, asked in one question (needs decomposition).

    python benchmarks/needle.py                        # default sizes, built-in engine
    python benchmarks/needle.py --compare              # baseline vs context-rot + recursive

The built-in engine is deterministic, so the numbers are exactly reproducible for a given seed.
Accuracy = the answer text contains the needle's value(s); retrieval = the needle's section is in
the context that would be sent. Both are reported with Wilson 95% intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from longdoc import Document, DocumentQA, QAConfig  # noqa: E402
from longdoc.extractive import ExtractiveClient  # noqa: E402

TOPICS = ("notice period", "liability cap", "payment term", "audit frequency", "insurance minimum")
UNITS = {
    "notice period": "{n} days",
    "liability cap": "{n} times the annual fees",
    "payment term": "{n} days from the invoice date",
    "audit frequency": "once every {n} months",
    "insurance minimum": "{n} million dollars",
}
BOILER = (
    "The parties acknowledge that this provision is material and that each relied on it. "
    "No waiver is effective unless made in writing and signed by the waiving party. "
    "Nothing in this section limits any right or remedy available at law or in equity. "
)


@dataclass
class Case:
    document: str
    question: str
    values: list[str]  # strings that must all appear in a correct answer
    needle_ids: list[str]  # entity names whose sections must be retrieved


def make_case(chars: int, depths: list[float], seed: int, kinds: list[str]) -> Case:
    rng = random.Random(seed)
    entities = [f"Service Z{rng.randint(1000, 9999)}-{i}" for i in range(len(depths))]
    topics = rng.sample(TOPICS, len(depths))
    values = [str(rng.randint(11, 97)) for _ in depths]
    sections: list[str] = []
    n = 0
    # distractors: same topics and wording, different entities and numbers
    while sum(len(s) for s in sections) < chars:
        n += 1
        topic = rng.choice(TOPICS)
        ent = f"Service A{rng.randint(1, 900)}"
        val = UNITS[topic].format(n=rng.randint(11, 97))
        sections.append(
            f"# {n}. {topic.title()} for {ent}\n\nThe {topic} for {ent} is {val}. "
            f"{BOILER * rng.randint(2, 5)}\n"
        )
    for ent, topic, value, depth in zip(entities, topics, values, depths, strict=True):
        val = UNITS[topic].format(n=value)
        needle = f"# N. {topic.title()} for {ent}\n\nThe {topic} for {ent} is {val}. {BOILER}\n"
        sections.insert(min(len(sections), int(depth * len(sections))), needle)
    if "multi" in kinds and len(depths) > 1:
        q = " and ".join(f"the {t} for {e}" for t, e in zip(topics, entities, strict=True))
        question = f"What are {q}?"
    else:
        question = f"What is the {topics[0]} for {entities[0]}?"
        values = values[:1]
        entities = entities[:1]
    return Case("\n".join(sections), question, values, entities)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def evaluate(config: QAConfig, kind: str, sizes: list[int], depths_list: list[list[float]],
             seeds: int) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for size in sizes:
        retr = acc = total = 0
        for depths in depths_list:
            for seed in range(seeds):
                case = make_case(size, depths, seed * 101 + size, [kind])
                qa = DocumentQA(Document.from_text(case.document), client=ExtractiveClient(),
                                config=config)
                context = qa.retrieve(case.question)
                joined = "\n".join(p.text for p in context)
                got_retr = all(e in joined for e in case.needle_ids)
                answer = qa.ask(case.question).text
                got_acc = all(v in answer for v in case.values)
                total += 1
                retr += got_retr
                acc += got_acc
        out[size] = {
            "n": total,
            "retrieval": [retr, wilson(retr, total)],
            "answer": [acc, wilson(acc, total)],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sizes", default="50000,300000,1500000", help="Document sizes in characters")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--compare", action="store_true", help="Compare configurations side by side")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="QAConfig override for the run, e.g. --set context_rot=true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",")]

    def cfg(**extra: object) -> QAConfig:
        return QAConfig(fallbacks=None, **extra)  # type: ignore[arg-type]

    overrides: dict[str, object] = {}
    for item in args.set:
        k, v = item.split("=", 1)
        overrides[k] = {"true": True, "false": False}.get(v.lower(), int(v) if v.isdigit() else v)
    configs = {"configured": cfg(**overrides)}
    if args.compare:
        configs = {
            "baseline": cfg(context_rot=False, recursive=False),
            "context_rot": cfg(context_rot=True, recursive=False),
            "recursive": cfg(context_rot=False, recursive=True),
            "both": cfg(context_rot=True, recursive=True),
        }
    single_depths = [[d] for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
    multi_depths = [[0.1, 0.9], [0.3, 0.6], [0.5, 0.52], [0.05, 0.95]]
    report: dict[str, object] = {}
    for name, config in configs.items():
        res = {
            "single": evaluate(config, "single", sizes, single_depths, args.seeds),
            "multi": evaluate(config, "multi", sizes, multi_depths, args.seeds),
        }
        report[name] = res
        print(f"== {name}")
        for kind, per_size in res.items():
            for size, r in per_size.items():
                print(f"  {kind:<6} {size:>9,} chars  n={r['n']:<3} "
                      f"retrieval {r['retrieval'][0]}/{r['n']} {r['retrieval'][1]}  "  # type: ignore[index]
                      f"answer {r['answer'][0]}/{r['n']} {r['answer'][1]}")  # type: ignore[index]
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
