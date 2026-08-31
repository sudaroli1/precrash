"""
Score the literature audit: inter-rater agreement, disagreements to adjudicate,
and the summary counts the paper reports.

WHY AGREEMENT MATTERS HERE
--------------------------
The audit's claim is a count: "N of M surveyed papers report a timing metric
without AP or AUC." A referee's first question is whether a different reader
would have counted the same way. Two independent coders and a reported
agreement statistic answer that before it is asked; one coder's spreadsheet
does not, however careful that coder was.

Cohen's kappa corrects for agreement expected by chance, which matters when a
field is skewed — if 90% of papers omit AP, two coders who both guess "omitted"
agree 90% of the time while demonstrating nothing. Raw agreement is reported
alongside it because kappa behaves badly on very skewed fields, and a reader
should see both.

Conventional reading of kappa (Landis & Koch): below 0.40 poor, 0.40-0.60
moderate, 0.60-0.80 substantial, above 0.80 almost perfect. Report the number,
not the adjective.

USAGE
-----
  python scripts/audit_agreement.py \
      --coder_a audit/coder_a.csv \
      --coder_b audit/coder_b.csv \
      --out audit/agreement.json \
      --markdown audit/audit_table.md

Disagreements are written to audit/disagreements.csv for adjudication. Resolve
them together, record the agreed value in a third file, and re-run with
--adjudicated audit/final.csv to produce the final counts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# Fields coded for every paper. Keep in sync with the protocol document.
CODED_FIELDS = [
    "reports_ap",
    "reports_auc",
    "reports_timing_metric",
    "evaluates_negatives",
    "timing_reference_point",
    "metric_enforcing_postproc",
    "reports_corpus_mean",
    "releases_code",
]

# Permitted values. Anything else is a coding error and is reported as such.
ALLOWED = {
    "reports_ap":                {"yes", "no"},
    "reports_auc":               {"yes", "no"},
    "reports_timing_metric":     {"yes", "no"},
    "evaluates_negatives":       {"yes", "no", "unclear"},
    "timing_reference_point":    {"onset", "clip_end", "unstated", "na"},
    "metric_enforcing_postproc": {"yes", "no", "unclear"},
    "reports_corpus_mean":       {"corpus", "subset", "both", "na"},
    "releases_code":             {"yes", "no"},
}

ID_FIELD = "paper_id"


def parse_args():
    p = argparse.ArgumentParser(description="Score the literature audit")
    p.add_argument("--coder_a", required=True)
    p.add_argument("--coder_b", required=True)
    p.add_argument("--adjudicated", default=None,
                   help="Resolved codings; when given, the summary counts come from this")
    p.add_argument("--out", default="audit/agreement.json")
    p.add_argument("--markdown", default=None)
    return p.parse_args()


def read_coding(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    if ID_FIELD not in rows[0]:
        raise SystemExit(f"{path} needs a '{ID_FIELD}' column")

    out, problems = {}, []
    for r in rows:
        pid = r[ID_FIELD].strip()
        if not pid:
            continue
        if pid in out:
            problems.append(f"duplicate paper_id {pid}")
        rec = {}
        for fld in CODED_FIELDS:
            v = (r.get(fld) or "").strip().lower()
            if v and v not in ALLOWED[fld]:
                problems.append(f"{pid}/{fld}: '{v}' not in {sorted(ALLOWED[fld])}")
            rec[fld] = v
        out[pid] = rec
    if problems:
        print(f"  coding problems in {path.name}:")
        for p_ in problems[:15]:
            print("   ", p_)
        if len(problems) > 15:
            print(f"    ... and {len(problems)-15} more")
    return out


def cohens_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    """Return (kappa, raw agreement). Kappa is nan when it is undefined —
    which happens when both coders used a single value throughout, and is worth
    seeing rather than hiding behind a zero."""
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")

    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))

    if abs(1.0 - expected) < 1e-12:
        return float("nan"), observed
    return (observed - expected) / (1.0 - expected), observed


def main():
    args = parse_args()
    A = read_coding(Path(args.coder_a))
    B = read_coding(Path(args.coder_b))

    only_a, only_b = set(A) - set(B), set(B) - set(A)
    if only_a or only_b:
        print(f"\n  papers coded by only one coder: "
              f"{len(only_a)} in A only, {len(only_b)} in B only")
        for pid in sorted(only_a | only_b)[:10]:
            print(f"    {pid}")
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit("No papers coded by both coders.")

    print(f"\n{len(shared)} papers coded independently by both coders\n")
    print(f"{'field':<28}{'kappa':>8}{'raw agr':>10}{'disagree':>10}")
    print("-" * 56)

    agreement, disagreements = {}, []
    for fld in CODED_FIELDS:
        va = [A[p][fld] for p in shared]
        vb = [B[p][fld] for p in shared]
        k, raw = cohens_kappa(va, vb)
        n_dis = sum(x != y for x, y in zip(va, vb))
        agreement[fld] = {"kappa": k, "raw_agreement": raw, "n_disagree": n_dis,
                          "n": len(shared)}
        ks = "  n/a " if k != k else f"{k:>7.3f}"
        print(f"{fld:<28}{ks}{raw:>10.3f}{n_dis:>10}")

        for p in shared:
            if A[p][fld] != B[p][fld]:
                disagreements.append({"paper_id": p, "field": fld,
                                      "coder_a": A[p][fld], "coder_b": B[p][fld]})

    kappas = [v["kappa"] for v in agreement.values() if v["kappa"] == v["kappa"]]
    if kappas:
        print(f"\nmean kappa across fields: {sum(kappas)/len(kappas):.3f}")
    print(f"cells to adjudicate: {len(disagreements)} "
          f"of {len(shared)*len(CODED_FIELDS)}")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if disagreements:
        dpath = out_dir / "disagreements.csv"
        with open(dpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["paper_id", "field", "coder_a", "coder_b"])
            w.writeheader(); w.writerows(disagreements)
        print(f"\nwrote {dpath} — resolve these together before reporting counts")

    # ---- summary counts ----
    if args.adjudicated:
        final = read_coding(Path(args.adjudicated))
        basis = "adjudicated"
    else:
        final = {p: A[p] for p in shared}
        basis = "coder A only (PROVISIONAL — adjudicate before reporting)"

    print(f"\nSummary counts, basis: {basis}")
    print("-" * 56)
    summary = {}
    for fld in CODED_FIELDS:
        counts = Counter(final[p][fld] for p in final if final[p][fld])
        summary[fld] = dict(counts)
        total = sum(counts.values())
        parts = ", ".join(f"{v} {k}" for k, v in counts.most_common())
        print(f"  {fld:<28} {parts}   (n={total})")

    # the headline claim
    n = len(final)
    no_ap_no_auc = sum(1 for p in final
                       if final[p]["reports_ap"] == "no"
                       and final[p]["reports_auc"] == "no"
                       and final[p]["reports_timing_metric"] == "yes")
    no_neg = sum(1 for p in final if final[p]["evaluates_negatives"] in ("no", "unclear"))
    unstated_ref = sum(1 for p in final if final[p]["timing_reference_point"] == "unstated")

    print("\nHeadline counts for the paper")
    print("-" * 56)
    print(f"  timing metric but neither AP nor AUC : {no_ap_no_auc} of {n}")
    print(f"  negatives absent or not stated       : {no_neg} of {n}")
    print(f"  timing reference point not stated    : {unstated_ref} of {n}")

    payload = {"n_shared": len(shared), "agreement": agreement,
               "summary": summary, "basis": basis,
               "headline": {"timing_without_ap_or_auc": no_ap_no_auc,
                            "negatives_absent_or_unclear": no_neg,
                            "reference_point_unstated": unstated_ref,
                            "n": n}}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")

    if args.markdown:
        md = Path(args.markdown); md.parent.mkdir(parents=True, exist_ok=True)
        with open(md, "w", encoding="utf-8") as f:
            f.write(f"**Inter-rater agreement** ({len(shared)} papers, two independent coders)\n\n")
            f.write("| Field | Cohen's kappa | Raw agreement | Disagreements |\n|---|---|---|---|\n")
            for fld, v in agreement.items():
                ks = "n/a" if v["kappa"] != v["kappa"] else f"{v['kappa']:.3f}"
                f.write(f"| `{fld}` | {ks} | {v['raw_agreement']:.3f} | {v['n_disagree']} |\n")
            f.write(f"\n**Coding outcomes** (basis: {basis})\n\n| Field | Values |\n|---|---|\n")
            for fld, counts in summary.items():
                f.write(f"| `{fld}` | " +
                        ", ".join(f"{v} {k}" for k, v in
                                  sorted(counts.items(), key=lambda x: -x[1])) + " |\n")
        print(f"wrote {md}")


if __name__ == "__main__":
    main()
