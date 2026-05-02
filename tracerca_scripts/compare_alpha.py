"""Compose the caller-discount sweep comparison report from results_alpha_*.txt."""
import re
from collections import OrderedDict

ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.8]
FILES = {a: f"results_alpha_{a if a != 0.0 else 0}.txt" for a in ALPHAS}

# pre-registered downstream-cause cases (TT, from source CLAUDE.md)
MECH_CASES = ["ts-order-service_delay/2", "ts-order-service_delay/3"]
# stress-check: order is the structural caller-accumulation service per original Light eval §2
STRESS_CASES_PREFIX = "ts-order-service_"


def parse_file(path):
    out = {"overall": {}, "fault": OrderedDict(), "service": OrderedDict(), "rank": {}}
    with open(path) as f:
        text = f.read()

    # Overall
    m = re.search(r"HR@1=([\d.]+)\s+HR@3=([\d.]+)\s+HR@5=([\d.]+)", text)
    if m:
        out["overall"] = {1: float(m.group(1)), 3: float(m.group(2)), 5: float(m.group(3))}

    # Per-fault block
    fb = re.search(r"=== Per-fault-type ===\n.*?\n((?:.*\n)*?)\n", text)
    if fb:
        for line in fb.group(1).strip().splitlines():
            parts = line.split()
            if len(parts) == 5 and not parts[0].startswith("fault"):
                out["fault"][parts[0]] = {1: float(parts[2]), 3: float(parts[3]), 5: float(parts[4])}

    # Per-service block
    sb = re.search(r"=== Per ground-truth service ===\n.*?\n((?:.*\n)*?)\n", text)
    if sb:
        for line in sb.group(1).strip().splitlines():
            parts = line.split()
            if len(parts) == 5 and parts[0].startswith("ts-"):
                out["service"][parts[0]] = {1: float(parts[2]), 3: float(parts[3]), 5: float(parts[4])}

    # Per-case rank-of-truth
    for line in re.findall(r"^(ts-\S+)\tgt=\S+\trank=(\S+)$", text, flags=re.M):
        case, r = line
        out["rank"][case] = None if r == "None" else int(r)

    return out


def main():
    runs = {a: parse_file(p) for a, p in FILES.items()}

    print("=" * 72)
    print("TraceRCA Caller-Discount Re-ranking — α sweep on RE2-TT")
    print("=" * 72)

    # Overall HR table
    print("\n## Overall HR@k (n=88)")
    print(f"{'alpha':>6} | {'HR@1':>5} {'Δ':>6} | {'HR@3':>5} {'Δ':>6} | {'HR@5':>5} {'Δ':>6}")
    print("-" * 60)
    base = runs[0.0]["overall"]
    for a in ALPHAS:
        o = runs[a]["overall"]
        d1, d3, d5 = o[1] - base[1], o[3] - base[3], o[5] - base[5]
        print(f"{a:>6.1f} | {o[1]:>5.2f} {d1:>+6.2f} | {o[3]:>5.2f} {d3:>+6.2f} | {o[5]:>5.2f} {d5:>+6.2f}")

    # Per-fault HR@1
    print("\n## Per-fault HR@1 by α")
    faults = list(runs[0.0]["fault"].keys())
    header = f"{'fault':10s} " + " ".join(f"α={a:>3.1f}" for a in ALPHAS)
    print(header)
    print("-" * len(header))
    for f in faults:
        row = f"{f:10s} " + " ".join(f"{runs[a]['fault'][f][1]:>5.2f}" for a in ALPHAS)
        print(row)

    # Per-service HR@1
    print("\n## Per-service HR@1 by α")
    svcs = list(runs[0.0]["service"].keys())
    header = f"{'service':22s} " + " ".join(f"α={a:>3.1f}" for a in ALPHAS)
    print(header)
    print("-" * len(header))
    for s in svcs:
        row = f"{s:22s} " + " ".join(f"{runs[a]['service'][s][1]:>5.2f}" for a in ALPHAS)
        print(row)

    # Per-service HR@3
    print("\n## Per-service HR@3 by α")
    print(header)
    print("-" * len(header))
    for s in svcs:
        row = f"{s:22s} " + " ".join(f"{runs[a]['service'][s][3]:>5.2f}" for a in ALPHAS)
        print(row)

    # Mechanism check — Δrank on pre-registered cases
    print("\n## Mechanism check: Δrank(gt) vs α=0 on pre-registered downstream-cause cases")
    print("(positive Δrank = improvement, i.e. true-cause climbed)")
    head = f"{'case':35s} " + " ".join(f"α={a:>3.1f}" for a in ALPHAS) + "  | Δ@best"
    print(head)
    print("-" * len(head))
    for c in MECH_CASES:
        ranks = [runs[a]["rank"].get(c) for a in ALPHAS]
        base_r = ranks[0]
        cells = []
        deltas = []
        for r in ranks:
            cells.append("None" if r is None else f"{r:>5d}")
            if r is not None and base_r is not None:
                deltas.append(base_r - r)
        best_d = max(deltas) if deltas else 0
        print(f"{c:35s} " + " ".join(cells) + f"  | {best_d:>+6d}")

    # Stress check on all ts-order-service cases
    print("\n## Stress check: all ts-order-service_* cases — Δrank vs α=0")
    order_cases = sorted(c for c in runs[0.0]["rank"] if c.startswith(STRESS_CASES_PREFIX))
    print(head)
    print("-" * len(head))
    total_delta = {a: 0 for a in ALPHAS}
    n_evaluable = 0
    for c in order_cases:
        ranks = [runs[a]["rank"].get(c) for a in ALPHAS]
        base_r = ranks[0]
        cells = []
        deltas = []
        for a, r in zip(ALPHAS, ranks):
            cells.append("None" if r is None else f"{r:>5d}")
            if r is not None and base_r is not None:
                d = base_r - r
                deltas.append(d)
                total_delta[a] += d
        if base_r is not None:
            n_evaluable += 1
        best_d = max(deltas) if deltas else 0
        print(f"{c:35s} " + " ".join(cells) + f"  | {best_d:>+6d}")
    print(f"{'TOTAL Δrank (sum across cases)':35s} " + " ".join(f"{total_delta[a]:>5d}" for a in ALPHAS))

    # Acceptance criteria summary
    print("\n## Acceptance summary")
    base_h1 = base[1]
    for a in ALPHAS[1:]:
        o = runs[a]["overall"]
        d1 = o[1] - base_h1
        verdict = "OK" if d1 >= -0.02 else "REGRESSION"
        print(f"  α={a}: HR@1 Δ={d1:+.3f} ({verdict}; ≤2-pt regression tolerance)")

    best_a = max(ALPHAS, key=lambda a: runs[a]["overall"][1])
    print(f"\n  Best α by HR@1: {best_a} (HR@1={runs[best_a]['overall'][1]:.2f}, "
          f"baseline={base_h1:.2f})")
    best_a3 = max(ALPHAS, key=lambda a: runs[a]["overall"][3])
    print(f"  Best α by HR@3: {best_a3} (HR@3={runs[best_a3]['overall'][3]:.2f}, "
          f"baseline={base[3]:.2f})")

    # Sweep shape
    h1s = [runs[a]["overall"][1] for a in ALPHAS]
    if h1s == sorted(h1s):
        shape = "monotone-up (under-tuned, push α higher?)"
    elif h1s == sorted(h1s, reverse=True):
        shape = "monotone-down (mechanism wrong here?)"
    elif len(set(h1s)) == 1:
        shape = "flat (discount not biting)"
    else:
        peak = ALPHAS[h1s.index(max(h1s))]
        shape = f"single-peaked at α={peak}"
    print(f"  Sweep shape (HR@1 vs α): {shape}")


if __name__ == "__main__":
    main()
