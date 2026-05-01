import glob
import json
import os
import subprocess
import sys

import pandas as pd

from RCAEval.e2e import tracerca

DATASET_ROOT = "./re2tt"
KS = (1, 3, 5)


def service_ranks(op_ranks):
    """Operation ranks -> deduped service ranks (first occurrence)."""
    seen = set()
    out = []
    for op in op_ranks:
        svc = op.split("_", 1)[0].replace("-db", "")
        if svc not in seen:
            seen.add(svc)
            out.append(svc)
    return out


def evaluate_case(case_dir):
    data = pd.read_csv(os.path.join(case_dir, "traces.csv"))
    with open(os.path.join(case_dir, "inject_time.txt")) as f:
        inject_time = int(f.read().strip())
    op_ranks = tracerca(data, inject_time=inject_time)["ranks"]
    return service_ranks(op_ranks)


def run_worker(case_dir):
    """Worker mode: evaluate one case, print JSON ranks to stdout."""
    svc_ranks = evaluate_case(case_dir)
    print(json.dumps(svc_ranks))


def run_case_subprocess(case_dir, timeout=300):
    """Spawn a subprocess to evaluate a case so segfaults don't kill the parent."""
    result = subprocess.run(
        [sys.executable, "-u", __file__, "--case", case_dir],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        reason = f"exit={result.returncode}"
        if result.returncode == -11 or result.returncode == 139:
            reason = "segfault"
        stderr_tail = result.stderr.strip().splitlines()[-1:] if result.stderr.strip() else []
        raise RuntimeError(f"{reason} {' | '.join(stderr_tail)}")
    last_line = result.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def main():
    case_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "*", "*")))
    case_dirs = [d for d in case_dirs if os.path.isdir(d)]

    per_fault = {}
    overall = {f"hit@{k}": 0 for k in KS}
    overall["total"] = 0
    errors = 0

    for case_dir in case_dirs:
        parent = os.path.basename(os.path.dirname(case_dir))
        instance = os.path.basename(case_dir)
        gt_service, fault_type = parent.rsplit("_", 1)

        try:
            svc_ranks = run_case_subprocess(case_dir)
        except Exception as e:
            errors += 1
            print(f"{parent}/{instance}: ERROR — {e}")
            continue

        hits = {f"hit@{k}": gt_service in svc_ranks[:k] for k in KS}
        flag_str = "  ".join(f"{name}={int(v)}" for name, v in hits.items())
        print(f"{parent}/{instance}: GT={gt_service}  top5={svc_ranks[:5]}  {flag_str}")

        bucket = per_fault.setdefault(fault_type, {f"hit@{k}": 0 for k in KS} | {"total": 0})
        bucket["total"] += 1
        overall["total"] += 1
        for k in KS:
            bucket[f"hit@{k}"] += int(hits[f"hit@{k}"])
            overall[f"hit@{k}"] += int(hits[f"hit@{k}"])

    print("\n=== Per-fault-type ===")
    print(f"{'fault':10s} {'n':>3s}  " + "  ".join(f"HR@{k}" for k in KS))
    for fault in sorted(per_fault):
        s = per_fault[fault]
        n = s["total"]
        rates = "  ".join(f"{s[f'hit@{k}']/n:5.2f}" for k in KS)
        print(f"{fault:10s} {n:>3d}  {rates}")

    print("\n=== Overall ===")
    n = overall["total"]
    print(f"cases evaluated: {n} (errors: {errors})")
    if n:
        rates = "  ".join(f"HR@{k}={overall[f'hit@{k}']/n:.2f}" for k in KS)
        print(rates)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--case":
        run_worker(sys.argv[2])
    else:
        main()
