#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def most_recent(runs_dir: Path, prefix: str) -> Optional[Path]:
    ds = sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith(prefix + "_")])
    for d in reversed(ds):
        if (d / "eval.json").exists():
            return d
    return None

def load_eval(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "eval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def choose_best(df: pd.DataFrame, target_l0: float) -> pd.DataFrame:
    """Sort by |L0-target|, then fvu, then decoder cosine."""
    out = df.copy()
    out["l0_err"] = (out["l0_mean"] - target_l0).abs()
    out = out.sort_values(["l0_err", "fvu", "avg_max_decoder_cos"], ascending=[True, True, True])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="l0_sweep_cfg.json")
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--target_l0", type=float, default=None)
    ap.add_argument("--show_all_cols", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    runs_dir = Path(args.runs_dir)

    target_l0 = float(args.target_l0 if args.target_l0 is not None else cfg.get("target_l0", 40.0))
    lambdas: List[str] = list(cfg.get("lambdas", []))
    methods: List[Dict[str, Any]] = list(cfg.get("methods", []))

    records: List[Dict[str, Any]] = []

    for m in methods:
        name = str(m["name"])
        prefix = str(m["run_prefix"])
        knob = str(m.get("knob", "lambda_base"))

        if knob == "target_l0":
            run_prefix = f"{prefix}{int(target_l0) if float(target_l0).is_integer() else target_l0}"
            d = most_recent(runs_dir, run_prefix)
            if d is None:
                continue
            ev = load_eval(d)
            if ev is None:
                continue
            rec = {
                "method": name,
                "knob": "target_l0",
                "value": float(target_l0),
                "run_name": run_prefix,
                "run_dir": d.name,
            }
            rec.update(ev)
            records.append(rec)
            continue

        # penalty methods: loop over lambdas
        for lam in lambdas:
            run_prefix = f"{prefix}{lam}"
            d = most_recent(runs_dir, run_prefix)
            if d is None:
                continue
            ev = load_eval(d)
            if ev is None:
                continue
            rec = {
                "method": name,
                "knob": "lambda_base",
                "lam": lam,
                "value": float(lam),
                "run_name": run_prefix,
                "run_dir": d.name,
            }
            rec.update(ev)
            records.append(rec)

    if not records:
        print("No eval.json files found for the configured sweeps.")
        return

    df = pd.DataFrame(records)

    # Columns to show
    core_cols = [
        "lam", "value", "l0_mean", "fvu", "avg_max_decoder_cos",
        "dead_frac_p_lt_1e_6", "rare_frac_p_lt_1e_4", "run_dir",
    ]
    if args.show_all_cols:
        show_cols = [c for c in df.columns if c not in ("method", "knob", "run_name")]  # show everything useful
    else:
        show_cols = [c for c in core_cols if c in df.columns]

    summary: List[Dict[str, Any]] = []

    for method, g in df.groupby("method"):
        g2 = g.copy()
        # Only penalty methods have lam column; batchtopk won't.
        if "l0_mean" in g2.columns and "fvu" in g2.columns and "avg_max_decoder_cos" in g2.columns:
            g2 = choose_best(g2, target_l0)

        print(f"\n=== {method} (target_l0={target_l0}) ===")
        cols = ["method"] + show_cols
        cols = [c for c in cols if c in g2.columns]
        print(g2[cols].sort_values('value').to_string(index=False))

        best = g2.iloc[0]
        summary.append({
            "method": method,
            "best_knob": best.get("knob", ""),
            "best_value": best.get("lam", best.get("value", "")),
            "best_l0_mean": float(best.get("l0_mean", float("nan"))),
            "best_fvu": float(best.get("fvu", float("nan"))),
            "best_avg_max_decoder_cos": float(best.get("avg_max_decoder_cos", float("nan"))),
            # "run_dir": best.get("run_dir", ""),
        })

    print("\n=== Best per method (closest to target_l0; tie-break fvu then cosine) ===")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
