#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

STAMP_RE = re.compile(r"^(?P<base>.+)_\d{8}_\d{6}$")  # ..._YYYYMMDD_HHMMSS


def strip_stamp(dir_name: str) -> str:
    m = STAMP_RE.match(dir_name)
    return m.group("base") if m else dir_name


def load_eval(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "eval.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def find_all_runs_for_prefix(runs_dir: Path, run_prefix: str) -> List[Tuple[Path, str, float]]:
    """Return list of (run_dir, tag_str, tag_value) for directories matching prefix and having eval.json."""
    out: List[Tuple[Path, str, float]] = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(run_prefix):
            continue
        if not (d / "eval.json").exists():
            continue

        base = strip_stamp(d.name)  # remove _YYYYMMDD_HHMMSS so tag is stable
        if not base.startswith(run_prefix):
            continue
        tag = base[len(run_prefix):]  # e.g. "2e-5" or "40"
        if not tag:
            continue

        # Parse tag as float
        try:
            val = float(tag)
        except Exception:
            # If you ever add non-float tags, you can handle them here.
            continue

        out.append((d, tag, val))
    return out


def choose_best(df: pd.DataFrame, target_l0: float) -> pd.DataFrame:
    out = df.copy()
    out["l0_err"] = (out["l0_mean"] - target_l0).abs()
    out = out.sort_values(["l0_err", "fvu", "avg_max_decoder_cos"], ascending=[True, True, True])
    return out


def keep_latest_per_value(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent run_dir per (method, value) based on timestamped folder name."""
    # Lexicographic sorting works for YYYYMMDD_HHMMSS.
    df2 = df.sort_values(["method", "value", "run_dir"])
    return df2.groupby(["method", "value"], as_index=False).tail(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="l0_sweep_cfg.json")
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--target_l0", type=float, default=None)
    ap.add_argument("--show_all_cols", action="store_true")
    ap.add_argument("--mode", type=str, default="latest_per_value",
                    choices=["latest_per_value", "all"],
                    help="Whether to show only latest run per lambda/k or all runs.")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    runs_dir = Path(args.runs_dir)

    target_l0 = float(args.target_l0 if args.target_l0 is not None else cfg.get("target_l0", 40.0))
    methods: List[Dict[str, Any]] = list(cfg.get("methods", []))

    records: List[Dict[str, Any]] = []

    for m in methods:
        name = str(m["name"])
        prefix = str(m["run_prefix"])
        knob = str(m.get("knob", "lambda_base"))

        found = find_all_runs_for_prefix(runs_dir, prefix)
        if not found:
            continue

        for run_dir, tag, val in found:
            ev = load_eval(run_dir)
            if ev is None:
                continue
            rec: Dict[str, Any] = {
                "method": name,
                "knob": knob,
                "lam": tag if knob != "target_l0" else None,  # keep 'lam' column for penalties
                "value": float(val),                          # numeric for sorting
                "run_dir": run_dir.name,
            }
            rec.update(ev)
            records.append(rec)

    if not records:
        print("No eval.json files found for the configured prefixes.")
        return

    df = pd.DataFrame(records)

    if args.mode == "latest_per_value":
        df = keep_latest_per_value(df)

    # Columns to show
    core_cols = [
        "lam", "value", "l0_mean", "fvu", "avg_max_decoder_cos",
        "dead_frac_p_lt_1e_6", "rare_frac_p_lt_1e_4", "run_dir",
    ]
    if args.show_all_cols:
        show_cols = [c for c in df.columns if c not in ("method",)]
    else:
        show_cols = [c for c in core_cols if c in df.columns]

    summary: List[Dict[str, Any]] = []

    for method, g in df.groupby("method"):
        g = g.copy()

        # Pick best by closeness to target L0 (tie-break fvu then cosine)
        if all(c in g.columns for c in ["l0_mean", "fvu", "avg_max_decoder_cos"]):
            g_best = choose_best(g, target_l0)
            best = g_best.iloc[0]
        else:
            best = g.iloc[0]

        # Print sorted by numeric lambda/k
        g_disp = g.sort_values("value")

        print(f"\n=== {method} (target_l0={target_l0}) ===")
        cols = ["method"] + show_cols
        cols = [c for c in cols if c in g_disp.columns]
        print(g_disp[cols].to_string(index=False))

        summary.append({
            "method": method,
            "best_knob": best.get("knob", ""),
            "best_value": best.get("lam", best.get("value", "")),
            "best_l0_mean": float(best.get("l0_mean", float("nan"))),
            "best_fvu": float(best.get("fvu", float("nan"))),
            "best_avg_max_decoder_cos": float(best.get("avg_max_decoder_cos", float("nan"))),
        })

    print("\n=== Best per method (closest to target_l0; tie-break fvu then cosine) ===")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
