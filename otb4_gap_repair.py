from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sync_alignment_tool as sat


def analyze_match(match: Dict[str, Any]) -> Dict[str, Any]:
    if not match.get("otb4") or not match.get("c3d"):
        raise ValueError("Match needs otb4 and c3d records.")
    candidates = sat._detect_otb4_repair_candidates(
        Path(match["otb4"]["path"]),
        match["otb4"].get("edge_times_sec") or [],
        match["c3d"].get("edge_times_sec") or [],
        float(match["otb4"].get("sample_rate") or 2000.0),
    )
    pair = ((sat._triplet_spike_agreement([match["otb4"], match["c3d"]]).get("pairwise") or {}).get("otb4_vs_c3d") or {})
    return {
        "match_id": int(match.get("match_id") or 0),
        "c3d_file": Path(match["c3d"]["path"]).name,
        "otb4_file": Path(match["otb4"]["path"]).name,
        "base_mean_abs_ms": pair.get("mean_abs_ms"),
        "base_max_abs_ms": pair.get("max_abs_ms"),
        "best_candidate": candidates[0] if candidates else None,
        "candidates": candidates,
    }


def analyze_mapping(mapping_path: Path) -> Dict[str, Any]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    reports: List[Dict[str, Any]] = []
    for match in mapping.get("matches", []):
        alignment = match.get("alignment") or {}
        if alignment.get("dedicated_sync_quality") == "poor" or alignment.get("dedicated_sync_pair_quality") == "poor":
            reports.append(analyze_match(match))
    return {"mapping": str(mapping_path), "reports": reports}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze OTB4 gap-repair candidates from the integrated repair logic.")
    parser.add_argument("mapping_json", help="Path to matched_files.json")
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args(argv)

    result = analyze_mapping(Path(args.mapping_json))
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
