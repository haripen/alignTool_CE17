from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "tmp_tracebio_export" / "asf_pppd_sidecar_v2_reexport_validation.json"
ROOT = Path(r"U:\Daten_speziell\Vicon\Eclipse Datenbanken\Projekte\ASF-PPPD")


def _load_tool():
    spec = importlib.util.spec_from_file_location("sync_alignment_tool", REPO_ROOT / "sync_alignment_tool.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load sync_alignment_tool.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sync_alignment_tool"] = module
    spec.loader.exec_module(module)
    return module


sat = _load_tool()


def _finite_float(value: Any) -> Optional[float]:
    parsed = sat._parse_tracebio_scalar(value)
    if isinstance(parsed, (int, float, np.integer, np.floating)):
        out = float(parsed)
        if np.isfinite(out):
            return out
    return None


def _format_json_float(value: Optional[float]) -> Optional[float]:
    if value is None or not np.isfinite(float(value)):
        return None
    return round(float(value), 12)


def _read_accepted(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    matches = data.get("matches") if isinstance(data, dict) else data
    return [m for m in (matches or []) if isinstance(m, dict)]


def _load_tsv_raw_offset(match: Dict[str, Any], selected_signal: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    tsv_path = Path((match.get("tsv") or {})["path"])
    _cols, tsv_df = sat._read_tsv_table(tsv_path)
    raw_column = f"{selected_signal} [raw]"
    offset_column = f"{selected_signal} [offset]"
    if raw_column not in tsv_df.columns or offset_column not in tsv_df.columns:
        raw_column, offset_column = sat._find_tsv_raw_offset_pair(list(tsv_df.columns))
    if raw_column is None or offset_column is None:
        raise ValueError("TSV raw/offset columns for traceBio reconstruction are unavailable.")
    return tsv_df, sat._tsv_numeric_column(tsv_df, raw_column), sat._tsv_numeric_column(tsv_df, offset_column)


def _compute_offset_and_fit(match: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    bundle = sat._build_common_c3d_target(match)
    target_t = np.asarray(bundle["target_t"], dtype=float)

    selected_signal = str(settings.get("SelectedSignal") or "").strip()
    signal_key = selected_signal.lower()
    if "/cop/cy " in signal_key:
        c3d_cop_label = "copy"
    elif "/cop/cx " in signal_key:
        c3d_cop_label = "copx"
    else:
        raise ValueError(f"Unsupported traceBio SelectedSignal: {selected_signal}")

    c3d_path = Path((match.get("c3d") or {})["path"])
    t_cop, cop_m = sat._load_c3d_cop_channel(c3d_path, c3d_cop_label)
    t_cop = np.asarray(t_cop, dtype=float) + float(sat._match_plot_shifts(match).get("c3d", 0.0))
    raw_m = sat._resample_to_target_time(target_t, t_cop, np.asarray(cop_m, dtype=float))

    tsv_df, tsv_raw, tsv_offset = _load_tsv_raw_offset(match, selected_signal)
    tsv_time_aligned: Optional[np.ndarray] = None
    if "t_rel" in tsv_df.columns:
        tsv_time_aligned = pd.to_numeric(tsv_df["t_rel"], errors="coerce").to_numpy(dtype=float)
        tsv_time_aligned = tsv_time_aligned + float(sat._match_plot_shifts(match).get("tsv", 0.0))

    zero_offset_m = sat._tracebio_zero_offset_m(
        tsv_raw,
        tsv_offset,
        raw_m,
        tsv_time_aligned=tsv_time_aligned,
        raw_time_aligned=target_t,
    )
    offset_m = raw_m - float(zero_offset_m)

    fit: Optional[Dict[str, float]] = None
    if sat._tracebio_bool(settings.get("RelativeMode")) and "performed" in tsv_df.columns and "t_rel" in tsv_df.columns:
        tsv_performed = sat._tsv_numeric_column(tsv_df, "performed")
        tsv_t = pd.to_numeric(tsv_df["t_rel"], errors="coerce").to_numpy(dtype=float)
        tsv_t = tsv_t + float(sat._match_plot_shifts(match).get("tsv", 0.0))
        valid_target = np.isfinite(target_t) & np.isfinite(offset_m)
        valid_tsv = np.isfinite(tsv_t) & np.isfinite(tsv_performed)
        if int(np.sum(valid_target)) >= 2 and int(np.sum(valid_tsv)) >= 10:
            target_valid_t = target_t[valid_target]
            target_valid_offset = offset_m[valid_target]
            order = np.argsort(target_valid_t)
            target_valid_t = target_valid_t[order]
            target_valid_offset = target_valid_offset[order]
            in_range = valid_tsv & (tsv_t >= float(target_valid_t[0])) & (tsv_t <= float(target_valid_t[-1]))
            if int(np.sum(in_range)) >= 10:
                offset_at_tsv = np.interp(tsv_t[in_range], target_valid_t, target_valid_offset)
                y = tsv_performed[in_range]
                fit_mask = np.isfinite(offset_at_tsv) & np.isfinite(y)
                if int(np.sum(fit_mask)) >= 10 and float(np.nanstd(offset_at_tsv[fit_mask])) > 1e-12:
                    design = np.column_stack([offset_at_tsv[fit_mask], np.ones(int(np.sum(fit_mask)), dtype=float)])
                    coef, *_ = np.linalg.lstsq(design, y[fit_mask], rcond=None)
                    gain = float(coef[0])
                    bias = float(coef[1])
                    if np.isfinite(gain) and np.isfinite(bias) and gain > 1e-12:
                        active_min = (0.0 - bias) / gain
                        active_max = (100.0 - bias) / gain
                        fit = {
                            "gain": gain,
                            "bias": bias,
                            "active_min": float(active_min),
                            "active_max": float(active_max),
                            "n_fit": int(np.sum(fit_mask)),
                        }

    return {
        "target_t": target_t,
        "offset_m": offset_m,
        "zero_offset_m": float(zero_offset_m),
        "tsv_df": tsv_df,
        "fit": fit,
    }


def _range_source(settings: Dict[str, Any], fit: Dict[str, float]) -> str:
    raw_min = _finite_float(settings.get("RecordMin"))
    raw_max = _finite_float(settings.get("RecordMax"))
    if raw_min is not None and raw_max is not None and raw_max > raw_min:
        raw_span = raw_max - raw_min
        active_span = fit["active_max"] - fit["active_min"]
        if abs(fit["active_min"] - raw_min) <= max(1e-6, 0.02 * raw_span) and abs(active_span - raw_span) <= max(1e-6, 0.02 * raw_span):
            return "recorded_raw"
    return "recorded_smoothed"


def _update_settings_file(path: Path, settings: Dict[str, Any], computed: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    old_text = path.read_text(encoding="utf-8-sig")
    current = json.loads(old_text)
    if not isinstance(current, dict):
        raise ValueError(f"Settings JSON is not an object: {path}")

    relative_mode = sat._tracebio_bool(current.get("RelativeMode"))
    raw_min = _finite_float(current.get("RecordMin"))
    raw_max = _finite_float(current.get("RecordMax"))
    smoothing_method = str(current.get("SmoothingMethod") or "")
    smoothing_frames = _finite_float(current.get("SmoothingFrames"))
    fit = computed.get("fit")

    update: Dict[str, Any] = {
        "PerformedTransformVersion": 2,
        "PerformedInput": "offset_m",
        "PerformedClampMin": 0.0,
        "PerformedClampMax": 100.0,
        "RecordedRawMin": _format_json_float(raw_min),
        "RecordedRawMax": _format_json_float(raw_max),
        "RecordedRangeSmoothingMethod": smoothing_method,
        "RecordedRangeSmoothingFrames": int(round(smoothing_frames)) if smoothing_frames is not None else None,
        "ZeroOffsetM": _format_json_float(computed.get("zero_offset_m")),
    }

    if relative_mode and fit is not None:
        source = _range_source(current, fit)
        update.update(
            {
                "ActiveRelativeRangeMin": _format_json_float(fit["active_min"]),
                "ActiveRelativeRangeMax": _format_json_float(fit["active_max"]),
                "ActiveRelativeRangeSource": source,
                "RecordedSmoothedMin": _format_json_float(fit["active_min"] if source == "recorded_smoothed" else raw_min),
                "RecordedSmoothedMax": _format_json_float(fit["active_max"] if source == "recorded_smoothed" else raw_max),
                "PerformedPercentGain": _format_json_float(fit["gain"]),
                "PerformedPercentBias": _format_json_float(fit["bias"]),
            }
        )
    else:
        update.update(
            {
                "ActiveRelativeRangeMin": None,
                "ActiveRelativeRangeMax": None,
                "ActiveRelativeRangeSource": "none",
                "RecordedSmoothedMin": None,
                "RecordedSmoothedMax": None,
                "PerformedPercentGain": None,
                "PerformedPercentBias": None,
            }
        )

    changed = False
    for key, value in update.items():
        if current.get(key) != value:
            current[key] = value
            changed = True

    if changed:
        path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed, update


def _validate_struct(match: Dict[str, Any], struct: Dict[str, Any]) -> Dict[str, Any]:
    tsv_path = Path((match.get("tsv") or {})["path"])
    _cols, tsv_df = sat._read_tsv_table(tsv_path)
    if "performed" not in tsv_df.columns or "t_rel" not in tsv_df.columns:
        return {"status": "skipped", "reason": "missing performed/t_rel"}
    y_tsv = sat._tsv_numeric_column(tsv_df, "performed")
    tsv_t = pd.to_numeric(tsv_df["t_rel"], errors="coerce").to_numpy(dtype=float)
    start_sec = float(((match.get("alignment") or {}).get("inner_merge") or {})["inner_merge_start_sec"])
    tsv_t_aligned = tsv_t + float(sat._match_plot_shifts(match).get("tsv", 0.0))
    recon_t_aligned = np.asarray(struct["time"], dtype=float).reshape(-1) + start_sec
    recon_y = np.asarray(struct["performed_percent_unsmoothed"], dtype=float).reshape(-1)

    valid_recon = np.isfinite(recon_t_aligned) & np.isfinite(recon_y)
    valid_tsv = np.isfinite(tsv_t_aligned) & np.isfinite(y_tsv)
    if int(np.sum(valid_recon)) < 2 or int(np.sum(valid_tsv)) < 10:
        return {"status": "skipped", "reason": "not enough finite samples"}
    rt = recon_t_aligned[valid_recon]
    ry = recon_y[valid_recon]
    order = np.argsort(rt)
    rt = rt[order]
    ry = ry[order]
    in_range = valid_tsv & (tsv_t_aligned >= float(rt[0])) & (tsv_t_aligned <= float(rt[-1]))
    if int(np.sum(in_range)) < 10:
        return {"status": "skipped", "reason": "not enough overlapping samples"}
    y_at_tsv = np.interp(tsv_t_aligned[in_range], rt, ry)
    diff = y_at_tsv - y_tsv[in_range]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"status": "skipped", "reason": "no finite differences"}
    return {
        "status": "ok",
        "n": int(diff.size),
        "median_abs_error_percent": float(np.nanmedian(np.abs(diff))),
        "p95_abs_error_percent": float(np.nanpercentile(np.abs(diff), 95)),
        "max_abs_error_percent": float(np.nanmax(np.abs(diff))),
        "mean_error_percent": float(np.nanmean(diff)),
        "first3_reconstructed": [float(x) for x in y_at_tsv[:3]],
        "first3_tsv": [float(x) for x in y_tsv[in_range][:3]],
        "peak_reconstructed": float(np.nanmax(y_at_tsv)),
        "peak_tsv": float(np.nanmax(y_tsv[in_range])),
    }


def main() -> int:
    accepted_paths = sorted(ROOT.rglob("accepted_matches.json"))
    report: Dict[str, Any] = {
        "root": str(ROOT),
        "accepted_json_files": [str(p) for p in accepted_paths],
        "updated_json_count": 0,
        "reexported_mat_count": 0,
        "reexported_pipe_count": 0,
        "failures": [],
        "files": [],
    }

    for accepted_path in accepted_paths:
        matches = _read_accepted(accepted_path)
        if not matches:
            continue
        for match in matches:
            if not match.get("tsv") or not match.get("c3d"):
                continue
            item: Dict[str, Any] = {
                "accepted_json": str(accepted_path),
                "match_id": match.get("match_id"),
                "tsv": (match.get("tsv") or {}).get("path"),
                "c3d": (match.get("c3d") or {}).get("path"),
            }
            try:
                settings_path = sat._tracebio_settings_path(match)
                if settings_path is None:
                    raise FileNotFoundError("traceBio settings JSON not found")
                settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
                computed = _compute_offset_and_fit(match, settings)
                changed, update = _update_settings_file(settings_path, settings, computed)
                if changed:
                    report["updated_json_count"] += 1
                item["settings_json"] = str(settings_path)
                item["settings_updated"] = bool(changed)
                item["sidecar_update"] = update

                export_root = Path((((match.get("export_plan") or {}).get("mat") or {}).get("path") or "")).parent
                if not str(export_root):
                    export_root = accepted_path.parent / "matched"
                export_root.mkdir(parents=True, exist_ok=True)
                export_match = copy.deepcopy(match)
                sat._set_tracebio_settings_path_override(export_match, settings_path)
                mat_path = sat.export_match_mat(export_match, export_root)
                report["reexported_mat_count"] += 1
                item["mat"] = str(mat_path)
                if sat._pipe_export_supported(export_match):
                    pipe_mat, pipe_json = sat.export_match_pipe_bundle(export_match, export_root)
                    report["reexported_pipe_count"] += 1
                    item["pipe_mat"] = str(pipe_mat)
                    item["pipe_json"] = str(pipe_json)

                struct = sat._build_tracebio_fullrate_struct(export_match, 0.0, 0.0)
                item["performed_percent_source"] = struct.get("performed_percent_source")
                item["validation"] = _validate_struct(export_match, struct)
                report["files"].append(item)
                print(f"OK match={match.get('match_id')} mat={mat_path.name} source={item['performed_percent_source']}")
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                report["failures"].append(item)
                print(f"FAIL match={match.get('match_id')} {item['error']}")

    ok_validations = [f.get("validation", {}) for f in report["files"] if (f.get("validation") or {}).get("status") == "ok"]
    report["validation_summary"] = {
        "ok_count": len(ok_validations),
        "median_abs_error_percent_median": float(np.nanmedian([v["median_abs_error_percent"] for v in ok_validations])) if ok_validations else None,
        "p95_abs_error_percent_median": float(np.nanmedian([v["p95_abs_error_percent"] for v in ok_validations])) if ok_validations else None,
        "p95_abs_error_percent_max": float(np.nanmax([v["p95_abs_error_percent"] for v in ok_validations])) if ok_validations else None,
        "max_abs_error_percent_max": float(np.nanmax([v["max_abs_error_percent"] for v in ok_validations])) if ok_validations else None,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["validation_summary"], indent=2))
    print(f"report={REPORT_PATH}")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
