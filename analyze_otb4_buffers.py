from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sync_alignment_tool import (
    _find_discrepancy_zones,
    _find_holes_and_jumpbacks,
    _is_counter_track,
    _load_otb4_track_data,
    _to_ascending_ramp,
    load_otb4_file,
)


def _format_datetime(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _file_recorded_datetime(path: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


def _recorded_range_text(results: Sequence["FileAnalysisResult"]) -> str:
    recorded = [result.recorded_at for result in results if result.recorded_at is not None]
    if not recorded:
        return "unknown"
    start = min(recorded)
    end = max(recorded)
    if start == end:
        return _format_datetime(start)
    return f"{_format_datetime(start)} to {_format_datetime(end)}"


REPORT_CONTACT_NAME = "Harald Penasso"
REPORT_CONTACT_EMAIL = "harald.penasso@hcw.ac.at"
REPORT_ACKNOWLEDGEMENT = "Thanks to Christina Knorr for recording the data."
REPORT_LOCATION = "Room C.E.17, Favoritenstrasse 226, University of Applied Sciences Campus Vienna"


def _report_contact_line() -> str:
    return f"{REPORT_CONTACT_NAME} <{REPORT_CONTACT_EMAIL}>"


def _configure_qt_runtime() -> None:
    candidates = []
    conda_prefix = Path(sys.executable).resolve().parent
    candidates.append(conda_prefix / "Library" / "lib" / "qt6" / "plugins")
    try:
        import PySide6  # type: ignore

        candidates.append(Path(PySide6.__file__).resolve().parent / "plugins")
    except Exception:
        pass
    plugin_root = next((path for path in candidates if path.exists() and (path / "platforms").exists()), None)
    if plugin_root is None:
        return
    platform_root = plugin_root / "platforms"
    os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_root))
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platform_root))
    windows_font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    if windows_font_dir.exists():
        os.environ.setdefault("QT_QPA_FONTDIR", str(windows_font_dir))
    dll_paths = [
        conda_prefix / "Library" / "bin",
        conda_prefix / "Library" / "lib" / "qt6" / "bin",
    ]
    current_path = os.environ.get("PATH", "")
    for dll_path in reversed([path for path in dll_paths if path.exists()]):
        text = str(dll_path)
        if text.lower() not in current_path.lower():
            current_path = text + os.pathsep + current_path
    os.environ["PATH"] = current_path


@dataclass
class BufferTrace:
    label: str
    device: str
    subtitle: str
    raw_values: np.ndarray
    values: np.ndarray
    sample_rate: float
    time_sec: np.ndarray
    event_indices: np.ndarray
    event_times_sec: np.ndarray
    is_device_buffer: bool = False
    is_control_buffer: bool = False
    is_aux2: bool = False
    is_ramp: bool = False
    ramp_zones: List[Tuple[int, int]] = field(default_factory=list)
    ramp_samples_added: int = 0


@dataclass
class PatternFit:
    observed_event_count: int
    expected_event_count: int
    expected_times_sec: np.ndarray
    start_interval_sec: Optional[float]
    start_phase_index: Optional[int]
    step_sec: Optional[float]
    min_interval_sec: float
    gap_mae_ms: Optional[float]
    max_gap_error_ms: Optional[float]
    inserted_missing_events: int
    observed_expected_indices: List[int] = field(default_factory=list)
    missing_between: List[int] = field(default_factory=list)
    pattern_observed: bool = False
    note: str = ""


@dataclass
class AlignmentSummary:
    matched_count: int
    unmatched_other: int
    unmatched_ref: int
    offset_sec: Optional[float]
    mean_lag_ms: Optional[float]
    ci95_low_ms: Optional[float]
    ci95_high_ms: Optional[float]
    mean_abs_residual_ms: Optional[float]
    max_abs_residual_ms: Optional[float]
    matched_lags_ms: List[float] = field(default_factory=list)


@dataclass
class DeviceRow:
    device_label: str
    event_count: int
    control_summary: AlignmentSummary
    aux2_summary: Optional[AlignmentSummary]
    reference_summary: AlignmentSummary
    optimal_lag_sec: Optional[float]
    sync_status: str
    sample_status: str
    probe_pattern_observed: bool
    probe_pattern_note: str
    ramp_zone_count: int
    ramp_samples_added: int
    non_sync: bool


@dataclass
class FileAnalysisResult:
    file_path: Path
    out_dir: Path
    plot_pdf_path: Path
    plot_png_path: Path
    report_path: Path
    device_rows: List[DeviceRow]
    file_summary: AlignmentSummary
    file_aux2_summary: Optional[AlignmentSummary]
    reference_file_summary: AlignmentSummary
    control_aux_summary: Optional[AlignmentSummary]
    control_event_count: int
    aux2_event_count: int
    control_expected_event_count: int
    aux2_expected_event_count: int
    reference_source: str
    reference_pattern: PatternFit
    control_pattern: Optional[PatternFit]
    aux2_pattern: Optional[PatternFit]
    control_aux_misaligned: bool
    syncstation_firmware_version: Optional[str] = None
    generated_at: Optional[datetime] = None
    recorded_at: Optional[datetime] = None


@dataclass
class DeviceBundle:
    device: str
    buffer_trace: BufferTrace
    ramp_trace: Optional[BufferTrace]


def _track_prefers_voltage_units(track: Dict[str, Any]) -> bool:
    device = str(track.get("Device", "")).strip().lower()
    subtitle = str(track.get("SubTitle", "")).strip().lower()
    desc_name = str(track.get("DescriptionName", "")).strip().lower()
    sensor_type = str(track.get("SensorType", "")).strip().lower()
    if device == "syncstation" and subtitle.startswith("aux"):
        return True
    if "trigger" in desc_name:
        return True
    if "auxtrigger" in sensor_type:
        return True
    return False


def _track_base_conversion(track: Dict[str, Any]) -> float:
    nbits = int(track.get("ADC_Nbits", 0) or 0)
    gain = float(track.get("Gain", 1) or 1.0)
    adc_range = float(track.get("ADC_Range", 0) or 0.0)
    if nbits <= 0 or gain == 0:
        return 1.0
    return adc_range / (2**nbits) / gain


def _track_is_emg_grid(track: Dict[str, Any]) -> bool:
    grid_info = track.get("GridInfo") or {}
    ied = int(grid_info.get("IED", 0) or 0)
    electrodes = int(track.get("NumberOfChannels", 0) or 0)
    if bool(track.get("IsControl")):
        return False
    if str(track.get("Device", "")).strip().lower() == "syncstation":
        return False
    return ied > 1 or electrodes > 4


def _track_conversion_factor(track: Dict[str, Any]) -> float:
    conv = _track_base_conversion(track)
    if _track_prefers_voltage_units(track):
        return conv
    if _track_is_emg_grid(track):
        return conv * 1000.0
    return conv


def _normalize_for_plot(values: np.ndarray) -> np.ndarray:
    x = np.abs(np.asarray(values, dtype=float).reshape(-1))
    x = np.nan_to_num(x, nan=0.0)
    scale = float(np.nanmax(x)) if x.size else 0.0
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(x)
    return np.clip(x / scale, 0.0, 1.0)


def _normalize_unit_interval(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.any(np.isfinite(x)) else 0.0)
    lo = float(np.nanmin(x)) if x.size else 0.0
    hi = float(np.nanmax(x)) if x.size else 0.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _detect_buffer_events(values: np.ndarray, sample_rate: float) -> np.ndarray:
    x = np.abs(np.asarray(values, dtype=float).reshape(-1))
    if x.size == 0:
        return np.asarray([], dtype=int)
    x = np.nan_to_num(x, nan=0.0)
    q99 = float(np.nanpercentile(x, 99.0))
    max_abs = float(np.nanmax(x))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return np.asarray([], dtype=int)
    if max_abs <= q99 * 4.0:
        return np.asarray([], dtype=int)
    threshold = max(max_abs * 0.5, float(np.nanpercentile(x, 99.99)))
    candidates = np.flatnonzero(x >= threshold)
    if candidates.size == 0:
        return np.asarray([], dtype=int)
    min_gap = max(1, int(round(float(sample_rate or 2000.0) * 0.15)))
    kept: List[int] = [int(candidates[0])]
    for idx in candidates[1:]:
        if int(idx) - kept[-1] >= min_gap:
            kept.append(int(idx))
    return np.asarray(kept, dtype=int)


def _schedule_interval_sec(start_interval_sec: float, step_sec: float, idx: int, min_interval_sec: float) -> float:
    start = float(start_interval_sec)
    step = float(step_sec)
    minimum = float(min_interval_sec)
    if step <= 0.0 or start <= minimum:
        return max(minimum, start)
    cycle_len = max(1, int(round((start - minimum) / step)) + 1)
    phase = int(idx) % cycle_len
    return max(minimum, start - step * float(phase))


def _schedule_gap_sec(start_interval_sec: float, step_sec: float, start_idx: int, span: int, min_interval_sec: float) -> float:
    return float(
        sum(
            _schedule_interval_sec(start_interval_sec, step_sec, start_idx + offset, min_interval_sec)
            for offset in range(int(span))
        )
    )


def _extend_expected_times_cyclic(
    expected_times_sec: Sequence[float],
    *,
    start_interval_sec: Optional[float],
    start_phase_index: Optional[int],
    step_sec: Optional[float],
    min_interval_sec: float,
    target_start_sec: Optional[float] = None,
    target_end_sec: Optional[float] = None,
) -> np.ndarray:
    expected = [float(v) for v in expected_times_sec]
    if not expected:
        return np.asarray([], dtype=float)
    if not isinstance(start_interval_sec, (int, float)) or not isinstance(start_phase_index, int) or not isinstance(step_sec, (int, float)):
        return np.asarray(expected, dtype=float)
    start_interval = float(start_interval_sec)
    step = float(step_sec)
    phase0 = int(start_phase_index)
    slack_sec = max(0.2, 2.0 * abs(step))
    intervals_used = max(0, len(expected) - 1)
    next_idx = phase0 + intervals_used
    while target_end_sec is not None:
        next_time = float(expected[-1]) + _schedule_interval_sec(start_interval, step, next_idx, min_interval_sec)
        if next_time > float(target_end_sec) + slack_sec:
            break
        expected.append(next_time)
        next_idx += 1
    prev_idx = phase0 - 1
    while target_start_sec is not None and prev_idx >= 0:
        prev_time = float(expected[0]) - _schedule_interval_sec(start_interval, step, prev_idx, min_interval_sec)
        if prev_time < float(target_start_sec) - slack_sec:
            break
        expected.insert(0, prev_time)
        prev_idx -= 1
    return np.asarray(expected, dtype=float)


def _fit_event_pattern(
    event_times_sec: Sequence[float],
    *,
    default_step_sec: float = 0.10,
    candidate_step_secs: Optional[Sequence[float]] = None,
    min_interval_sec: float = 3.0,
    max_missing_between: int = 4,
    cycle_start_sec: float = 5.0,
) -> PatternFit:
    observed = np.asarray(event_times_sec, dtype=float).reshape(-1)
    if observed.size == 0:
        return PatternFit(
            observed_event_count=0,
            expected_event_count=0,
            expected_times_sec=np.asarray([], dtype=float),
            start_interval_sec=None,
            start_phase_index=None,
            step_sec=None,
            min_interval_sec=min_interval_sec,
            gap_mae_ms=None,
            max_gap_error_ms=None,
            inserted_missing_events=0,
            pattern_observed=False,
            note="no spikes detected",
        )
    if observed.size == 1:
        return PatternFit(
            observed_event_count=1,
            expected_event_count=1,
            expected_times_sec=observed.copy(),
            start_interval_sec=None,
            start_phase_index=None,
            step_sec=None,
            min_interval_sec=min_interval_sec,
            gap_mae_ms=None,
            max_gap_error_ms=None,
            inserted_missing_events=0,
            observed_expected_indices=[0],
            pattern_observed=False,
            note="single spike only",
        )

    gaps = np.diff(observed)
    start_interval_sec = float(cycle_start_sec)
    step_candidates = [float(v) for v in (candidate_step_secs or (0.05, float(default_step_sec))) if float(v) > 0.0]
    if not step_candidates:
        step_candidates = [float(default_step_sec or 0.10)]
    step_candidates = sorted({round(v, 6) for v in step_candidates})
    best: Optional[Tuple[Tuple[float, float, int, int, float], float, int, List[int], List[float]]] = None
    for selected_step_sec in step_candidates:
        max_phase_idx = max(0, int(round(max(0.0, float(start_interval_sec) - float(min_interval_sec)) / float(selected_step_sec or 0.10))))
        for start_phase_idx in range(0, max_phase_idx + 1):
            gap_errors = []
            missing_between = []
            schedule_idx = int(start_phase_idx)
            for gap_sec in gaps:
                best_local: Optional[Tuple[float, int]] = None
                for missing_count in range(0, max_missing_between + 1):
                    predicted_gap_sec = _schedule_gap_sec(
                        start_interval_sec,
                        float(selected_step_sec),
                        schedule_idx,
                        missing_count + 1,
                        min_interval_sec,
                    )
                    cost = abs(predicted_gap_sec - float(gap_sec))
                    cand = (cost, missing_count)
                    if best_local is None or cand < best_local:
                        best_local = cand
                assert best_local is not None
                gap_errors.append(float(best_local[0]))
                missing_between.append(int(best_local[1]))
                schedule_idx += int(best_local[1]) + 1
            mae_ms = float(np.mean(gap_errors) * 1000.0)
            max_ms = float(np.max(gap_errors) * 1000.0)
            score = (
                mae_ms,
                max_ms,
                int(sum(missing_between)),
                int(start_phase_idx),
                abs(float(selected_step_sec) - float(default_step_sec)),
            )
            if best is None or score < best[0]:
                best = (score, float(selected_step_sec), int(start_phase_idx), missing_between, gap_errors)

    assert best is not None
    _score, selected_step_sec, start_phase_idx, missing_between, gap_errors = best
    expected_times: List[float] = [float(observed[0])]
    observed_expected_indices: List[int] = [0]
    schedule_idx = int(start_phase_idx)
    current_time = float(observed[0])
    for missing_count in missing_between:
        for _ in range(int(missing_count) + 1):
            current_time += _schedule_interval_sec(start_interval_sec, selected_step_sec, schedule_idx, min_interval_sec)
            expected_times.append(current_time)
            schedule_idx += 1
        observed_expected_indices.append(len(expected_times) - 1)

    gap_mae_ms = float(np.mean(gap_errors) * 1000.0)
    max_gap_error_ms = float(np.max(gap_errors) * 1000.0)
    pattern_observed = bool(gap_mae_ms <= 80.0 and max_gap_error_ms <= 160.0)
    note = (
        f"fixed cyclic 5.00->3.00 s pattern observed (step {int(round(selected_step_sec * 1000.0))} ms, phase {int(start_phase_idx)})"
        if pattern_observed
        else f"fixed cyclic 5.00->3.00 s pattern not cleanly observed (step {int(round(selected_step_sec * 1000.0))} ms, gap MAE {gap_mae_ms:.1f} ms, phase {int(start_phase_idx)})"
    )
    return PatternFit(
        observed_event_count=int(observed.size),
        expected_event_count=len(expected_times),
        expected_times_sec=np.asarray(expected_times, dtype=float),
        start_interval_sec=float(start_interval_sec),
        start_phase_index=int(start_phase_idx),
        step_sec=float(selected_step_sec),
        min_interval_sec=min_interval_sec,
        gap_mae_ms=gap_mae_ms,
        max_gap_error_ms=max_gap_error_ms,
        inserted_missing_events=int(sum(missing_between)),
        observed_expected_indices=observed_expected_indices,
        missing_between=[int(v) for v in missing_between],
        pattern_observed=pattern_observed,
        note=note,
    )


def _expected_times_from_pattern(pattern: Optional[PatternFit], fallback_times_sec: Sequence[float]) -> np.ndarray:
    if pattern is not None and pattern.expected_times_sec.size:
        return np.asarray(pattern.expected_times_sec, dtype=float)
    return np.asarray(fallback_times_sec, dtype=float).reshape(-1)


def _expected_times_from_pattern_window(
    pattern: Optional[PatternFit],
    fallback_times_sec: Sequence[float],
    *,
    target_start_sec: Optional[float] = None,
    target_end_sec: Optional[float] = None,
) -> np.ndarray:
    if pattern is not None and pattern.expected_times_sec.size:
        return _extend_expected_times_cyclic(
            pattern.expected_times_sec,
            start_interval_sec=pattern.start_interval_sec,
            start_phase_index=pattern.start_phase_index,
            step_sec=pattern.step_sec,
            min_interval_sec=pattern.min_interval_sec,
            target_start_sec=target_start_sec,
            target_end_sec=target_end_sec,
        )
    return np.asarray(fallback_times_sec, dtype=float).reshape(-1)


def _pattern_quality_key(pattern: Optional[PatternFit], label: str) -> Tuple[int, float, float, int, int]:
    if pattern is None:
        return (1, float("inf"), float("inf"), 1, 1)
    return (
        0 if pattern.pattern_observed else 1,
        float(pattern.gap_mae_ms if pattern.gap_mae_ms is not None else float("inf")),
        float(pattern.max_gap_error_ms if pattern.max_gap_error_ms is not None else float("inf")),
        -int(pattern.expected_event_count),
        0 if label == "control" else 1,
    )


def _select_reference_pattern(
    control_buffer: Optional[BufferTrace],
    aux2_trace: Optional[BufferTrace],
) -> Tuple[str, PatternFit, Optional[PatternFit], Optional[PatternFit]]:
    control_pattern = _fit_event_pattern(control_buffer.event_times_sec) if control_buffer is not None else None
    aux2_pattern = _fit_event_pattern(aux2_trace.event_times_sec) if aux2_trace is not None else None
    candidates: List[Tuple[str, PatternFit]] = []
    if control_pattern is not None:
        candidates.append(("control", control_pattern))
    if aux2_pattern is not None:
        candidates.append(("aux2", aux2_pattern))
    if not candidates:
        empty = _fit_event_pattern([])
        return ("none", empty, control_pattern, aux2_pattern)
    selected_label, selected_pattern = min(candidates, key=lambda item: _pattern_quality_key(item[1], item[0]))
    return selected_label, selected_pattern, control_pattern, aux2_pattern


def _ramp_sample_status(ramp_trace: Optional[BufferTrace]) -> Tuple[str, int, int]:
    if ramp_trace is None:
        return ("no ramp available", 0, 0)
    zone_count = len(ramp_trace.ramp_zones)
    sample_count = int(ramp_trace.ramp_samples_added)
    if zone_count <= 0 or sample_count <= 0:
        return ("all samples present (ramp clean)", 0, 0)
    return (f"repairable from ramp (+{sample_count} samples across {zone_count} zone(s))", zone_count, sample_count)


def _sync_status_from_summary(summary: AlignmentSummary) -> Tuple[str, bool]:
    if summary.matched_count <= 0:
        return ("non-sync", True)
    if summary.unmatched_ref == 0 and summary.unmatched_other == 0 and (summary.mean_abs_residual_ms or 0.0) <= 25.0:
        return ("synced", False)
    if summary.unmatched_ref <= 1 and summary.unmatched_other == 0:
        return ("deviating", True)
    return ("non-sync", True)


def _ci95_bounds(values_ms: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    arr = np.asarray(values_ms, dtype=float).reshape(-1)
    if arr.size == 0:
        return None, None, None
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    half = 1.96 * sem
    return mean, mean - half, mean + half


def _best_event_alignment(
    other_times_sec: Sequence[float],
    ref_times_sec: Sequence[float],
    tolerance_sec: float = 0.15,
    *,
    trace_start_sec: Optional[float] = None,
    trace_end_sec: Optional[float] = None,
    edge_guard_sec: Optional[float] = None,
) -> AlignmentSummary:
    other = np.asarray(other_times_sec, dtype=float).reshape(-1)
    ref = np.asarray(ref_times_sec, dtype=float).reshape(-1)
    if other.size == 0 or ref.size == 0:
        return AlignmentSummary(
            matched_count=0,
            unmatched_other=int(other.size),
            unmatched_ref=int(ref.size),
            offset_sec=None,
            mean_lag_ms=None,
            ci95_low_ms=None,
            ci95_high_ms=None,
            mean_abs_residual_ms=None,
            max_abs_residual_ms=None,
        )

    candidates = np.unique(np.round(np.append(0.0, (other[:, None] - ref[None, :]).reshape(-1)), 6))
    best_score: Optional[Tuple[int, float, float]] = None
    best_offset: Optional[float] = None
    best_lags_ms: List[float] = []
    best_abs_residual_ms: List[float] = []
    best_unmatched_other = int(other.size)
    best_unmatched_ref = int(ref.size)
    guard_sec = float(edge_guard_sec if edge_guard_sec is not None else max(0.02, tolerance_sec))

    for offset in candidates:
        shifted = other - float(offset)
        effective_ref = ref
        if trace_start_sec is not None and trace_end_sec is not None:
            shifted_start = float(trace_start_sec) - float(offset)
            shifted_end = float(trace_end_sec) - float(offset)
            mask = (ref >= (shifted_start - guard_sec)) & (ref <= (shifted_end + guard_sec))
            effective_ref = ref[mask]
            if effective_ref.size == 0:
                continue
        i = 0
        j = 0
        lags_ms: List[float] = []
        abs_residual_ms: List[float] = []
        while i < other.size and j < effective_ref.size:
            delta = float(shifted[i] - effective_ref[j])
            if abs(delta) <= tolerance_sec:
                raw_lag_sec = float(other[i] - effective_ref[j])
                lags_ms.append(raw_lag_sec * 1000.0)
                abs_residual_ms.append(abs(delta) * 1000.0)
                i += 1
                j += 1
                continue
            if delta < -tolerance_sec:
                i += 1
            else:
                j += 1
        unmatched_other = int(other.size - len(lags_ms))
        unmatched_ref = int(effective_ref.size - len(lags_ms))
        score = (len(lags_ms), -unmatched_ref, -unmatched_other, -float(sum(abs_residual_ms)), -abs(float(offset)))
        if best_score is None or score > best_score:
            best_score = score
            best_offset = float(offset)
            best_lags_ms = lags_ms
            best_abs_residual_ms = abs_residual_ms
            best_unmatched_other = unmatched_other
            best_unmatched_ref = unmatched_ref

    mean_lag_ms, ci95_low_ms, ci95_high_ms = _ci95_bounds(best_lags_ms)
    return AlignmentSummary(
        matched_count=len(best_lags_ms),
        unmatched_other=best_unmatched_other,
        unmatched_ref=best_unmatched_ref,
        offset_sec=best_offset,
        mean_lag_ms=mean_lag_ms,
        ci95_low_ms=ci95_low_ms,
        ci95_high_ms=ci95_high_ms,
        mean_abs_residual_ms=float(np.mean(best_abs_residual_ms)) if best_abs_residual_ms else None,
        max_abs_residual_ms=float(np.max(best_abs_residual_ms)) if best_abs_residual_ms else None,
        matched_lags_ms=best_lags_ms,
    )


def _shifted_trace_time(trace: BufferTrace, lag_sec: Optional[float]) -> np.ndarray:
    offset = float(lag_sec or 0.0)
    return np.asarray(trace.time_sec, dtype=float) - offset


def _select_time_axis(path: Path, sample_count: int, sample_rate: float) -> Tuple[np.ndarray, bool]:
    try:
        _data, time_vec, _descs, _fs, _name, _size = load_otb4_file(str(path))
        time_arr = np.asarray(time_vec, dtype=float).reshape(-1)
        if time_arr.size == sample_count and np.all(np.isfinite(time_arr)):
            return time_arr, True
    except Exception:
        pass
    if sample_rate > 0:
        return np.arange(sample_count, dtype=float) / float(sample_rate), False
    return np.arange(sample_count, dtype=float), False


def _load_buffer_traces(path: Path) -> Tuple[List[BufferTrace], Dict[str, Any]]:
    track_traces: List[BufferTrace] = []
    seen_labels: set[str] = set()
    track_data = _load_otb4_track_data(path)
    sample_count = int(track_data[0][2].shape[1]) if track_data else 0
    sample_rate = float(track_data[0][1].get("SamplingFrequency") or 0.0) if track_data else 0.0
    time_sec, has_saved_time = _select_time_axis(path, sample_count, sample_rate)

    for _offset, track, raw_arr in track_data:
        nchan = int(track.get("NumberOfChannels", 0) or 0)
        if nchan != 1:
            continue
        device = str(track.get("Device", "")).strip()
        subtitle = str(track.get("SubTitle", "")).strip()
        raw = np.asarray(raw_arr[0], dtype=float).reshape(-1)
        converted = raw * float(_track_conversion_factor(track))
        is_ramp = _is_counter_track(track, raw)
        is_aux2 = device == "Syncstation" and subtitle == "AUX 2"
        is_control_buffer = device == "Syncstation" and bool(track.get("IsControl")) and not subtitle and not is_ramp
        is_device_buffer = device != "Syncstation" and subtitle == "Buffer" and bool(track.get("IsControl"))
        is_device_ramp = device != "Syncstation" and subtitle == "Ramp" and bool(track.get("IsControl"))
        is_control_ramp = device == "Syncstation" and bool(track.get("IsControl")) and is_ramp
        if not (is_aux2 or is_control_buffer or is_device_buffer or is_device_ramp or is_control_ramp):
            continue
        if is_control_buffer:
            label = "Syncstation Control Buffer"
        elif is_control_ramp:
            label = "Syncstation Ramp"
        elif is_aux2:
            label = "Syncstation AUX 2 [V]"
        elif is_device_ramp:
            label = f"{device} Ramp"
        else:
            label = f"{device} Buffer"
        if label in seen_labels:
            continue
        seen_labels.add(label)
        events = _detect_buffer_events(raw, sample_rate) if not (is_ramp or is_device_ramp or is_control_ramp) else np.asarray([], dtype=int)
        event_times = time_sec[events] if events.size and time_sec.size else np.asarray([], dtype=float)
        ramp_zones: List[Tuple[int, int]] = []
        ramp_samples_added = 0
        if is_ramp or is_device_ramp or is_control_ramp:
            holes = _find_holes_and_jumpbacks(_to_ascending_ramp(raw))
            ramp_zones = [(int(start), int(length)) for start, length in _find_discrepancy_zones(holes["cleaned_ramp"]) if int(length) > 0]
            ramp_samples_added = int(sum(length for _start, length in ramp_zones))
        track_traces.append(
            BufferTrace(
                label=label,
                device=device,
                subtitle=subtitle,
                raw_values=raw,
                values=converted,
                sample_rate=sample_rate,
                time_sec=np.asarray(time_sec, dtype=float),
                event_indices=events,
                event_times_sec=np.asarray(event_times, dtype=float),
                is_device_buffer=is_device_buffer,
                is_control_buffer=is_control_buffer,
                is_aux2=is_aux2,
                is_ramp=bool(is_ramp or is_device_ramp or is_control_ramp),
                ramp_zones=ramp_zones,
                ramp_samples_added=ramp_samples_added,
            )
        )

    info = {
        "sample_count": sample_count,
        "sample_rate": sample_rate,
        "has_saved_time": has_saved_time,
    }
    return track_traces, info


def _first_or_none(items: Sequence[BufferTrace], predicate) -> Optional[BufferTrace]:
    for item in items:
        if predicate(item):
            return item
    return None


def _build_device_bundles(traces: Sequence[BufferTrace]) -> List[DeviceBundle]:
    buffers = {
        trace.device: trace
        for trace in traces
        if trace.is_device_buffer
    }
    ramps = {
        trace.device: trace
        for trace in traces
        if trace.is_ramp and trace.device != "Syncstation"
    }
    bundles: List[DeviceBundle] = []
    for device in sorted(buffers):
        bundles.append(DeviceBundle(device=device, buffer_trace=buffers[device], ramp_trace=ramps.get(device)))
    return bundles


def _format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _format_ci(summary: AlignmentSummary, digits: int = 2) -> str:
    if summary.mean_lag_ms is None:
        return "NA"
    return f"[{_format_float(summary.ci95_low_ms, digits)}, {_format_float(summary.ci95_high_ms, digits)}]"


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def _count_pct_text(count: int, total: int, digits: int = 1) -> str:
    return f"{count} ({_pct(count, total):.{digits}f}%)"


def _control_aux_misaligned(control_aux_summary: Optional[AlignmentSummary], control_count: int, aux2_count: int, residual_tol_ms: float = 1.0) -> bool:
    if control_aux_summary is None:
        return True
    if int(control_aux_summary.matched_count) != min(int(control_count), int(aux2_count)):
        return True
    if int(control_aux_summary.unmatched_ref) > 0 or int(control_aux_summary.unmatched_other) > 0:
        return True
    if (control_aux_summary.mean_abs_residual_ms or 0.0) > residual_tol_ms:
        return True
    return False


def _make_device_rows(
    device_bundles: Sequence[DeviceBundle],
    *,
    control_expected_times_sec: Sequence[float],
    aux2_expected_times_sec: Sequence[float],
    reference_expected_times_sec: Sequence[float],
) -> List[DeviceRow]:
    rows: List[DeviceRow] = []
    control_expected = np.asarray(control_expected_times_sec, dtype=float).reshape(-1)
    aux2_expected = np.asarray(aux2_expected_times_sec, dtype=float).reshape(-1)
    reference_expected = np.asarray(reference_expected_times_sec, dtype=float).reshape(-1)
    for bundle in device_bundles:
        trace = bundle.buffer_trace
        trace_start = float(trace.time_sec[0]) if trace.time_sec.size else None
        trace_end = float(trace.time_sec[-1]) if trace.time_sec.size else None
        control_summary = _best_event_alignment(
            trace.event_times_sec,
            control_expected,
            trace_start_sec=trace_start,
            trace_end_sec=trace_end,
        ) if control_expected.size else _best_event_alignment([], [])
        aux2_summary = _best_event_alignment(
            trace.event_times_sec,
            aux2_expected,
            trace_start_sec=trace_start,
            trace_end_sec=trace_end,
        ) if aux2_expected.size else None
        reference_summary = _best_event_alignment(
            trace.event_times_sec,
            reference_expected,
            trace_start_sec=trace_start,
            trace_end_sec=trace_end,
        ) if reference_expected.size else _best_event_alignment([], [])
        probe_pattern = _fit_event_pattern(trace.event_times_sec)
        sample_status, ramp_zone_count, ramp_samples_added = _ramp_sample_status(bundle.ramp_trace)
        sync_status, non_sync = _sync_status_from_summary(reference_summary)
        rows.append(
            DeviceRow(
                device_label=trace.label,
                event_count=int(trace.event_times_sec.size),
                control_summary=control_summary,
                aux2_summary=aux2_summary,
                reference_summary=reference_summary,
                optimal_lag_sec=reference_summary.offset_sec,
                sync_status=sync_status,
                sample_status=sample_status,
                probe_pattern_observed=bool(probe_pattern.pattern_observed),
                probe_pattern_note=probe_pattern.note,
                ramp_zone_count=int(ramp_zone_count),
                ramp_samples_added=int(ramp_samples_added),
                non_sync=bool(non_sync),
            )
        )
    return rows


def _pool_alignment_summaries(summaries: Sequence[AlignmentSummary]) -> AlignmentSummary:
    matched_lags_ms: List[float] = []
    matched_count = 0
    unmatched_other = 0
    unmatched_ref = 0
    residuals_ms: List[float] = []
    for summary in summaries:
        matched_count += int(summary.matched_count)
        unmatched_other += int(summary.unmatched_other)
        unmatched_ref += int(summary.unmatched_ref)
        matched_lags_ms.extend(float(v) for v in summary.matched_lags_ms)
        if summary.mean_abs_residual_ms is not None and summary.matched_count > 0:
            residuals_ms.extend([float(summary.mean_abs_residual_ms)] * int(summary.matched_count))
    mean_lag_ms, ci95_low_ms, ci95_high_ms = _ci95_bounds(matched_lags_ms)
    return AlignmentSummary(
        matched_count=matched_count,
        unmatched_other=unmatched_other,
        unmatched_ref=unmatched_ref,
        offset_sec=None,
        mean_lag_ms=mean_lag_ms,
        ci95_low_ms=ci95_low_ms,
        ci95_high_ms=ci95_high_ms,
        mean_abs_residual_ms=float(np.mean(residuals_ms)) if residuals_ms else None,
        max_abs_residual_ms=float(np.max(residuals_ms)) if residuals_ms else None,
        matched_lags_ms=matched_lags_ms,
    )


def _reference_rollup(result: FileAnalysisResult, reference: str = "control") -> Dict[str, Any]:
    if reference == "control":
        ref_count = int(result.control_expected_event_count)
        summary = result.file_summary
    else:
        ref_count = int(result.aux2_expected_event_count)
        summary = result.file_aux2_summary or AlignmentSummary(0, 0, 0, None, None, None, None, None, None, [])
    probe_count = len(result.device_rows)
    return {
        "reference": reference,
        "ref_count": ref_count,
        "probe_count": probe_count,
        "ideal_total": probe_count * ref_count,
        "detected_total": sum(int(row.event_count) for row in result.device_rows),
        "aligned_total": int(summary.matched_count),
        "missing_total": int(summary.unmatched_ref),
        "extra_total": int(summary.unmatched_other),
        "mean_lag_ms": summary.mean_lag_ms,
        "ci95": _format_ci(summary, 2),
        "mean_abs_residual_ms": summary.mean_abs_residual_ms,
        "aligned_pct": _pct(int(summary.matched_count), probe_count * ref_count),
        "missing_pct": _pct(int(summary.unmatched_ref), probe_count * ref_count),
        "misaligned_aux2_control": bool(result.control_aux_misaligned),
    }


def _probe_rollup(results: Sequence[FileAnalysisResult], device_label: str, reference: str = "control") -> Dict[str, Any]:
    ref_total = 0
    detected_total = 0
    summaries: List[AlignmentSummary] = []
    files_with_events = 0
    for result in results:
        row = next((row for row in result.device_rows if row.device_label == device_label), None)
        if row is None:
            continue
        ref_total += int(result.control_expected_event_count if reference == "control" else result.aux2_expected_event_count)
        detected_total += int(row.event_count)
        if row.event_count > 0:
            files_with_events += 1
        summary = row.control_summary if reference == "control" else row.aux2_summary
        if summary is not None:
            summaries.append(summary)
    pooled = _pool_alignment_summaries(summaries)
    return {
        "device_label": device_label,
        "files_with_events": files_with_events,
        "ideal_total": ref_total,
        "detected_total": detected_total,
        "aligned_total": int(pooled.matched_count),
        "missing_total": int(pooled.unmatched_ref),
        "extra_total": int(pooled.unmatched_other),
        "mean_lag_ms": pooled.mean_lag_ms,
        "ci95": _format_ci(pooled, 2),
        "mean_abs_residual_ms": pooled.mean_abs_residual_ms,
        "aligned_pct": _pct(int(pooled.matched_count), ref_total),
        "missing_pct": _pct(int(pooled.unmatched_ref), ref_total),
        "misaligned_file_count": sum(1 for result in results if result.control_aux_misaligned),
        "summary": pooled,
    }


def _save_report(
    out_path: Path,
    *,
    file_path: Path,
    syncstation_firmware_version: Optional[str],
    generated_at: Optional[datetime],
    recorded_at: Optional[datetime],
    device_rows: Sequence[DeviceRow],
    file_summary: AlignmentSummary,
    file_aux2_summary: Optional[AlignmentSummary],
    reference_file_summary: AlignmentSummary,
    control_aux_summary: Optional[AlignmentSummary],
    control_event_count: int,
    aux2_event_count: int,
    control_expected_event_count: int,
    aux2_expected_event_count: int,
    reference_source: str,
    reference_pattern: PatternFit,
    control_pattern: Optional[PatternFit],
    aux2_pattern: Optional[PatternFit],
    info: Dict[str, Any],
) -> None:
    probe_count = len(device_rows)
    control_ideal_total = probe_count * int(control_expected_event_count)
    aux2_ideal_total = probe_count * int(aux2_expected_event_count)
    control_aux_flag = "YES" if _control_aux_misaligned(control_aux_summary, int(control_event_count), int(aux2_event_count)) else "NO"
    reference_ideal_total = probe_count * int(reference_pattern.expected_event_count)
    non_sync_rows = [row for row in device_rows if row.non_sync]
    lines: List[str] = [
        f"file={file_path}",
        f"syncstation_firmware_version={syncstation_firmware_version or 'unknown'}",
        f"generated_at={_format_datetime(generated_at)}",
        f"data_recorded_at={_format_datetime(recorded_at)}",
        f"contact={_report_contact_line()}",
        f"acknowledgement={REPORT_ACKNOWLEDGEMENT}",
        f"location={REPORT_LOCATION}",
        f"device_specific_buffer_channels={probe_count}",
        f"sample_rate_hz={info.get('sample_rate')}",
        f"saved_time_vector_present={bool(info.get('has_saved_time'))}",
        f"sample_count={info.get('sample_count')}",
        f"global_control_buffer_events_observed={control_event_count}",
        f"global_control_buffer_events_expected={control_expected_event_count}",
        f"aux2_events_observed={aux2_event_count}",
        f"aux2_events_expected={aux2_expected_event_count}",
        "",
        "[Reference Pattern]",
        f"selected_reference_source={reference_source}",
        f"reference_pattern_observed={'YES' if reference_pattern.pattern_observed else 'NO'}",
        f"reference_expected_event_count={reference_pattern.expected_event_count}",
        f"reference_observed_event_count={reference_pattern.observed_event_count}",
        f"reference_start_interval_sec={_format_float(reference_pattern.start_interval_sec, 3)}",
        f"reference_step_ms={_format_float(reference_pattern.step_sec * 1000.0 if reference_pattern.step_sec is not None else None, 1)}",
        f"reference_gap_mae_ms={_format_float(reference_pattern.gap_mae_ms, 2)}",
        f"reference_gap_max_error_ms={_format_float(reference_pattern.max_gap_error_ms, 2)}",
        f"reference_missing_inserted={reference_pattern.inserted_missing_events}",
        f"reference_note={reference_pattern.note}",
        "",
        "[Ideal Summary]",
        f"ideal_total_probe_events_vs_selected_reference={reference_ideal_total}",
        f"aligned_total_vs_selected_reference={_count_pct_text(int(reference_file_summary.matched_count), reference_ideal_total)}",
        f"missing_total_vs_selected_reference={_count_pct_text(int(reference_file_summary.unmatched_ref), reference_ideal_total)}",
        f"extra_total_vs_selected_reference={reference_file_summary.unmatched_other}",
        f"mean_lag_vs_selected_reference_ms={_format_float(reference_file_summary.mean_lag_ms, 3)}",
        f"ideal_total_probe_events_vs_control={control_ideal_total}",
        f"aligned_total_vs_control={_count_pct_text(int(file_summary.matched_count), control_ideal_total)}",
        f"missing_total_vs_control={_count_pct_text(int(file_summary.unmatched_ref), control_ideal_total)}",
        f"extra_total_vs_control={file_summary.unmatched_other}",
        f"mean_lag_vs_control_ms={_format_float(file_summary.mean_lag_ms, 3)}",
        f"lag_vs_control_95ci_ms={_format_ci(file_summary, 3)}",
        f"mean_abs_residual_vs_control_ms={_format_float(file_summary.mean_abs_residual_ms, 3)}",
        f"ideal_total_probe_events_vs_aux2={aux2_ideal_total}",
        f"aligned_total_vs_aux2={_count_pct_text(int(file_aux2_summary.matched_count), aux2_ideal_total) if file_aux2_summary is not None else 'NA'}",
        f"missing_total_vs_aux2={_count_pct_text(int(file_aux2_summary.unmatched_ref), aux2_ideal_total) if file_aux2_summary is not None else 'NA'}",
        f"extra_total_vs_aux2={file_aux2_summary.unmatched_other if file_aux2_summary is not None else 'NA'}",
        f"mean_lag_vs_aux2_ms={_format_float(file_aux2_summary.mean_lag_ms, 3) if file_aux2_summary is not None else 'NA'}",
        f"lag_vs_aux2_95ci_ms={_format_ci(file_aux2_summary, 3) if file_aux2_summary is not None else 'NA'}",
        f"mean_abs_residual_vs_aux2_ms={_format_float(file_aux2_summary.mean_abs_residual_ms, 3) if file_aux2_summary is not None else 'NA'}",
        f"non_sync_probe_count={len(non_sync_rows)}",
        "",
    ]
    if file_aux2_summary is not None:
        lines.extend(
            [
                "[Reference Agreement]",
                f"control_events_observed={control_event_count}",
                f"aux2_events_observed={aux2_event_count}",
                f"control_aux2_misaligned={control_aux_flag}",
                f"control_aux2_aligned={control_aux_summary.matched_count if control_aux_summary is not None else 'NA'}",
                f"control_aux2_missing_in_aux2={control_aux_summary.unmatched_ref if control_aux_summary is not None else 'NA'}",
                f"control_aux2_extra_in_aux2={control_aux_summary.unmatched_other if control_aux_summary is not None else 'NA'}",
                f"control_aux2_mean_lag_ms={_format_float(control_aux_summary.mean_lag_ms, 3) if control_aux_summary is not None else 'NA'}",
                f"control_pattern_note={control_pattern.note if control_pattern is not None else 'NA'}",
                f"aux2_pattern_note={aux2_pattern.note if aux2_pattern is not None else 'NA'}",
                "",
            ]
        )

    sorted_rows = sorted(
        device_rows,
        key=lambda row: (-_pct(int(row.reference_summary.matched_count), int(reference_pattern.expected_event_count)), row.device_label.lower()),
    )
    for row in sorted_rows:
        ref_aligned = _count_pct_text(int(row.reference_summary.matched_count), int(reference_pattern.expected_event_count))
        ref_missing = _count_pct_text(int(row.reference_summary.unmatched_ref), int(reference_pattern.expected_event_count))
        control_aligned = _count_pct_text(int(row.control_summary.matched_count), int(control_expected_event_count))
        control_missing = _count_pct_text(int(row.control_summary.unmatched_ref), int(control_expected_event_count))
        aux2_aligned = _count_pct_text(int(row.aux2_summary.matched_count), int(aux2_expected_event_count)) if row.aux2_summary is not None else "NA"
        aux2_missing = _count_pct_text(int(row.aux2_summary.unmatched_ref), int(aux2_expected_event_count)) if row.aux2_summary is not None else "NA"
        lines.extend(
            [
                f"[{row.device_label}]",
                f"sync_status={row.sync_status}",
                f"sample_status={row.sample_status}",
                f"pattern_observed_in_probe_spikes={'YES' if row.probe_pattern_observed else 'NO'}",
                f"probe_pattern_note={row.probe_pattern_note}",
                f"detected_events={row.event_count}",
                f"selected_reference_expected_events={reference_pattern.expected_event_count}",
                f"aligned_vs_selected_reference={ref_aligned}",
                f"missing_vs_selected_reference={ref_missing}",
                f"extra_vs_selected_reference={row.reference_summary.unmatched_other}",
                f"mean_lag_vs_selected_reference_ms={_format_float(row.reference_summary.mean_lag_ms, 3)}",
                f"optimal_probe_shift_ms={_format_float((row.optimal_lag_sec or 0.0) * 1000.0, 3)}",
                f"selected_reference_95ci_ms={_format_ci(row.reference_summary, 3)}",
                f"control_reference_events_expected={control_expected_event_count}",
                f"aligned_vs_control={control_aligned}",
                f"missing_vs_control={control_missing}",
                f"extra_vs_control={row.control_summary.unmatched_other}",
                f"mean_lag_vs_control_ms={_format_float(row.control_summary.mean_lag_ms, 3)}",
                f"lag_vs_control_95ci_ms={_format_ci(row.control_summary, 3)}",
                f"mean_abs_residual_vs_control_ms={_format_float(row.control_summary.mean_abs_residual_ms, 3)}",
                f"aux2_reference_events_expected={aux2_expected_event_count}",
                f"aligned_vs_aux2={aux2_aligned}",
                f"ramp_zone_count={row.ramp_zone_count}",
                f"ramp_samples_added={row.ramp_samples_added}",
            ]
        )
        if row.aux2_summary is not None:
            lines.extend(
                [
                    f"missing_vs_aux2={aux2_missing}",
                    f"extra_vs_aux2={row.aux2_summary.unmatched_other}",
                    f"mean_lag_vs_aux2_ms={_format_float(row.aux2_summary.mean_lag_ms, 3)}",
                    f"lag_vs_aux2_95ci_ms={_format_ci(row.aux2_summary, 3)}",
                    f"mean_abs_residual_vs_aux2_ms={_format_float(row.aux2_summary.mean_abs_residual_ms, 3)}",
                ]
            )
        lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _plot_alignment_figure(
    out_pdf_path: Path,
    out_png_path: Path,
    *,
    file_path: Path,
    syncstation_firmware_version: Optional[str],
    generated_at: Optional[datetime],
    recorded_at: Optional[datetime],
    device_bundles: Sequence[DeviceBundle],
    device_rows: Sequence[DeviceRow],
    control_buffer: Optional[BufferTrace],
    aux2_trace: Optional[BufferTrace],
    reference_expected_times_sec: Sequence[float],
    reference_source: str,
    info: Dict[str, Any],
) -> None:
    if not device_bundles:
        raise ValueError(f"No device-specific buffer channels found in {file_path.name}")
    row_lookup = {row.device_label: row for row in device_rows}
    non_sync_bundles = [bundle for bundle in device_bundles if row_lookup.get(bundle.buffer_trace.label) and row_lookup[bundle.buffer_trace.label].non_sync]
    top_count = len(device_bundles)
    detail_count = len(non_sync_bundles)
    count = top_count + detail_count
    fig, axes = plt.subplots(count, 1, sharex=True, figsize=(18, max(4.5, 3.3 * top_count + 3.0 * detail_count)))
    if count == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.07, right=0.78, top=0.93, bottom=0.06, hspace=0.62)
    use_time = bool(info.get("has_saved_time"))
    reference_expected = np.asarray(reference_expected_times_sec, dtype=float).reshape(-1)
    reference_x = reference_expected if use_time else reference_expected * float(info.get("sample_rate") or 1.0)

    for ax, bundle in zip(axes[:top_count], device_bundles):
        trace = bundle.buffer_trace
        row = row_lookup.get(trace.label)
        shifted_time = _shifted_trace_time(trace, row.optimal_lag_sec if row is not None else None)
        x = shifted_time if use_time else shifted_time * float(info.get("sample_rate") or 1.0)
        ax.plot(x, _normalize_for_plot(trace.values), color="#005F73", linewidth=1.3, label=trace.label, zorder=3)
        if control_buffer is not None:
            cx = control_buffer.time_sec if use_time else np.arange(control_buffer.values.size, dtype=float)
            ax.plot(cx, _normalize_for_plot(control_buffer.values), color="#111111", linewidth=1.0, alpha=0.5, label=control_buffer.label, zorder=1)
        if aux2_trace is not None:
            axx = aux2_trace.time_sec if use_time else np.arange(aux2_trace.values.size, dtype=float)
            ax.plot(axx, _normalize_for_plot(aux2_trace.values), color="#C16622", linewidth=1.0, alpha=0.5, label=aux2_trace.label, zorder=2)
        for ref_x in reference_x:
            ax.axvline(float(ref_x), color="#8D99AE", linewidth=0.8, linestyle="--", alpha=0.35, zorder=0)

        if trace.event_indices.size:
            event_x = shifted_time[trace.event_indices] if use_time else shifted_time[trace.event_indices] * float(info.get("sample_rate") or 1.0)
            ax.scatter(
                event_x,
                _normalize_for_plot(trace.values)[trace.event_indices],
                s=90,
                color="#0A9396",
                edgecolors="#001219",
                linewidths=0.7,
                label=f"{trace.label} events",
                zorder=5,
            )

        if row is not None:
            aligned_pct = _pct(int(row.reference_summary.matched_count), int(row.reference_summary.matched_count + row.reference_summary.unmatched_ref))
        else:
            aligned_pct = 0.0
        if row is not None and row.reference_summary.mean_lag_ms is not None:
            title = (
                f"{trace.label} | {row.sync_status} | aligned={row.reference_summary.matched_count}/{row.reference_summary.matched_count + row.reference_summary.unmatched_ref} ({aligned_pct:.1f}%) | "
                f"extra={row.reference_summary.unmatched_other} | lag={_format_float(row.reference_summary.mean_lag_ms, 2)} ms | shift={_format_float((row.optimal_lag_sec or 0.0) * 1000.0, 2)} ms | "
                f"sample={row.sample_status}"
            )
        else:
            title = f"{trace.label} | events={int(trace.event_times_sec.size)} | sample={row.sample_status if row is not None else 'NA'}"
        ax.set_title(title)
        ax.set_ylabel("Norm. abs")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

    for ax, bundle in zip(axes[top_count:], non_sync_bundles):
        trace = bundle.buffer_trace
        row = row_lookup.get(trace.label)
        shifted_time = _shifted_trace_time(trace, row.optimal_lag_sec if row is not None else None)
        x = shifted_time if use_time else shifted_time * float(info.get("sample_rate") or 1.0)
        ax.plot(x, _normalize_for_plot(trace.values), color="#005F73", linewidth=1.2, label=f"{trace.device} sync buffer", zorder=3)
        for ref_x in reference_x:
            ax.axvline(float(ref_x), color="#6C757D", linewidth=0.9, linestyle="--", alpha=0.55, zorder=1)
        if trace.event_indices.size:
            event_x = shifted_time[trace.event_indices] if use_time else shifted_time[trace.event_indices] * float(info.get("sample_rate") or 1.0)
            ax.scatter(
                event_x,
                _normalize_for_plot(trace.values)[trace.event_indices],
                s=95,
                color="#0A9396",
                edgecolors="#001219",
                linewidths=0.7,
                label="Observed probe spikes",
                zorder=5,
            )
        ax.set_ylabel("Sync")
        ax.grid(True, alpha=0.25)
        ramp_ax = ax.twinx()
        if bundle.ramp_trace is not None:
            ramp_trace = bundle.ramp_trace
            ramp_shifted_time = _shifted_trace_time(ramp_trace, row.optimal_lag_sec if row is not None else None)
            ramp_x = ramp_shifted_time if use_time else ramp_shifted_time * float(info.get("sample_rate") or 1.0)
            ramp_y = _normalize_unit_interval(_to_ascending_ramp(ramp_trace.raw_values))
            ramp_ax.plot(ramp_x, ramp_y, color="#AE2012", linewidth=1.0, alpha=0.9, label=f"{trace.device} ramp", zorder=2)
            for start_idx, length in ramp_trace.ramp_zones:
                start_x = float(ramp_x[min(max(int(start_idx), 0), ramp_x.size - 1)])
                end_x = float(ramp_x[min(max(int(start_idx + length), 0), ramp_x.size - 1)])
                ax.axvspan(start_x, end_x, color="#F4A261", alpha=0.18, zorder=0)
        ramp_ax.set_ylabel("Ramp")
        title = (
            f"{trace.device} non-sync detail | pattern-in-spikes={'YES' if row and row.probe_pattern_observed else 'NO'} | "
            f"expected={reference_expected.size} | observed={trace.event_indices.size} | shift={_format_float(((row.optimal_lag_sec or 0.0) * 1000.0) if row is not None else None, 2)} ms | {row.sample_status if row is not None else 'NA'}"
        )
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        ramp_handles, ramp_labels = ramp_ax.get_legend_handles_labels()
        ax.legend(handles + ramp_handles, labels + ramp_labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

    axes[-1].set_xlabel("Time [s]" if use_time else "Samples")
    fig.suptitle(
        f"{file_path.name} | probes={len(device_bundles)} | reference={reference_source} | expected events={reference_expected.size} | "
        f"control observed={control_buffer.event_times_sec.size if control_buffer is not None else 0} | aux2 observed={aux2_trace.event_times_sec.size if aux2_trace is not None else 0}",
        fontsize=13,
    )
    fig.text(
        0.07,
        0.965,
        f"Firmware {syncstation_firmware_version or 'unknown'} | Recorded { _format_datetime(recorded_at) } | Generated { _format_datetime(generated_at) }",
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.07,
        0.948,
        f"Contact {_report_contact_line()} | {REPORT_ACKNOWLEDGEMENT} | {REPORT_LOCATION}",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
    )
    fig.savefig(out_pdf_path, format="pdf")
    fig.savefig(out_png_path, dpi=170)
    plt.close(fig)


def analyze_file(path: Path, *, syncstation_firmware_version: Optional[str] = None) -> FileAnalysisResult:
    generated_at = datetime.now()
    recorded_at = _file_recorded_datetime(path)
    traces, info = _load_buffer_traces(path)
    device_bundles = _build_device_bundles(traces)
    device_buffers = [bundle.buffer_trace for bundle in device_bundles]
    control_buffer = _first_or_none(traces, lambda trace: trace.is_control_buffer)
    aux2_trace = _first_or_none(traces, lambda trace: trace.is_aux2)
    if not device_buffers:
        raise ValueError(f"No device-specific buffer channels found in {path}")

    reference_source, reference_pattern, control_pattern, aux2_pattern = _select_reference_pattern(control_buffer, aux2_trace)
    observed_sets = []
    for trace in (control_buffer, aux2_trace):
        arr = np.asarray(trace.event_times_sec if trace is not None else [], dtype=float)
        if arr.size:
            observed_sets.append(arr)
    window_start = float(min(arr[0] for arr in observed_sets)) if observed_sets else None
    window_end = float(max(arr[-1] for arr in observed_sets)) if observed_sets else None
    control_expected_times = _expected_times_from_pattern_window(control_pattern, control_buffer.event_times_sec if control_buffer is not None else [], target_start_sec=window_start, target_end_sec=window_end)
    aux2_expected_times = _expected_times_from_pattern_window(aux2_pattern, aux2_trace.event_times_sec if aux2_trace is not None else [], target_start_sec=window_start, target_end_sec=window_end)
    reference_expected_times = _expected_times_from_pattern_window(
        reference_pattern,
        control_expected_times if reference_source == "control" else aux2_expected_times,
        target_start_sec=window_start,
        target_end_sec=window_end,
    )
    device_rows = _make_device_rows(
        device_bundles,
        control_expected_times_sec=control_expected_times,
        aux2_expected_times_sec=aux2_expected_times,
        reference_expected_times_sec=reference_expected_times,
    )
    file_summary = _pool_alignment_summaries([row.control_summary for row in device_rows])
    file_aux2_summary = _pool_alignment_summaries([row.aux2_summary for row in device_rows if row.aux2_summary is not None]) if aux2_trace is not None else None
    reference_file_summary = _pool_alignment_summaries([row.reference_summary for row in device_rows])
    control_aux_summary = None
    if control_buffer is not None and aux2_trace is not None:
        control_aux_summary = _best_event_alignment(aux2_trace.event_times_sec, control_buffer.event_times_sec)

    out_dir = path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_pdf_path = out_dir / f"{path.stem}_buffer_alignment.pdf"
    plot_png_path = out_dir / f"{path.stem}_buffer_alignment.png"
    report_path = out_dir / f"{path.stem}_buffer_alignment.txt"
    _plot_alignment_figure(
        plot_pdf_path,
        plot_png_path,
        file_path=path,
        syncstation_firmware_version=syncstation_firmware_version,
        generated_at=generated_at,
        recorded_at=recorded_at,
        device_bundles=device_bundles,
        device_rows=device_rows,
        control_buffer=control_buffer,
        aux2_trace=aux2_trace,
        reference_expected_times_sec=reference_expected_times,
        reference_source=reference_source,
        info=info,
    )
    _save_report(
        report_path,
        file_path=path,
        syncstation_firmware_version=syncstation_firmware_version,
        generated_at=generated_at,
        recorded_at=recorded_at,
        device_rows=device_rows,
        file_summary=file_summary,
        file_aux2_summary=file_aux2_summary,
        reference_file_summary=reference_file_summary,
        control_aux_summary=control_aux_summary,
        control_event_count=int(control_buffer.event_times_sec.size) if control_buffer is not None else 0,
        aux2_event_count=int(aux2_trace.event_times_sec.size) if aux2_trace is not None else 0,
        control_expected_event_count=int(control_expected_times.size),
        aux2_expected_event_count=int(aux2_expected_times.size),
        reference_source=reference_source,
        reference_pattern=reference_pattern,
        control_pattern=control_pattern,
        aux2_pattern=aux2_pattern,
        info=info,
    )
    control_aux_misaligned = _control_aux_misaligned(
        control_aux_summary,
        int(control_buffer.event_times_sec.size) if control_buffer is not None else 0,
        int(aux2_trace.event_times_sec.size) if aux2_trace is not None else 0,
    )
    return FileAnalysisResult(
        file_path=path,
        out_dir=out_dir,
        plot_pdf_path=plot_pdf_path,
        plot_png_path=plot_png_path,
        report_path=report_path,
        device_rows=device_rows,
        file_summary=file_summary,
        file_aux2_summary=file_aux2_summary,
        reference_file_summary=reference_file_summary,
        control_aux_summary=control_aux_summary,
        control_event_count=int(control_buffer.event_times_sec.size) if control_buffer is not None else 0,
        aux2_event_count=int(aux2_trace.event_times_sec.size) if aux2_trace is not None else 0,
        control_expected_event_count=int(control_expected_times.size),
        aux2_expected_event_count=int(aux2_expected_times.size),
        reference_source=reference_source,
        reference_pattern=reference_pattern,
        control_pattern=control_pattern,
        aux2_pattern=aux2_pattern,
        control_aux_misaligned=control_aux_misaligned,
        syncstation_firmware_version=syncstation_firmware_version,
        generated_at=generated_at,
        recorded_at=recorded_at,
    )


def _dataset_probe_rows(results: Sequence[FileAnalysisResult], reference: str = "control") -> List[Tuple[str, AlignmentSummary, int]]:
    grouped: Dict[str, List[AlignmentSummary]] = {}
    files_with_events: Dict[str, int] = {}
    for result in results:
        for row in result.device_rows:
            summary = row.control_summary if reference == "control" else row.aux2_summary
            if summary is None:
                continue
            grouped.setdefault(row.device_label, []).append(summary)
            if row.event_count > 0:
                files_with_events[row.device_label] = files_with_events.get(row.device_label, 0) + 1
    out: List[Tuple[str, AlignmentSummary, int]] = []
    for device_label in sorted(grouped):
        out.append((device_label, _pool_alignment_summaries(grouped[device_label]), files_with_events.get(device_label, 0)))
    return out


def _summary_plot_counts_and_lags(
    labels: Sequence[str],
    summaries: Sequence[AlignmentSummary],
    *,
    title_prefix: str,
    metadata_text: Optional[str],
    out_png_path: Path,
    out_pdf_path: Optional[Path] = None,
) -> None:
    aligned = np.asarray([summary.matched_count for summary in summaries], dtype=float)
    missing = np.asarray([summary.unmatched_ref for summary in summaries], dtype=float)
    extra = np.asarray([summary.unmatched_other for summary in summaries], dtype=float)
    means = np.asarray([summary.mean_lag_ms if summary.mean_lag_ms is not None else np.nan for summary in summaries], dtype=float)
    ci_low = np.asarray([summary.ci95_low_ms if summary.ci95_low_ms is not None else np.nan for summary in summaries], dtype=float)
    ci_high = np.asarray([summary.ci95_high_ms if summary.ci95_high_ms is not None else np.nan for summary in summaries], dtype=float)

    y = np.arange(len(labels), dtype=float)
    fig_height = max(8, 0.45 * len(labels) + 4)
    fig, axes = plt.subplots(2, 1, figsize=(14, fig_height))
    fig.subplots_adjust(left=0.27, right=0.98, top=0.90, bottom=0.06, hspace=0.35)

    axes[0].barh(y, aligned, color="#0A9396", label="Aligned")
    axes[0].barh(y, missing, left=aligned, color="#AE2012", label="Missing vs reference")
    axes[0].barh(y, extra, left=aligned + missing, color="#CA6702", label="Extra probe events")
    axes[0].set_ylabel("")
    axes[0].set_title(f"{title_prefix} Ideal vs Deviation Counts")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()

    valid = np.isfinite(means)
    if np.any(valid):
        xerr = np.vstack([means[valid] - ci_low[valid], ci_high[valid] - means[valid]])
        axes[1].errorbar(means[valid], y[valid], xerr=xerr, fmt="o", color="#005F73", ecolor="#005F73", capsize=4)
    axes[1].axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("")
    axes[1].set_title(f"{title_prefix} Mean Lag and 95% CI")
    axes[1].grid(True, axis="x", alpha=0.25)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean lag [ms]")

    if metadata_text:
        fig.text(0.01, 0.98, metadata_text, ha="left", va="top", fontsize=9, color="#444444")

    fig.savefig(out_png_path, dpi=170)
    if out_pdf_path is not None:
        fig.savefig(out_pdf_path, format="pdf")
    plt.close(fig)


def _relpath(target: Path, start: Path) -> str:
    return os.path.relpath(str(target), str(start))


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _latex_escape(text: Any) -> str:
    value = "" if text is None else str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = value
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _latex_table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, landscape: bool = True, font_size: str = r"\scriptsize") -> str:
    colspec = "l" + "c" * max(0, len(headers) - 1)
    body_lines = [
        r"\begin{center}",
        font_size,
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        body_lines.append(" & ".join(_latex_escape(cell) for cell in row) + r" \\")
    body_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{center}",
        ]
    )
    table_block = "\n".join(body_lines)
    if not landscape:
        return table_block
    return "\n".join([r"\begin{landscape}", table_block, r"\end{landscape}"])


def _copy_into_dir(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _write_latex_summary(
    *,
    report_dir: Path,
    sorted_results: Sequence[FileAnalysisResult],
    file_rollups_control: Sequence[Dict[str, Any]],
    probe_rollups_control: Sequence[Dict[str, Any]],
    probe_rollups_aux2: Sequence[Dict[str, Any]],
    per_file_table_rows: Sequence[Sequence[str]],
    per_file_aux2_table_rows: Sequence[Sequence[str]],
    per_probe_table_rows: Sequence[Sequence[str]],
    per_probe_aux2_table_rows: Sequence[Sequence[str]],
    summary_plot_pdfs: Sequence[Path],
) -> Path:
    latex_dir = report_dir / "latex_report"
    figures_dir = latex_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    copied_summary_plots = [_copy_into_dir(path, figures_dir) for path in summary_plot_pdfs]
    copied_result_plots = {result.file_path.name: _copy_into_dir(result.plot_pdf_path, figures_dir) for result in sorted_results}

    tex_path = latex_dir / "otb4_buffer_alignment_summary.tex"
    pdf_path = latex_dir / "otb4_buffer_alignment_summary.pdf"

    firmware_values = sorted({str(result.syncstation_firmware_version).strip() for result in sorted_results if str(result.syncstation_firmware_version or "").strip()})
    firmware_text = ", ".join(firmware_values) if firmware_values else "unknown"
    generated_at = datetime.now()
    recorded_range = _recorded_range_text(sorted_results)
    tex_lines: List[str] = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=18mm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{pdflscape}",
        r"\usepackage{adjustbox}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\captionsetup{font=small}",
        r"\begin{document}",
        r"\title{OTB4 Buffer Alignment Summary}",
        rf"\date{{Generated { _latex_escape(_format_datetime(generated_at)) }}}",
        r"\maketitle",
        rf"Selected files: {_latex_escape(len(sorted_results))}\\",
        rf"Syncstation firmware: {_latex_escape(firmware_text)}\\",
        rf"Data recorded range: {_latex_escape(recorded_range)}\\",
        rf"Contact: {_latex_escape(_report_contact_line())}\\",
        rf"{_latex_escape(REPORT_ACKNOWLEDGEMENT)}\\",
        rf"Location: {_latex_escape(REPORT_LOCATION)}\\",
        r"\section*{Device-Pooled Per File}",
        r"\subsection*{Versus Control}",
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width=\textwidth]{{figures/{_latex_escape(copied_summary_plots[0].name)}}}",
        r"\end{figure}",
        _latex_table(
            [
                "File",
                "AUX2!=CTRL",
                "Probes",
                "Control expected events",
                "Ideal total",
                "Aligned",
                "Missing",
                "Extra",
                "Mean lag [ms]",
                "95% CI [ms]",
                "Mean abs residual [ms]",
            ],
            per_file_table_rows,
            landscape=True,
        ),
        r"\subsection*{Versus AUX 2}",
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width=\textwidth]{{figures/{_latex_escape(copied_summary_plots[1].name)}}}",
        r"\end{figure}",
        _latex_table(
            [
                "File",
                "AUX2!=CTRL",
                "Probes",
                "AUX 2 expected events",
                "Ideal total",
                "Aligned",
                "Missing",
                "Extra",
                "Mean lag [ms]",
                "95% CI [ms]",
                "Mean abs residual [ms]",
            ],
            per_file_aux2_table_rows,
            landscape=True,
        ),
        r"\section*{Probe-Pooled Per Dataset}",
        r"\subsection*{Versus Control}",
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width=\textwidth]{{figures/{_latex_escape(copied_summary_plots[2].name)}}}",
        r"\end{figure}",
        _latex_table(
            ["Probe", "AUX2!=CTRL files", "Files with events", "Ideal total", "Detected total", "Aligned", "Missing", "Extra", "Mean lag [ms]", "95% CI [ms]", "Mean abs residual [ms]"],
            per_probe_table_rows,
            landscape=True,
        ),
        r"\subsection*{Versus AUX 2}",
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width=\textwidth]{{figures/{_latex_escape(copied_summary_plots[3].name)}}}",
        r"\end{figure}",
        _latex_table(
            ["Probe", "AUX2!=CTRL files", "Files with events", "Ideal total", "Detected total", "Aligned", "Missing", "Extra", "Mean lag [ms]", "95% CI [ms]", "Mean abs residual [ms]"],
            per_probe_aux2_table_rows,
            landscape=True,
        ),
        r"\section*{Detailed Per-File Figures and Tables}",
    ]

    ordered_results = [item["result"] for item in file_rollups_control]
    for result in ordered_results:
        tex_lines.extend(
            [
                rf"\subsection*{{{_latex_escape(result.file_path.name)}}}",
                rf"Syncstation firmware: {_latex_escape(result.syncstation_firmware_version or 'unknown')}\\",
                rf"Data recorded: {_latex_escape(_format_datetime(result.recorded_at))}\\",
                rf"Report generated: {_latex_escape(_format_datetime(result.generated_at))}\\",
                rf"Contact: {_latex_escape(_report_contact_line())}\\",
                rf"{_latex_escape(REPORT_ACKNOWLEDGEMENT)}\\",
                rf"Location: {_latex_escape(REPORT_LOCATION)}\\",
                rf"Reference agreement ({_latex_escape('MISALIGNED' if result.control_aux_misaligned else 'OK')}): "
                rf"control observed/expected={_latex_escape(result.control_event_count)}/{_latex_escape(result.control_expected_event_count)}, "
                rf"aux2 observed/expected={_latex_escape(result.aux2_event_count)}/{_latex_escape(result.aux2_expected_event_count)}, "
                rf"selected reference={_latex_escape(result.reference_source)}, "
                rf"aligned={_latex_escape(result.control_aux_summary.matched_count if result.control_aux_summary is not None else 'NA')}, "
                rf"missing in aux2={_latex_escape(result.control_aux_summary.unmatched_ref if result.control_aux_summary is not None else 'NA')}, "
                rf"extra in aux2={_latex_escape(result.control_aux_summary.unmatched_other if result.control_aux_summary is not None else 'NA')}.",
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=\textwidth]{{figures/{_latex_escape(copied_result_plots[result.file_path.name].name)}}}",
                r"\end{figure}",
            ]
        )
        detail_control_rows: List[List[str]] = []
        detail_aux2_rows: List[List[str]] = []
        sorted_device_rows = sorted(
            result.device_rows,
            key=lambda row: (-_pct(int(row.reference_summary.matched_count), int(result.reference_pattern.expected_event_count)), row.device_label.lower()),
        )
        for row in sorted_device_rows:
            detail_control_rows.append(
                [
                    row.device_label,
                    row.sync_status,
                    row.sample_status,
                    str(row.event_count),
                    str(result.control_expected_event_count),
                    _count_pct_text(int(row.control_summary.matched_count), int(result.control_expected_event_count)),
                    _count_pct_text(int(row.control_summary.unmatched_ref), int(result.control_expected_event_count)),
                    str(row.control_summary.unmatched_other),
                    _format_float(row.control_summary.mean_lag_ms, 2),
                    _format_ci(row.control_summary, 2),
                    _format_float(row.control_summary.mean_abs_residual_ms, 2),
                ]
            )
            detail_aux2_rows.append(
                [
                    row.device_label,
                    row.sync_status,
                    row.sample_status,
                    str(row.event_count),
                    str(result.aux2_expected_event_count),
                    _count_pct_text(int(row.aux2_summary.matched_count), int(result.aux2_expected_event_count)) if row.aux2_summary is not None else "NA",
                    _count_pct_text(int(row.aux2_summary.unmatched_ref), int(result.aux2_expected_event_count)) if row.aux2_summary is not None else "NA",
                    str(row.aux2_summary.unmatched_other) if row.aux2_summary is not None else "NA",
                    _format_float(row.aux2_summary.mean_lag_ms, 2) if row.aux2_summary is not None else "NA",
                    _format_ci(row.aux2_summary, 2) if row.aux2_summary is not None else "NA",
                    _format_float(row.aux2_summary.mean_abs_residual_ms, 2) if row.aux2_summary is not None else "NA",
                ]
            )
        tex_lines.append(
            _latex_table(
                [
                    "Probe",
                    "Sync status",
                    "Sample status",
                    "Detected probe events",
                    "Control expected events",
                    "Aligned vs control",
                    "Missing vs control",
                    "Extra vs control",
                    "Mean lag vs control [ms]",
                    "95% CI vs control [ms]",
                    "Mean abs residual vs control [ms]",
                ],
                detail_control_rows,
                landscape=True,
            )
        )
        tex_lines.append(
            _latex_table(
                [
                    "Probe",
                    "Sync status",
                    "Sample status",
                    "Detected probe events",
                    "AUX 2 expected events",
                    "Aligned vs AUX 2",
                    "Missing vs AUX 2",
                    "Extra vs AUX 2",
                    "Mean lag vs AUX 2 [ms]",
                    "95% CI vs AUX 2 [ms]",
                    "Mean abs residual vs AUX 2 [ms]",
                ],
                detail_aux2_rows,
                landscape=True,
            )
        )

    tex_lines.append(r"\end{document}")
    tex_path.write_text("\n".join(tex_lines).rstrip() + "\n", encoding="utf-8")

    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(latex_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"LaTeX build failed for {tex_path}: {proc.stdout[-4000:]}")
    if not pdf_path.exists():
        raise RuntimeError(f"LaTeX build did not produce {pdf_path}")
    return pdf_path


def _write_overall_summary(results: Sequence[FileAnalysisResult]) -> Optional[Path]:
    if len(results) < 2:
        return None
    common_parent = Path(os.path.commonpath([str(result.file_path.parent) for result in results]))
    report_path = common_parent / "otb4_buffer_alignment_summary.md"
    per_file_control_plot_path = common_parent / "otb4_buffer_alignment_summary_per_file_control.png"
    per_file_control_plot_pdf_path = common_parent / "otb4_buffer_alignment_summary_per_file_control.pdf"
    per_file_aux2_plot_path = common_parent / "otb4_buffer_alignment_summary_per_file_aux2.png"
    per_file_aux2_plot_pdf_path = common_parent / "otb4_buffer_alignment_summary_per_file_aux2.pdf"
    per_probe_control_plot_path = common_parent / "otb4_buffer_alignment_summary_per_probe_control.png"
    per_probe_control_plot_pdf_path = common_parent / "otb4_buffer_alignment_summary_per_probe_control.pdf"
    per_probe_aux2_plot_path = common_parent / "otb4_buffer_alignment_summary_per_probe_aux2.png"
    per_probe_aux2_plot_pdf_path = common_parent / "otb4_buffer_alignment_summary_per_probe_aux2.pdf"

    sorted_results = sorted(results, key=lambda result: result.file_path.name.lower())
    generated_at = datetime.now()
    recorded_range = _recorded_range_text(sorted_results)
    file_rollups_control = sorted(
        [_reference_rollup(result, "control") | {"result": result} for result in sorted_results],
        key=lambda item: (-float(item["aligned_pct"]), item["result"].file_path.name.lower()),
    )
    control_labels = [
        f"{item['result'].file_path.stem}{' !' if item['misaligned_aux2_control'] else ''} ({item['aligned_pct']:.1f}%)"
        for item in file_rollups_control
    ]
    _summary_plot_counts_and_lags(
        control_labels,
        [item["result"].file_summary for item in file_rollups_control],
        title_prefix="Device-Pooled Per File vs Control (! = AUX2/control mismatch)",
        metadata_text=f"Generated {_format_datetime(generated_at)} | Data recorded {recorded_range} | Contact {_report_contact_line()} | {REPORT_ACKNOWLEDGEMENT} | {REPORT_LOCATION}",
        out_png_path=per_file_control_plot_path,
        out_pdf_path=per_file_control_plot_pdf_path,
    )

    file_rollups_aux2 = sorted(
        [_reference_rollup(result, "aux2") | {"result": result} for result in sorted_results],
        key=lambda item: (-float(item["aligned_pct"]), item["result"].file_path.name.lower()),
    )
    aux2_labels = [
        f"{item['result'].file_path.stem}{' !' if item['misaligned_aux2_control'] else ''} ({item['aligned_pct']:.1f}%)"
        for item in file_rollups_aux2
    ]
    _summary_plot_counts_and_lags(
        aux2_labels,
        [item["result"].file_aux2_summary if item["result"].file_aux2_summary is not None else AlignmentSummary(0, 0, 0, None, None, None, None, None, None, []) for item in file_rollups_aux2],
        title_prefix="Device-Pooled Per File vs AUX 2 (! = AUX2/control mismatch)",
        metadata_text=f"Generated {_format_datetime(generated_at)} | Data recorded {recorded_range} | Contact {_report_contact_line()} | {REPORT_ACKNOWLEDGEMENT} | {REPORT_LOCATION}",
        out_png_path=per_file_aux2_plot_path,
        out_pdf_path=per_file_aux2_plot_pdf_path,
    )

    probe_rows = _dataset_probe_rows(sorted_results, reference="control")
    probe_rollups_control = sorted(
        [_probe_rollup(sorted_results, device_label, "control") for device_label, _summary, _count in probe_rows],
        key=lambda item: (-float(item["aligned_pct"]), str(item["device_label"]).lower()),
    )
    _summary_plot_counts_and_lags(
        [
            f"{item['device_label']}{' !' if item['misaligned_file_count'] else ''} ({item['aligned_pct']:.1f}%)"
            for item in probe_rollups_control
        ],
        [item["summary"] for item in probe_rollups_control],
        title_prefix="Probe-Pooled Per Dataset vs Control (! = AUX2/control mismatch present)",
        metadata_text=f"Generated {_format_datetime(generated_at)} | Data recorded {recorded_range} | Contact {_report_contact_line()} | {REPORT_ACKNOWLEDGEMENT} | {REPORT_LOCATION}",
        out_png_path=per_probe_control_plot_path,
        out_pdf_path=per_probe_control_plot_pdf_path,
    )
    probe_rows_aux2 = _dataset_probe_rows(sorted_results, reference="aux2")
    probe_rollups_aux2 = sorted(
        [_probe_rollup(sorted_results, device_label, "aux2") for device_label, _summary, _count in probe_rows_aux2],
        key=lambda item: (-float(item["aligned_pct"]), str(item["device_label"]).lower()),
    )
    _summary_plot_counts_and_lags(
        [
            f"{item['device_label']}{' !' if item['misaligned_file_count'] else ''} ({item['aligned_pct']:.1f}%)"
            for item in probe_rollups_aux2
        ],
        [item["summary"] for item in probe_rollups_aux2],
        title_prefix="Probe-Pooled Per Dataset vs AUX 2 (! = AUX2/control mismatch present)",
        metadata_text=f"Generated {_format_datetime(generated_at)} | Data recorded {recorded_range} | Contact {_report_contact_line()} | {REPORT_ACKNOWLEDGEMENT} | {REPORT_LOCATION}",
        out_png_path=per_probe_aux2_plot_path,
        out_pdf_path=per_probe_aux2_plot_pdf_path,
    )

    per_file_control_table_rows: List[List[str]] = []
    per_file_aux2_table_rows: List[List[str]] = []
    for item in file_rollups_control:
        result = item["result"]
        ctrl = item
        aux = _reference_rollup(result, "aux2")
        per_file_control_table_rows.append(
            [
                result.file_path.name,
                "YES" if result.control_aux_misaligned else "NO",
                str(ctrl["probe_count"]),
                str(ctrl["ref_count"]),
                str(ctrl["ideal_total"]),
                _count_pct_text(int(ctrl["aligned_total"]), int(ctrl["ideal_total"])),
                _count_pct_text(int(ctrl["missing_total"]), int(ctrl["ideal_total"])),
                str(ctrl["extra_total"]),
                _format_float(ctrl["mean_lag_ms"], 2),
                ctrl["ci95"],
                _format_float(ctrl["mean_abs_residual_ms"], 2),
            ]
        )
        per_file_aux2_table_rows.append(
            [
                result.file_path.name,
                "YES" if result.control_aux_misaligned else "NO",
                str(aux["probe_count"]),
                str(aux["ref_count"]),
                str(aux["ideal_total"]),
                _count_pct_text(int(aux["aligned_total"]), int(aux["ideal_total"])),
                _count_pct_text(int(aux["missing_total"]), int(aux["ideal_total"])),
                str(aux["extra_total"]),
                _format_float(aux["mean_lag_ms"], 2),
                aux["ci95"],
                _format_float(aux["mean_abs_residual_ms"], 2),
            ]
        )

    per_probe_table_rows: List[List[str]] = []
    for roll in probe_rollups_control:
        per_probe_table_rows.append(
            [
                roll["device_label"],
                str(roll["misaligned_file_count"]),
                str(roll["files_with_events"]),
                str(roll["ideal_total"]),
                str(roll["detected_total"]),
                _count_pct_text(int(roll["aligned_total"]), int(roll["ideal_total"])),
                _count_pct_text(int(roll["missing_total"]), int(roll["ideal_total"])),
                str(roll["extra_total"]),
                _format_float(roll["mean_lag_ms"], 2),
                roll["ci95"],
                _format_float(roll["mean_abs_residual_ms"], 2),
            ]
        )

    per_probe_aux2_table_rows: List[List[str]] = []
    for roll in probe_rollups_aux2:
        per_probe_aux2_table_rows.append(
            [
                roll["device_label"],
                str(roll["misaligned_file_count"]),
                str(roll["files_with_events"]),
                str(roll["ideal_total"]),
                str(roll["detected_total"]),
                _count_pct_text(int(roll["aligned_total"]), int(roll["ideal_total"])),
                _count_pct_text(int(roll["missing_total"]), int(roll["ideal_total"])),
                str(roll["extra_total"]),
                _format_float(roll["mean_lag_ms"], 2),
                roll["ci95"],
                _format_float(roll["mean_abs_residual_ms"], 2),
            ]
        )

    firmware_values = sorted({str(result.syncstation_firmware_version).strip() for result in results if str(result.syncstation_firmware_version or "").strip()})
    firmware_text = ", ".join(firmware_values) if firmware_values else "unknown"
    md_lines: List[str] = [
        "# OTB4 Buffer Alignment Summary",
        "",
        f"Generated: {_format_datetime(generated_at)}",
        "",
        f"Selected files: {len(sorted_results)}",
        f"Syncstation firmware: {firmware_text}",
        f"Data recorded range: {recorded_range}",
        f"Contact: {_report_contact_line()}",
        REPORT_ACKNOWLEDGEMENT,
        f"Location: {REPORT_LOCATION}",
        "",
        "## Device-Pooled Per File",
        "",
        "### Versus Control",
        "",
        f"![Device-pooled per file vs control]({_relpath(per_file_control_plot_path, report_path.parent)})",
        "",
        _markdown_table(
            [
                "File",
                "AUX2!=CTRL",
                "Probes",
                "Control expected events",
                "Ideal total",
                "Aligned",
                "Missing",
                "Extra",
                "Mean lag [ms]",
                "95% CI [ms]",
                "Mean abs residual [ms]",
            ],
            per_file_control_table_rows,
        ),
        "",
        "### Versus AUX 2",
        "",
        f"![Device-pooled per file vs AUX 2]({_relpath(per_file_aux2_plot_path, report_path.parent)})",
        "",
        _markdown_table(
            [
                "File",
                "AUX2!=CTRL",
                "Probes",
                "AUX 2 expected events",
                "Ideal total",
                "Aligned",
                "Missing",
                "Extra",
                "Mean lag [ms]",
                "95% CI [ms]",
                "Mean abs residual [ms]",
            ],
            per_file_aux2_table_rows,
        ),
        "",
        "## Probe-Pooled Per Dataset",
        "",
        "### Versus Control",
        "",
        f"![Probe-pooled per dataset vs control]({_relpath(per_probe_control_plot_path, report_path.parent)})",
        "",
        _markdown_table(
            ["Probe", "AUX2!=CTRL files", "Files with events", "Ideal total", "Detected total", "Aligned", "Missing", "Extra", "Mean lag [ms]", "95% CI [ms]", "Mean abs residual [ms]"],
            per_probe_table_rows,
        ),
        "",
        "### Versus AUX 2",
        "",
        f"![Probe-pooled per dataset vs AUX 2]({_relpath(per_probe_aux2_plot_path, report_path.parent)})",
        "",
        _markdown_table(
            ["Probe", "AUX2!=CTRL files", "Files with events", "Ideal total", "Detected total", "Aligned", "Missing", "Extra", "Mean lag [ms]", "95% CI [ms]", "Mean abs residual [ms]"],
            per_probe_aux2_table_rows,
        ),
        "",
        "## Detailed Per-File Tables and Figures",
        "",
    ]

    for result in sorted_results:
        md_lines.extend(
            [
                f"### {result.file_path.name}",
                "",
                f"Syncstation firmware: {result.syncstation_firmware_version or 'unknown'}",
                f"Data recorded: {_format_datetime(result.recorded_at)}",
                f"Report generated: {_format_datetime(result.generated_at)}",
                f"Contact: {_report_contact_line()}",
                REPORT_ACKNOWLEDGEMENT,
                f"Location: {REPORT_LOCATION}",
                "",
                f"Plot PDF: [{result.plot_pdf_path.name}]({_relpath(result.plot_pdf_path, report_path.parent)})",
                "",
                f"![{result.file_path.stem}]({_relpath(result.plot_png_path, report_path.parent)})",
                "",
                f"Text report: [{result.report_path.name}]({_relpath(result.report_path, report_path.parent)})",
                "",
            ]
        )
        if result.control_aux_summary is not None:
            md_lines.extend(
                [
                    f"Reference agreement ({'MISALIGNED' if result.control_aux_misaligned else 'OK'}): control_observed/expected={result.control_event_count}/{result.control_expected_event_count}, aux2_observed/expected={result.aux2_event_count}/{result.aux2_expected_event_count}, selected_reference={result.reference_source}, "
                    f"aligned={result.control_aux_summary.matched_count}, missing_in_aux2={result.control_aux_summary.unmatched_ref}, "
                    f"extra_in_aux2={result.control_aux_summary.unmatched_other}, mean_lag_ms={_format_float(result.control_aux_summary.mean_lag_ms, 2)}, "
                    f"95% CI={_format_ci(result.control_aux_summary, 2)}",
                    "",
                ]
            )
        detail_control_rows: List[List[str]] = []
        detail_aux2_rows: List[List[str]] = []
        sorted_device_rows = sorted(
            result.device_rows,
            key=lambda row: (-_pct(int(row.reference_summary.matched_count), int(result.reference_pattern.expected_event_count)), row.device_label.lower()),
        )
        for row in sorted_device_rows:
            detail_control_rows.append(
                [
                    row.device_label,
                    row.sync_status,
                    row.sample_status,
                    str(row.event_count),
                    str(result.control_expected_event_count),
                    _count_pct_text(int(row.control_summary.matched_count), int(result.control_expected_event_count)),
                    _count_pct_text(int(row.control_summary.unmatched_ref), int(result.control_expected_event_count)),
                    str(row.control_summary.unmatched_other),
                    _format_float(row.control_summary.mean_lag_ms, 2),
                    _format_ci(row.control_summary, 2),
                    _format_float(row.control_summary.mean_abs_residual_ms, 2),
                ]
            )
            detail_aux2_rows.append(
                [
                    row.device_label,
                    row.sync_status,
                    row.sample_status,
                    str(row.event_count),
                    str(result.aux2_expected_event_count),
                    _count_pct_text(int(row.aux2_summary.matched_count), int(result.aux2_expected_event_count)) if row.aux2_summary is not None else "NA",
                    _count_pct_text(int(row.aux2_summary.unmatched_ref), int(result.aux2_expected_event_count)) if row.aux2_summary is not None else "NA",
                    str(row.aux2_summary.unmatched_other) if row.aux2_summary is not None else "NA",
                    _format_float(row.aux2_summary.mean_lag_ms, 2) if row.aux2_summary is not None else "NA",
                    _format_ci(row.aux2_summary, 2) if row.aux2_summary is not None else "NA",
                    _format_float(row.aux2_summary.mean_abs_residual_ms, 2) if row.aux2_summary is not None else "NA",
                ]
            )
        md_lines.extend(
            [
                "**Versus Control**",
                "",
                _markdown_table(
                    [
                        "Probe",
                        "Sync status",
                        "Sample status",
                        "Detected probe events",
                        "Control expected events",
                        "Aligned vs control",
                        "Missing vs control",
                        "Extra vs control",
                        "Mean lag vs control [ms]",
                        "95% CI vs control [ms]",
                        "Mean abs residual vs control [ms]",
                    ],
                    detail_control_rows,
                ),
                "",
                "**Versus AUX 2**",
                "",
                _markdown_table(
                    [
                        "Probe",
                        "Sync status",
                        "Sample status",
                        "Detected probe events",
                        "AUX 2 expected events",
                        "Aligned vs AUX 2",
                        "Missing vs AUX 2",
                        "Extra vs AUX 2",
                        "Mean lag vs AUX 2 [ms]",
                        "95% CI vs AUX 2 [ms]",
                        "Mean abs residual vs AUX 2 [ms]",
                    ],
                    detail_aux2_rows,
                ),
                "",
            ]
        )

    report_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    _write_latex_summary(
        report_dir=common_parent,
        sorted_results=sorted_results,
        file_rollups_control=file_rollups_control,
        probe_rollups_control=probe_rollups_control,
        probe_rollups_aux2=probe_rollups_aux2,
        per_file_table_rows=per_file_control_table_rows,
        per_file_aux2_table_rows=per_file_aux2_table_rows,
        per_probe_table_rows=per_probe_table_rows,
        per_probe_aux2_table_rows=per_probe_aux2_table_rows,
        summary_plot_pdfs=[
            per_file_control_plot_pdf_path,
            per_file_aux2_plot_pdf_path,
            per_probe_control_plot_pdf_path,
            per_probe_aux2_plot_pdf_path,
        ],
    )
    return report_path


def _pick_files_via_dialog() -> List[Path]:
    _configure_qt_runtime()
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    parent = QtWidgets.QWidget()
    parent.setWindowTitle("Analyze OTB4 Buffers")
    QtWidgets.QMessageBox.information(
        parent,
        "Select OTB4 Files",
        "Select one or more OTB4 files to analyze buffer channel alignment against the global control buffer and AUX 2.",
    )
    files, _filter = QtWidgets.QFileDialog.getOpenFileNames(
        parent,
        "Select OTB4 Files",
        str(Path.cwd()),
        "OTB4 files (*.otb4);;All files (*)",
    )
    return [Path(file) for file in files]


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_qt_runtime()
    parser = argparse.ArgumentParser(description="Analyze OTB4 device buffer channels against Syncstation control and AUX 2.")
    parser.add_argument("files", nargs="*", help="Optional OTB4 files. If omitted, a multi-file picker is shown.")
    parser.add_argument("--syncstation-firmware", default=None, help="Optional Syncstation firmware version to include in all generated reports.")
    args = parser.parse_args(argv)

    paths = [Path(arg) for arg in args.files] if args.files else _pick_files_via_dialog()
    if not paths:
        print("No files selected.")
        return 1

    failures: List[Tuple[Path, str]] = []
    results: List[FileAnalysisResult] = []
    for path in paths:
        try:
            result = analyze_file(path, syncstation_firmware_version=args.syncstation_firmware)
            results.append(result)
            print(f"{path.name}: wrote {result.plot_pdf_path}, {result.plot_png_path}, and {result.report_path}")
        except Exception as exc:
            failures.append((path, str(exc)))
            print(f"{path.name}: FAILED - {exc}")

    if len(results) >= 2:
        summary_path = _write_overall_summary(results)
        if summary_path is not None:
            print(f"Overall summary: {summary_path}")

    if failures:
        print("\nFailures:")
        for path, message in failures:
            print(f"- {path}: {message}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
