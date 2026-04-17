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
    _is_counter_track,
    _load_otb4_track_data,
    load_otb4_file,
)


@dataclass
class BufferTrace:
    label: str
    device: str
    values: np.ndarray
    sample_rate: float
    time_sec: np.ndarray
    event_indices: np.ndarray
    event_times_sec: np.ndarray
    is_device_buffer: bool = False
    is_control_buffer: bool = False
    is_aux2: bool = False


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
    control_aux_summary: Optional[AlignmentSummary]
    control_event_count: int
    aux2_event_count: int
    control_aux_misaligned: bool


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


def _best_event_alignment(other_times_sec: Sequence[float], ref_times_sec: Sequence[float], tolerance_sec: float = 0.15) -> AlignmentSummary:
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

    for offset in candidates:
        shifted = other - float(offset)
        i = 0
        j = 0
        lags_ms: List[float] = []
        abs_residual_ms: List[float] = []
        while i < other.size and j < ref.size:
            delta = float(shifted[i] - ref[j])
            if abs(delta) <= tolerance_sec:
                raw_lag_sec = float(other[i] - ref[j])
                lags_ms.append(raw_lag_sec * 1000.0)
                abs_residual_ms.append(abs(delta) * 1000.0)
                i += 1
                j += 1
                continue
            if delta < -tolerance_sec:
                i += 1
            else:
                j += 1
        score = (len(lags_ms), -float(sum(abs_residual_ms)), -abs(float(offset)))
        if best_score is None or score > best_score:
            best_score = score
            best_offset = float(offset)
            best_lags_ms = lags_ms
            best_abs_residual_ms = abs_residual_ms

    mean_lag_ms, ci95_low_ms, ci95_high_ms = _ci95_bounds(best_lags_ms)
    return AlignmentSummary(
        matched_count=len(best_lags_ms),
        unmatched_other=int(other.size - len(best_lags_ms)),
        unmatched_ref=int(ref.size - len(best_lags_ms)),
        offset_sec=best_offset,
        mean_lag_ms=mean_lag_ms,
        ci95_low_ms=ci95_low_ms,
        ci95_high_ms=ci95_high_ms,
        mean_abs_residual_ms=float(np.mean(best_abs_residual_ms)) if best_abs_residual_ms else None,
        max_abs_residual_ms=float(np.max(best_abs_residual_ms)) if best_abs_residual_ms else None,
        matched_lags_ms=best_lags_ms,
    )


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
        if not (is_aux2 or is_control_buffer or is_device_buffer):
            continue
        if is_control_buffer:
            label = "Syncstation Control Buffer"
        elif is_aux2:
            label = "Syncstation AUX 2 [V]"
        else:
            label = f"{device} Buffer"
        if label in seen_labels:
            continue
        seen_labels.add(label)
        events = _detect_buffer_events(raw, sample_rate)
        event_times = time_sec[events] if events.size and time_sec.size else np.asarray([], dtype=float)
        track_traces.append(
            BufferTrace(
                label=label,
                device=device,
                values=converted,
                sample_rate=sample_rate,
                time_sec=np.asarray(time_sec, dtype=float),
                event_indices=events,
                event_times_sec=np.asarray(event_times, dtype=float),
                is_device_buffer=is_device_buffer,
                is_control_buffer=is_control_buffer,
                is_aux2=is_aux2,
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


def _make_device_rows(device_buffers: Sequence[BufferTrace], control_buffer: Optional[BufferTrace], aux2_trace: Optional[BufferTrace]) -> List[DeviceRow]:
    rows: List[DeviceRow] = []
    for trace in device_buffers:
        control_summary = _best_event_alignment(trace.event_times_sec, control_buffer.event_times_sec) if control_buffer is not None else _best_event_alignment([], [])
        aux2_summary = _best_event_alignment(trace.event_times_sec, aux2_trace.event_times_sec) if aux2_trace is not None else None
        rows.append(
            DeviceRow(
                device_label=trace.label,
                event_count=int(trace.event_times_sec.size),
                control_summary=control_summary,
                aux2_summary=aux2_summary,
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
        ref_count = int(result.control_event_count)
        summary = result.file_summary
    else:
        ref_count = int(result.aux2_event_count)
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
        ref_total += int(result.control_event_count if reference == "control" else result.aux2_event_count)
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
    device_rows: Sequence[DeviceRow],
    file_summary: AlignmentSummary,
    file_aux2_summary: Optional[AlignmentSummary],
    control_aux_summary: Optional[AlignmentSummary],
    control_event_count: int,
    aux2_event_count: int,
    info: Dict[str, Any],
) -> None:
    probe_count = len(device_rows)
    control_ideal_total = probe_count * int(control_event_count)
    aux2_ideal_total = probe_count * int(aux2_event_count)
    control_aux_flag = "YES" if _control_aux_misaligned(control_aux_summary, int(control_event_count), int(aux2_event_count)) else "NO"
    lines: List[str] = [
        f"file={file_path}",
        f"device_specific_buffer_channels={probe_count}",
        f"sample_rate_hz={info.get('sample_rate')}",
        f"saved_time_vector_present={bool(info.get('has_saved_time'))}",
        f"sample_count={info.get('sample_count')}",
        f"global_control_buffer_events={control_event_count}",
        f"aux2_events={aux2_event_count}",
        "",
        "[Ideal Summary]",
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
        "",
    ]
    if file_aux2_summary is not None:
        lines.extend(
            [
                "[Reference Agreement]",
                f"control_events={control_event_count}",
                f"aux2_events={aux2_event_count}",
                f"control_aux2_misaligned={control_aux_flag}",
                f"control_aux2_aligned={control_aux_summary.matched_count if control_aux_summary is not None else 'NA'}",
                f"control_aux2_missing_in_aux2={control_aux_summary.unmatched_ref if control_aux_summary is not None else 'NA'}",
                f"control_aux2_extra_in_aux2={control_aux_summary.unmatched_other if control_aux_summary is not None else 'NA'}",
                f"control_aux2_mean_lag_ms={_format_float(control_aux_summary.mean_lag_ms, 3) if control_aux_summary is not None else 'NA'}",
                "",
            ]
        )

    sorted_rows = sorted(
        device_rows,
        key=lambda row: (-_pct(int(row.control_summary.matched_count), int(control_event_count)), row.device_label.lower()),
    )
    for row in sorted_rows:
        control_aligned = _count_pct_text(int(row.control_summary.matched_count), int(control_event_count))
        control_missing = _count_pct_text(int(row.control_summary.unmatched_ref), int(control_event_count))
        aux2_aligned = _count_pct_text(int(row.aux2_summary.matched_count), int(aux2_event_count)) if row.aux2_summary is not None else "NA"
        aux2_missing = _count_pct_text(int(row.aux2_summary.unmatched_ref), int(aux2_event_count)) if row.aux2_summary is not None else "NA"
        lines.extend(
            [
                f"[{row.device_label}]",
                f"detected_events={row.event_count}",
                f"control_reference_events={control_event_count}",
                f"aligned_vs_control={control_aligned}",
                f"missing_vs_control={control_missing}",
                f"extra_vs_control={row.control_summary.unmatched_other}",
                f"mean_lag_vs_control_ms={_format_float(row.control_summary.mean_lag_ms, 3)}",
                f"lag_vs_control_95ci_ms={_format_ci(row.control_summary, 3)}",
                f"mean_abs_residual_vs_control_ms={_format_float(row.control_summary.mean_abs_residual_ms, 3)}",
                f"aux2_reference_events={aux2_event_count}",
                f"aligned_vs_aux2={aux2_aligned}",
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
    device_buffers: Sequence[BufferTrace],
    device_rows: Sequence[DeviceRow],
    control_buffer: Optional[BufferTrace],
    aux2_trace: Optional[BufferTrace],
    info: Dict[str, Any],
) -> None:
    if not device_buffers:
        raise ValueError(f"No device-specific buffer channels found in {file_path.name}")
    row_lookup = {row.device_label: row for row in device_rows}
    count = len(device_buffers)
    fig, axes = plt.subplots(count, 1, sharex=True, figsize=(18, max(3.8, 3.2 * count)))
    if count == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.07, right=0.78, top=0.95, bottom=0.06, hspace=0.55)
    use_time = bool(info.get("has_saved_time"))

    for ax, trace in zip(axes, device_buffers):
        row = row_lookup.get(trace.label)
        x = trace.time_sec if use_time else np.arange(trace.values.size, dtype=float)
        ax.plot(x, _normalize_for_plot(trace.values), color="#005F73", linewidth=1.3, label=trace.label, zorder=3)
        if control_buffer is not None:
            cx = control_buffer.time_sec if use_time else np.arange(control_buffer.values.size, dtype=float)
            ax.plot(cx, _normalize_for_plot(control_buffer.values), color="#111111", linewidth=1.0, alpha=0.5, label=control_buffer.label, zorder=1)
        if aux2_trace is not None:
            axx = aux2_trace.time_sec if use_time else np.arange(aux2_trace.values.size, dtype=float)
            ax.plot(axx, _normalize_for_plot(aux2_trace.values), color="#C16622", linewidth=1.0, alpha=0.5, label=aux2_trace.label, zorder=2)

        if trace.event_indices.size:
            event_x = trace.time_sec[trace.event_indices] if use_time else trace.event_indices.astype(float)
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
            aligned_pct = _pct(int(row.control_summary.matched_count), int(row.control_summary.matched_count + row.control_summary.unmatched_ref))
        else:
            aligned_pct = 0.0
        if row is not None and row.control_summary.mean_lag_ms is not None:
            title = (
                f"{trace.label} | aligned={row.control_summary.matched_count}/{row.control_summary.matched_count + row.control_summary.unmatched_ref} ({aligned_pct:.1f}%) | "
                f"extra={row.control_summary.unmatched_other} | lag={_format_float(row.control_summary.mean_lag_ms, 2)} ms | "
                f"95% CI={_format_ci(row.control_summary, 2)}"
            )
        else:
            title = f"{trace.label} | events={int(trace.event_times_sec.size)}"
        ax.set_title(title)
        ax.set_ylabel("Norm. abs")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

    axes[-1].set_xlabel("Time [s]" if use_time else "Samples")
    fig.suptitle(f"{file_path.name} | device buffers={len(device_buffers)} | control={control_buffer is not None} | aux2={aux2_trace is not None}", fontsize=13)
    fig.savefig(out_pdf_path, format="pdf")
    fig.savefig(out_png_path, dpi=170)
    plt.close(fig)


def analyze_file(path: Path) -> FileAnalysisResult:
    traces, info = _load_buffer_traces(path)
    device_buffers = [trace for trace in traces if trace.is_device_buffer]
    control_buffer = _first_or_none(traces, lambda trace: trace.is_control_buffer)
    aux2_trace = _first_or_none(traces, lambda trace: trace.is_aux2)
    if not device_buffers:
        raise ValueError(f"No device-specific buffer channels found in {path}")

    device_rows = _make_device_rows(device_buffers, control_buffer, aux2_trace)
    file_summary = _pool_alignment_summaries([row.control_summary for row in device_rows])
    file_aux2_summary = _pool_alignment_summaries([row.aux2_summary for row in device_rows if row.aux2_summary is not None]) if aux2_trace is not None else None
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
        device_buffers=device_buffers,
        device_rows=device_rows,
        control_buffer=control_buffer,
        aux2_trace=aux2_trace,
        info=info,
    )
    _save_report(
        report_path,
        file_path=path,
        device_rows=device_rows,
        file_summary=file_summary,
        file_aux2_summary=file_aux2_summary,
        control_aux_summary=control_aux_summary,
        control_event_count=int(control_buffer.event_times_sec.size) if control_buffer is not None else 0,
        aux2_event_count=int(aux2_trace.event_times_sec.size) if aux2_trace is not None else 0,
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
        control_aux_summary=control_aux_summary,
        control_event_count=int(control_buffer.event_times_sec.size) if control_buffer is not None else 0,
        aux2_event_count=int(aux2_trace.event_times_sec.size) if aux2_trace is not None else 0,
        control_aux_misaligned=control_aux_misaligned,
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
    fig.subplots_adjust(left=0.27, right=0.98, top=0.92, bottom=0.06, hspace=0.35)

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
        rf"\date{{Generated { _latex_escape(datetime.now().isoformat(timespec='seconds')) }}}",
        r"\maketitle",
        rf"Selected files: {_latex_escape(len(sorted_results))}\\",
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
                "Control ref events",
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
                "AUX 2 ref events",
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
                rf"Reference agreement ({_latex_escape('MISALIGNED' if result.control_aux_misaligned else 'OK')}): "
                rf"control events={_latex_escape(result.control_event_count)}, "
                rf"aux2 events={_latex_escape(result.aux2_event_count)}, "
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
            key=lambda row: (-_pct(int(row.control_summary.matched_count), int(result.control_event_count)), row.device_label.lower()),
        )
        for row in sorted_device_rows:
            detail_control_rows.append(
                [
                    row.device_label,
                    str(row.event_count),
                    str(result.control_event_count),
                    _count_pct_text(int(row.control_summary.matched_count), int(result.control_event_count)),
                    _count_pct_text(int(row.control_summary.unmatched_ref), int(result.control_event_count)),
                    str(row.control_summary.unmatched_other),
                    _format_float(row.control_summary.mean_lag_ms, 2),
                    _format_ci(row.control_summary, 2),
                    _format_float(row.control_summary.mean_abs_residual_ms, 2),
                ]
            )
            detail_aux2_rows.append(
                [
                    row.device_label,
                    str(row.event_count),
                    str(result.aux2_event_count),
                    _count_pct_text(int(row.aux2_summary.matched_count), int(result.aux2_event_count)) if row.aux2_summary is not None else "NA",
                    _count_pct_text(int(row.aux2_summary.unmatched_ref), int(result.aux2_event_count)) if row.aux2_summary is not None else "NA",
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
                    "Detected probe events",
                    "Control ref events",
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
                    "Detected probe events",
                    "AUX 2 ref events",
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

    md_lines: List[str] = [
        "# OTB4 Buffer Alignment Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Selected files: {len(sorted_results)}",
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
                "Control ref events",
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
                "AUX 2 ref events",
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
                    f"Reference agreement ({'MISALIGNED' if result.control_aux_misaligned else 'OK'}): control_events={result.control_event_count}, aux2_events={result.aux2_event_count}, "
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
            key=lambda row: (-_pct(int(row.control_summary.matched_count), int(result.control_event_count)), row.device_label.lower()),
        )
        for row in sorted_device_rows:
            detail_control_rows.append(
                [
                    row.device_label,
                    str(row.event_count),
                    str(result.control_event_count),
                    _count_pct_text(int(row.control_summary.matched_count), int(result.control_event_count)),
                    _count_pct_text(int(row.control_summary.unmatched_ref), int(result.control_event_count)),
                    str(row.control_summary.unmatched_other),
                    _format_float(row.control_summary.mean_lag_ms, 2),
                    _format_ci(row.control_summary, 2),
                    _format_float(row.control_summary.mean_abs_residual_ms, 2),
                ]
            )
            detail_aux2_rows.append(
                [
                    row.device_label,
                    str(row.event_count),
                    str(result.aux2_event_count),
                    _count_pct_text(int(row.aux2_summary.matched_count), int(result.aux2_event_count)) if row.aux2_summary is not None else "NA",
                    _count_pct_text(int(row.aux2_summary.unmatched_ref), int(result.aux2_event_count)) if row.aux2_summary is not None else "NA",
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
                        "Detected probe events",
                        "Control ref events",
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
                        "Detected probe events",
                        "AUX 2 ref events",
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
    parser = argparse.ArgumentParser(description="Analyze OTB4 device buffer channels against Syncstation control and AUX 2.")
    parser.add_argument("files", nargs="*", help="Optional OTB4 files. If omitted, a multi-file picker is shown.")
    args = parser.parse_args(argv)

    paths = [Path(arg) for arg in args.files] if args.files else _pick_files_via_dialog()
    if not paths:
        print("No files selected.")
        return 1

    failures: List[Tuple[Path, str]] = []
    results: List[FileAnalysisResult] = []
    for path in paths:
        try:
            result = analyze_file(path)
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
