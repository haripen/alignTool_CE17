from __future__ import annotations

import argparse
import concurrent.futures
import gc
import html
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import ezc3d
except Exception as exc:  # pragma: no cover
    raise RuntimeError("ezc3d is required for C3D access.") from exc

from scipy.optimize import linear_sum_assignment
from scipy.io import savemat
from scipy.signal import find_peaks

try:
    from hdsemg_shared.fileio.otb_4_file_io import load_otb4_file, parse_otb4_tracks_xml
except Exception:
    local_vendor = Path(__file__).resolve().parent / "vendor"
    if local_vendor.exists():
        sys.path.insert(0, str(local_vendor))
        from hdsemg_shared.fileio.otb_4_file_io import load_otb4_file, parse_otb4_tracks_xml
    else:
        raise RuntimeError(
            "hdsemg_shared.fileio.otb_4_file_io could not be imported. "
            "This repo expects the vendored copy under ./vendor for portable use."
        )


SYNC_OTB_DEVICE = "Syncstation"
SYNC_OTB_SUBTITLE = "AUX 2"
SYNC_OTB_LABEL = "Syncstation AUX 2 [V]"
SYNC_C3D_LABEL = "Voltage.2_Sync"
SYNC_TSV_EXACT_SUFFIX = "2_Sync [Volt] [sync]"
TSV_RAW_SUFFIX = "[raw]"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_timestamp_from_name(path: Path) -> Optional[datetime]:
    m = re.search(r"_(\d{8})_(\d{6})", path.stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except Exception:
        return None


def _filename_clock_offset_sec(path: Path) -> Optional[float]:
    ts = _parse_timestamp_from_name(path)
    if ts is None:
        return None
    try:
        return float(path.stat().st_mtime - ts.timestamp())
    except Exception:
        return None


def _extract_otb4_tracks(path: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    tmpdir = Path(tempfile.mkdtemp(prefix="otb4_sync_"))
    try:
        if tarfile.is_tarfile(path):
            with tarfile.open(path, "r") as handle:
                handle.extractall(path=tmpdir)
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as handle:
                handle.extractall(path=tmpdir)
        else:
            raise ValueError(f"Unrecognized OTB4 archive format: {path}")
        tracks = parse_otb4_tracks_xml(str(tmpdir / "Tracks_000.xml"))
        return tmpdir, tracks
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def _release_otb4_tmpdir(tmpdir: Path) -> None:
    shutil.rmtree(tmpdir, ignore_errors=True)


def _otb4_track_offsets(tracks: Sequence[Dict[str, Any]], signal_path: str) -> List[Tuple[int, Dict[str, Any]]]:
    offsets: List[Tuple[int, Dict[str, Any]]] = []
    running = 0
    for track in tracks:
        if track["SignalStreamPath"] == signal_path:
            offsets.append((running, track))
            running += int(track["NumberOfChannels"])
    return offsets


def _load_otb4_aux_channel(path: Path, *, device: str, subtitle: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    tmpdir, tracks = _extract_otb4_tracks(path)
    try:
        target = next(
            (
                track
                for track in tracks
                if str(track.get("Device")) == device and str(track.get("SubTitle")) == subtitle
            ),
            None,
        )
        if target is None:
            raise ValueError(f"Missing OTB4 track {device} {subtitle}")

        signal_path = str(target["SignalStreamPath"])
        offsets = _otb4_track_offsets(tracks, signal_path)
        target_offset = next((offset for offset, track in offsets if track is target), None)
        if target_offset is None:
            raise ValueError(f"Unable to locate track offset for {device} {subtitle}")

        sig_file = tmpdir / signal_path
        raw = np.fromfile(sig_file, dtype=np.int16)
        total_channels = sum(int(track["NumberOfChannels"]) for _offset, track in offsets)
        if total_channels <= 0 or raw.size % total_channels != 0:
            raise ValueError(f"Invalid OTB4 signal layout for {signal_path}")
        samples = raw.size // total_channels
        data = raw.reshape((total_channels, samples), order="F").astype(np.float64)

        bits = int(target["ADC_Nbits"])
        gain = float(target["Gain"])
        adc_range = float(target["ADC_Range"])
        conv_v = adc_range / (2 ** bits) / gain if bits > 0 and gain != 0 else 1.0
        x = data[target_offset, :] * conv_v
        fs = float(target["SamplingFrequency"])
        t = np.arange(len(x), dtype=float) / fs
        meta = {
            "label": SYNC_OTB_LABEL,
            "sampling_frequency": fs,
            "device": device,
            "subtitle": subtitle,
            "conversion_v_per_count": conv_v,
            "signal_path": signal_path,
            "channel_offset": target_offset,
        }
        return t, x, meta
    finally:
        _release_otb4_tmpdir(tmpdir)


def _to_ascending_ramp(ramp: np.ndarray) -> np.ndarray:
    ramp = np.asarray(ramp, dtype=np.int64).reshape(-1)
    if ramp.size == 0:
        return ramp.copy()
    prod = ramp[:-1] * ramp[1:]
    prod = prod.copy()
    prod[(prod < 0) & (ramp[1:] > 0)] = 1
    jump_indexes = np.flatnonzero(prod < 0)
    out = np.empty_like(ramp, dtype=np.int64)
    start = 0
    for jump_count, idx in enumerate(jump_indexes, start=1):
        out[start:idx + 1] = ramp[start:idx + 1] + jump_count * 65536
        start = idx + 1
    out[start:] = ramp[start:] + len(jump_indexes) * 65536
    return out


def _logical_groups(logical_array: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(np.asarray(logical_array, dtype=bool))
    if idx.size == 0:
        return []
    dif = np.diff(idx)
    starts = np.r_[0, np.flatnonzero(dif != 1) + 1]
    lengths = np.r_[np.diff(starts), idx.size - starts[-1]]
    return [(int(idx[s]), int(length)) for s, length in zip(starts, lengths)]


def _find_holes_and_jumpbacks(reference_signal: np.ndarray) -> Dict[str, Any]:
    ref = np.asarray(reference_signal, dtype=np.int64).reshape(-1)
    logical_holes = (np.diff(ref) != 1) & ((np.mod(ref[:-1], 65536)) == 0)
    hole_indexes = np.flatnonzero(logical_holes)
    ref_clean = np.delete(ref, hole_indexes)

    difference = np.diff(ref_clean)
    discrepancies = np.flatnonzero(difference <= 0)
    logical_remove = np.zeros(ref_clean.shape[0], dtype=bool)
    for discrepancy_index in discrepancies:
        restarting_point = ref_clean[discrepancy_index + 1]
        involved_difference = restarting_point - ref_clean[:discrepancy_index + 1]
        logical_remove[:discrepancy_index + 1] |= (involved_difference <= 0)

    jumpback_indexes = np.flatnonzero(logical_remove)
    return {
        "hole_indexes": hole_indexes.tolist(),
        "hole_groups": _logical_groups(logical_holes),
        "jumpback_indexes": jumpback_indexes.tolist(),
        "jumpback_groups": _logical_groups(logical_remove),
        "cleaned_ramp": np.delete(ref_clean, jumpback_indexes),
    }


def _find_discrepancy_zones(reference_signal: np.ndarray) -> List[Tuple[int, int]]:
    diff = np.diff(np.asarray(reference_signal, dtype=np.int64))
    discrepancy_indexes = np.flatnonzero(diff > 1).astype(np.int64)
    if discrepancy_indexes.size == 0:
        return []
    deltas = diff[discrepancy_indexes]
    zones: List[Tuple[int, int]] = []
    for i in range(discrepancy_indexes.size):
        hole_dim = int(deltas[i] - 1)
        if hole_dim > 0:
            zones.append((int(discrepancy_indexes[i]), hole_dim))
            if i + 1 < discrepancy_indexes.size:
                discrepancy_indexes[i + 1:] += hole_dim
    return zones


def _is_counter_track(track: Dict[str, Any], raw_track: np.ndarray) -> bool:
    subtitle = str(track.get("SubTitle") or "")
    device = str(track.get("Device") or "")
    if subtitle == "Ramp":
        return True
    if device != "Syncstation" or subtitle:
        return False
    y = np.asarray(raw_track, dtype=np.int64).reshape(-1)
    if y.size < 8:
        return False
    asc = _to_ascending_ramp(y)
    diff = np.diff(asc)
    return bool(np.mean(diff == 1) > 0.8)


def _insert_nan_holes_signal(signal: np.ndarray, zones: Sequence[Tuple[int, int]]) -> np.ndarray:
    out = np.asarray(signal, dtype=float).reshape(-1)
    offset = 0
    for start_idx, hole_dim in zones:
        if int(hole_dim) <= 0:
            continue
        insert_at = int(start_idx) + 1 + offset
        out = np.concatenate([out[:insert_at], np.full(int(hole_dim), np.nan), out[insert_at:]])
        offset += int(hole_dim)
    return out


def _insert_nan_holes_matrix(matrix: np.ndarray, zones: Sequence[Tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    return np.vstack([_insert_nan_holes_signal(row, zones) for row in arr])


def _edge_error_ms(a: Sequence[float], b: Sequence[float]) -> Tuple[Optional[float], Optional[float], int]:
    n = min(len(a), len(b))
    if n <= 0:
        return None, None, 0
    aa = np.asarray(a[:n], dtype=float)
    bb = np.asarray(b[:n], dtype=float)
    delta_ms = np.abs(aa - bb) * 1000.0
    return float(np.mean(delta_ms)), float(np.max(delta_ms)), int(n)


def _load_otb4_track_data(path: Path) -> List[Tuple[int, Dict[str, Any], np.ndarray]]:
    tmpdir, tracks = _extract_otb4_tracks(path)
    try:
        signal_paths = sorted({str(track.get("SignalStreamPath") or "") for track in tracks})
        out: List[Tuple[int, Dict[str, Any], np.ndarray]] = []
        for signal_path in signal_paths:
            if not signal_path:
                continue
            offsets = _otb4_track_offsets(tracks, signal_path)
            raw = np.fromfile(tmpdir / signal_path, dtype=np.int16)
            total_channels = sum(int(track["NumberOfChannels"]) for _off, track in offsets)
            if total_channels <= 0 or raw.size % total_channels != 0:
                continue
            samples = raw.size // total_channels
            data = raw.reshape((total_channels, samples), order="F")
            for offset, track in offsets:
                nchan = int(track.get("NumberOfChannels") or 0)
                out.append((offset, track, data[offset:offset + nchan].copy()))
        return out
    finally:
        _release_otb4_tmpdir(tmpdir)


def _detect_otb4_repair_candidates(path: Path, base_edges: Sequence[float], c3d_edges: Sequence[float], fs: float) -> List[Dict[str, Any]]:
    _t_aux, aux2, _meta = _load_otb4_aux_channel(path, device=SYNC_OTB_DEVICE, subtitle=SYNC_OTB_SUBTITLE)
    base_mean_ms, base_max_ms, base_matched = _edge_error_ms(base_edges, c3d_edges)
    candidates: List[Dict[str, Any]] = []
    for track_offset, track, raw_arr in _load_otb4_track_data(path):
        raw_track = raw_arr[0]
        if not _is_counter_track(track, raw_track):
            continue
        holes = _find_holes_and_jumpbacks(_to_ascending_ramp(raw_track))
        zones = _find_discrepancy_zones(holes["cleaned_ramp"])
        if not zones:
            continue
        repaired_aux = _insert_nan_holes_signal(aux2, zones)
        repaired_edges = _collapse_edge_pairs(_binary_edges(np.nan_to_num(repaired_aux, nan=float(np.nanmedian(repaired_aux))), 0.5 * (float(np.nanmin(np.nan_to_num(repaired_aux, nan=0.0))) + float(np.nanmax(np.nan_to_num(repaired_aux, nan=0.0))))), max_gap_samples=2)
        repaired_edge_times = [float(v) / float(fs) for v in repaired_edges.tolist()]
        mean_ms, max_ms, matched = _edge_error_ms(repaired_edge_times, c3d_edges)
        candidates.append(
            {
                "track_offset": int(track_offset),
                "device": str(track.get("Device") or ""),
                "subtitle": str(track.get("SubTitle") or "") or "Control",
                "hole_groups": [{"start_index": int(s), "length": int(l)} for s, l in holes["hole_groups"]],
                "jumpback_groups": [{"start_index": int(s), "length": int(l)} for s, l in holes["jumpback_groups"]],
                "zones": [{"start_index": int(s), "hole_samples": int(d)} for s, d in zones],
                "samples_added": int(sum(dim for _start, dim in zones)),
                "repaired_edge_times_sec": repaired_edge_times,
                "mean_abs_ms": mean_ms,
                "max_abs_ms": max_ms,
                "matched_edges": matched,
                "base_mean_abs_ms": base_mean_ms,
                "base_max_abs_ms": base_max_ms,
                "base_matched_edges": base_matched,
                "improved": (
                    base_mean_ms is not None
                    and mean_ms is not None
                    and mean_ms + 1.0 < base_mean_ms
                ),
            }
        )
    candidates.sort(
        key=lambda item: (
            1e9 if item["mean_abs_ms"] is None else float(item["mean_abs_ms"]),
            1e9 if item["max_abs_ms"] is None else float(item["max_abs_ms"]),
        )
    )
    return candidates


def _detect_otb4_edges_with_zones(path: Path, zones: Sequence[Tuple[int, int]], fs: float) -> List[float]:
    _t_aux, aux2, _meta = _load_otb4_aux_channel(path, device=SYNC_OTB_DEVICE, subtitle=SYNC_OTB_SUBTITLE)
    repaired_aux = _insert_nan_holes_signal(aux2, _merge_index_zones(zones))
    threshold = 0.5 * (
        float(np.nanmin(np.nan_to_num(repaired_aux, nan=0.0)))
        + float(np.nanmax(np.nan_to_num(repaired_aux, nan=0.0)))
    )
    repaired_edges = _collapse_edge_pairs(
        _binary_edges(np.nan_to_num(repaired_aux, nan=float(np.nanmedian(repaired_aux))), threshold),
        max_gap_samples=2,
    )
    return [float(v) / float(fs) for v in repaired_edges.tolist()]


def _best_edge_alignment_summary(
    otb_edges: Sequence[float],
    c3d_edges: Sequence[float],
    *,
    max_skip_otb: int = 2,
    max_skip_c3d: int = 2,
) -> Dict[str, Any]:
    a = np.asarray(otb_edges, dtype=float)
    b = np.asarray(c3d_edges, dtype=float)
    best: Optional[Dict[str, Any]] = None
    for skip_otb in range(0, min(int(max_skip_otb), max(0, len(a) - 2)) + 1):
        for skip_c3d in range(0, min(int(max_skip_c3d), max(0, len(b) - 2)) + 1):
            aa = a[skip_otb:]
            bb = b[skip_c3d:]
            n = min(len(aa), len(bb))
            if n < 8:
                continue
            shift_sec = float(bb[0] - aa[0])
            delta_ms = (aa[:n] + shift_sec - bb[:n]) * 1000.0
            cand = {
                "otb4_skip": int(skip_otb),
                "c3d_skip": int(skip_c3d),
                "shift_sec": shift_sec,
                "mean_abs_ms": float(np.mean(np.abs(delta_ms))),
                "max_abs_ms": float(np.max(np.abs(delta_ms))),
                "matched_count": int(n),
            }
            if best is None:
                best = cand
                continue
            cur_ok = float(best["mean_abs_ms"]) <= 20.0 and float(best["max_abs_ms"]) <= 50.0
            new_ok = float(cand["mean_abs_ms"]) <= 20.0 and float(cand["max_abs_ms"]) <= 50.0
            if new_ok and not cur_ok:
                best = cand
                continue
            if (new_ok == cur_ok) and (
                (float(cand["mean_abs_ms"]), float(cand["max_abs_ms"]), int(cand["otb4_skip"]) + int(cand["c3d_skip"]))
                < (float(best["mean_abs_ms"]), float(best["max_abs_ms"]), int(best["otb4_skip"]) + int(best["c3d_skip"]))
            ):
                best = cand
    return best or {
        "otb4_skip": 0,
        "c3d_skip": 0,
        "shift_sec": None,
        "mean_abs_ms": None,
        "max_abs_ms": None,
        "matched_count": 0,
    }


def _repair_application_block_reason(
    base_summary: Dict[str, Any],
    selected_row: Dict[str, Any],
    *,
    sample_count: int,
) -> Optional[str]:
    zones = selected_row.get("zones") or []
    if not zones:
        return None
    base_mean = base_summary.get("mean_abs_ms")
    base_max = base_summary.get("max_abs_ms")
    if not isinstance(base_mean, (int, float)) or not isinstance(base_max, (int, float)):
        return None
    base_skip_total = int(base_summary.get("otb4_skip") or 0) + int(base_summary.get("c3d_skip") or 0)
    sel_summary = selected_row.get("summary") or {}
    selected_skip_total = int(sel_summary.get("otb4_skip") or 0) + int(sel_summary.get("c3d_skip") or 0)
    base_matched = int(base_summary.get("matched_count") or 0)
    selected_matched = int(sel_summary.get("matched_count") or 0)
    samples_added = int(selected_row.get("samples_added") or 0)
    insertion_limit = max(50, int(max(0, sample_count) * 0.005))

    excellent_base = float(base_mean) <= 3.0 and float(base_max) <= 10.0
    strong_base = float(base_mean) <= 8.0 and float(base_max) <= 20.0

    reasons: List[str] = []
    if strong_base and selected_skip_total > base_skip_total:
        reasons.append(f"base sync is already strong ({base_mean:.3f}/{base_max:.3f} ms) and repair requires more skipped pulses")
    if strong_base and selected_matched < base_matched:
        reasons.append(f"base sync is already strong and repair reduces matched spikes ({selected_matched} < {base_matched})")
    if strong_base and samples_added > insertion_limit:
        reasons.append(f"base sync is already strong and repair inserts {samples_added} samples (> {insertion_limit})")
    if excellent_base and samples_added > 0:
        reasons.append(f"base sync is already excellent ({base_mean:.3f}/{base_max:.3f} ms) so no gap insertion is needed")

    if reasons:
        return "; ".join(reasons)
    return None


def _repair_info_for_match(match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    alignment = match.get("alignment") or {}
    repair = alignment.get("otb4_repair")
    if isinstance(repair, dict):
        path = repair.get("json_path")
        if isinstance(path, str) and path and Path(path).exists():
            try:
                return json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                return repair
        return repair
    return None


def _otb_label_device(label: str) -> Optional[str]:
    m = re.match(r"^(Syncstation|Muovi\+?\s+\d+|Due\+\s+\d+)\b", str(label).strip())
    return m.group(1) if m else None


def _repair_entry_zones(repair: Optional[Dict[str, Any]]) -> List[Tuple[int, int]]:
    if not isinstance(repair, dict):
        return []
    zones = repair.get("zones") or []
    return [(int(z.get("start_index") or 0), int(z.get("hole_samples") or 0)) for z in zones if int(z.get("hole_samples") or 0) > 0]


def _merge_index_zones(zones: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    intervals = sorted((int(start), int(start) + int(length)) for start, length in zones if int(length) > 0)
    if not intervals:
        return []
    merged: List[List[int]] = [[intervals[0][0], intervals[0][1]]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(start), int(end - start)) for start, end in merged if end > start]


def _device_repairs_for_match(match: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    repair_payload = _repair_info_for_match(match)
    if isinstance(repair_payload, dict) and isinstance(repair_payload.get("device_repairs"), dict):
        device_repairs = repair_payload.get("device_repairs") or {}
    else:
        alignment = match.get("alignment") or {}
        device_repairs = alignment.get("otb4_device_repairs") or {}
    if not isinstance(device_repairs, dict):
        return {}
    return {str(key): value for key, value in device_repairs.items() if isinstance(value, dict)}


def _repair_zones_for_match(match: Dict[str, Any], device: Optional[str] = None) -> List[Tuple[int, int]]:
    zones: List[Tuple[int, int]] = []
    repair = _repair_info_for_match(match)
    if repair:
        zones.extend(_repair_entry_zones(repair))
    if device:
        zones.extend(_repair_entry_zones(_device_repairs_for_match(match).get(device)))
    return _merge_index_zones(zones)


def _max_otb_repaired_length(match: Dict[str, Any], base_len: int, labels: Sequence[str]) -> int:
    lengths = [int(base_len)]
    global_zones = _repair_zones_for_match(match)
    lengths.append(int(base_len) + int(sum(length for _start, length in global_zones)))
    devices = {device for device in (_otb_label_device(label) for label in labels) if device}
    for device in devices:
        zones = _repair_zones_for_match(match, device=device)
        lengths.append(int(base_len) + int(sum(length for _start, length in zones)))
    return max(lengths)


def _binary_edges(x: np.ndarray, threshold: float) -> np.ndarray:
    b = (np.asarray(x, dtype=float).reshape(-1) > threshold).astype(np.int8)
    return np.flatnonzero(np.diff(b) != 0)


def _collapse_edge_pairs(edges: np.ndarray, max_gap_samples: int = 2) -> np.ndarray:
    if len(edges) == 0:
        return edges
    pulses = [int(edges[0])]
    for e in edges[1:]:
        if int(e) - pulses[-1] <= max_gap_samples:
            continue
        pulses.append(int(e))
    return np.asarray(pulses, dtype=int)


def _normalize_trace(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    med = np.nanmedian(x) if np.any(np.isfinite(x)) else 0.0
    x = np.nan_to_num(x, nan=med)
    lo, hi = np.nanpercentile(x, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


def _normalize_unit_interval(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    med = np.nanmedian(arr) if np.any(np.isfinite(arr)) else 0.0
    arr = np.nan_to_num(arr, nan=med)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _dtw_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 1 or len(b) < 1:
        return float("inf")
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    ma = np.median(aa) if np.median(aa) != 0 else 1.0
    mb = np.median(bb) if np.median(bb) != 0 else 1.0
    aa = aa / ma
    bb = bb / mb
    n, m = len(aa), len(bb)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = aa[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - bb[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))


def _orient_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    med = float(np.nanmedian(x)) if np.any(np.isfinite(x)) else 0.0
    pos = np.nan_to_num(x - med, nan=0.0)
    neg = np.nan_to_num(med - x, nan=0.0)
    return pos if float(np.nanmax(pos)) >= float(np.nanmax(neg)) else neg


def _interval_cv(times_sec: Sequence[float]) -> float:
    arr = np.asarray(times_sec, dtype=float)
    if arr.size < 3:
        return float("inf")
    diff = np.diff(arr)
    med = float(np.median(diff))
    if med <= 0:
        return float("inf")
    return float(np.std(diff) / med)


def _detect_sparse_spikes(
    x: np.ndarray,
    sample_rate: float,
    prominence_grid: Sequence[float],
    distance_grid_sec: Sequence[float],
    time_axis: Optional[np.ndarray] = None,
) -> List[float]:
    sig = _orient_signal(x)
    best_score = float("inf")
    best_times: List[float] = []
    axis = np.asarray(time_axis, dtype=float) if time_axis is not None else None
    for prominence in prominence_grid:
        if prominence <= 0:
            continue
        for distance_sec in distance_grid_sec:
            distance = max(1, int(sample_rate * distance_sec))
            peaks, _ = find_peaks(sig, prominence=float(prominence), distance=distance)
            if len(peaks) < 4:
                continue
            times = axis[peaks].tolist() if axis is not None else (peaks / sample_rate).tolist()
            cv = _interval_cv(times)
            count_penalty = 0.0 if 6 <= len(times) <= 24 else abs(len(times) - 16) * 0.2
            score = cv + count_penalty
            if score < best_score:
                best_score = score
                best_times = [float(v) for v in times]
    return best_times


def _detect_tsv_spikes_with_template(x: np.ndarray, t_rel: np.ndarray, template_intervals: Optional[Sequence[float]] = None) -> List[float]:
    sig = _orient_signal(x)
    dt = np.diff(t_rel)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    sample_rate = float(1.0 / np.median(dt)) if len(dt) else 1.0
    std = float(np.std(sig))
    amp = float(np.max(sig)) if len(sig) else 0.0
    prominence_grid = [v for v in [std * 1.5, std * 2.0, std * 2.5, std * 3.0, amp * 0.15, amp * 0.2, amp * 0.25] if v > 0]
    distance_grid = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
    best_score = float("inf")
    best_times: List[float] = []
    template = list(template_intervals or [])
    for prominence in prominence_grid:
        for distance_sec in distance_grid:
            distance = max(1, int(sample_rate * distance_sec))
            peaks, _ = find_peaks(sig, prominence=float(prominence), distance=distance)
            if len(peaks) < 4:
                continue
            times = [float(t_rel[int(i)]) for i in peaks if int(i) < len(t_rel)]
            if len(times) < 4:
                continue
            cv = _interval_cv(times)
            if template:
                score = _dtw_distance(np.diff(times), template) + 0.05 * abs(len(times) - (len(template) + 1))
            else:
                score = cv + 0.05 * abs(len(times) - 16)
            if score < best_score:
                best_score = score
                best_times = times
    return best_times


def _tsv_raw_channels(record: Dict[str, Any]) -> List[str]:
    return [c for c in (record.get("channel_names") or []) if str(c).lower().endswith(TSV_RAW_SUFFIX)]


def _analog_time_from_c3d(c3d_file: ezc3d.c3d) -> np.ndarray:
    point_rate = float(c3d_file["parameters"]["POINT"]["RATE"]["value"][0])
    analog_rate = float(c3d_file["parameters"]["ANALOG"]["RATE"]["value"][0])
    actual_start = int(c3d_file["parameters"]["TRIAL"]["ACTUAL_START_FIELD"]["value"][0])
    n_point_frames = c3d_file["data"]["points"].shape[2]
    point_frames = np.arange(n_point_frames) + actual_start
    time_c3d = point_frames / point_rate - 1.0 / point_rate
    analog_first = int(c3d_file["header"]["analogs"]["first_frame"])
    analog_last = int(c3d_file["header"]["analogs"]["last_frame"])
    n_analog_frames = analog_last - analog_first + 1
    return np.arange(n_analog_frames, dtype=float) / analog_rate + time_c3d[0]


def _unit_scale_to_meters(unit_value: Any) -> float:
    text = str(unit_value).strip().lower()
    if text in {"m", "meter", "meters"}:
        return 1.0
    if text in {"mm", "millimeter", "millimeters"}:
        return 1e-3
    if text in {"cm", "centimeter", "centimeters"}:
        return 1e-2
    return 1.0


def _classify_tsv_raw_channel(raw_col: str) -> Optional[Dict[str, str]]:
    text = str(raw_col).strip()
    lower = text.lower()
    if "/cop/cx " in lower:
        return {"tsv_channel": text, "c3d_kind": "cop", "c3d_channel": "copx", "axis": "x", "label_match": "exact_cop"}
    if "/cop/cy " in lower:
        return {"tsv_channel": text, "c3d_kind": "cop", "c3d_channel": "copy", "axis": "y", "label_match": "exact_cop"}
    return None


def _normalized_label_token(text: str) -> str:
    base = re.sub(r"\s+\[(raw|offset|sync)\]\s*$", "", str(text), flags=re.IGNORECASE)
    base = re.sub(r"\s+\[[^\]]+\]", "", base)
    return re.sub(r"[^a-z0-9]+", "", base.lower())


def _exact_raw_pair_candidates(tsv_record: Dict[str, Any], c3d_record: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for raw_col in _tsv_raw_channels(tsv_record):
        cop = _classify_tsv_raw_channel(raw_col)
        if cop is not None:
            out.append(cop)
            continue
        raw_token = _normalized_label_token(raw_col)
        if not raw_token:
            continue
        for label in c3d_record.get("channel_names") or []:
            if raw_token == _normalized_label_token(label):
                out.append(
                    {
                        "tsv_channel": str(raw_col),
                        "c3d_kind": "analog",
                        "c3d_channel": str(label),
                        "axis": "",
                        "label_match": "exact_label",
                    }
                )
                break
    return out


def _load_c3d_cop_channel(path: Path, channel_name: str) -> Tuple[np.ndarray, np.ndarray]:
    c3d = ezc3d.c3d(str(path), extract_forceplat_data=True)
    platforms = c3d["data"].get("platform", [])
    if not platforms:
        raise ValueError("No force-platform data found in C3D file.")
    platform = platforms[0]
    cop = np.asarray(platform["center_of_pressure"], dtype=float)
    if cop.shape[0] < 2:
        raise ValueError("C3D center_of_pressure does not contain x/y data.")
    idx = {"copx": 0, "copy": 1, "copz": 2}.get(channel_name.lower())
    if idx is None or idx >= cop.shape[0]:
        raise ValueError(f"Unsupported C3D CoP channel: {channel_name}")
    scale = _unit_scale_to_meters(platform.get("unit_position", "m"))
    return _analog_time_from_c3d(c3d), np.asarray(cop[idx], dtype=float) * scale


def _load_c3d_series(path: Path, kind: str, channel_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if kind == "cop":
        return _load_c3d_cop_channel(path, channel_name)
    if kind == "point":
        return _load_c3d_point_channel(path, channel_name)
    return _load_c3d_analog_channel(path, channel_name)


def _resample_shifted_pair(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_mov: np.ndarray,
    y_mov: np.ndarray,
    shift_sec: float,
    step_sec: float = 0.01,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    start = max(float(np.nanmin(t_ref)), float(np.nanmin(t_mov + shift_sec)))
    end = min(float(np.nanmax(t_ref)), float(np.nanmax(t_mov + shift_sec)))
    if not np.isfinite(start) or not np.isfinite(end) or end - start < 10.0:
        return None
    grid = np.arange(start, end, step_sec, dtype=float)
    if len(grid) < 100:
        return None
    a = np.interp(grid, t_ref, y_ref)
    b = np.interp(grid - shift_sec, t_mov, y_mov)
    a = np.nan_to_num(a, nan=float(np.nanmedian(a)))
    b = np.nan_to_num(b, nan=float(np.nanmedian(b)))
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    sd_a = float(np.std(a))
    sd_b = float(np.std(b))
    if sd_a > 0:
        a = a / sd_a
    if sd_b > 0:
        b = b / sd_b
    return a, b


def _best_shifted_corr(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_mov: np.ndarray,
    y_mov: np.ndarray,
    *,
    max_shift_sec: float,
    step_sec: float = 0.01,
) -> Dict[str, Any]:
    best_corr: Optional[float] = None
    best_shift: Optional[float] = None
    best_samples = 0
    max_steps = int(max_shift_sec / step_sec)
    for shift_steps in range(-max_steps, max_steps + 1):
        shift_sec = shift_steps * step_sec
        pair = _resample_shifted_pair(t_ref, y_ref, t_mov, y_mov, shift_sec, step_sec=step_sec)
        if pair is None:
            continue
        a, b = pair
        if len(a) < 100:
            continue
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
        if not np.isfinite(corr):
            continue
        if best_corr is None or abs(corr) > abs(best_corr):
            best_corr = corr
            best_shift = shift_sec
            best_samples = int(len(a))
    return {"corr": best_corr, "shift_sec": best_shift, "samples": best_samples}


def _best_shifted_corr_refined(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_mov: np.ndarray,
    y_mov: np.ndarray,
    *,
    max_shift_sec: float,
    coarse_step_sec: float = 0.01,
    fine_window_sec: float = 0.3,
    fine_step_sec: float = 0.001,
) -> Dict[str, Any]:
    coarse = _best_shifted_corr(
        t_ref,
        y_ref,
        t_mov,
        y_mov,
        max_shift_sec=max_shift_sec,
        step_sec=coarse_step_sec,
    )
    coarse_shift = coarse.get("shift_sec")
    if coarse_shift is None:
        return coarse
    lo = max(-float(max_shift_sec), float(coarse_shift) - float(fine_window_sec))
    hi = min(float(max_shift_sec), float(coarse_shift) + float(fine_window_sec))
    best_corr = coarse.get("corr")
    best_shift = float(coarse_shift)
    best_samples = int(coarse.get("samples") or 0)
    shift = lo
    while shift <= hi + 1e-12:
        pair = _resample_shifted_pair(t_ref, y_ref, t_mov, y_mov, shift, step_sec=0.01)
        if pair is not None:
            a, b = pair
            if len(a) >= 100:
                corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
                if np.isfinite(corr) and (best_corr is None or abs(corr) > abs(float(best_corr))):
                    best_corr = corr
                    best_shift = float(round(shift, 6))
                    best_samples = int(len(a))
        shift += float(fine_step_sec)
    return {"corr": best_corr, "shift_sec": best_shift, "samples": best_samples}


def _raw_quality(abs_corr: float, abs_shift_sec: float) -> str:
    if abs_corr >= 0.999 and abs_shift_sec <= 0.25:
        return "excellent"
    if abs_corr >= 0.995 and abs_shift_sec <= 1.0:
        return "good"
    if abs_corr >= 0.95 and abs_shift_sec <= 5.0:
        return "fair"
    return "poor"


def _effective_raw_quality(raw_result: Dict[str, Any], edge_align: Optional[Dict[str, Any]] = None) -> str:
    quality = str(raw_result.get("quality") or "missing")
    corr = raw_result.get("corr")
    if not isinstance(corr, (int, float)):
        return quality
    if (edge_align or {}).get("basis") == "late_c3d_raw_bridge":
        abs_corr = abs(float(corr))
        if abs_corr >= 0.999:
            return "excellent"
        if abs_corr >= 0.995:
            return "good"
    return quality


def _best_tsv_raw_alignment(tsv_record: Dict[str, Any], c3d_record: Dict[str, Any]) -> Dict[str, Any]:
    pairs = _exact_raw_pair_candidates(tsv_record, c3d_record)
    if not pairs:
        return {
            "quality": "missing",
            "corr": None,
            "lag_sec": None,
            "tsv_channel": None,
            "c3d_channel": None,
            "c3d_kind": None,
            "label_matched": False,
            "samples": 0,
        }

    best: Optional[Dict[str, Any]] = None
    for pair in pairs:
        try:
            t_tsv, y_tsv = _load_tsv_channel(Path(tsv_record["path"]), pair["tsv_channel"])
            t_c3d, y_c3d = _load_c3d_series(Path(c3d_record["path"]), pair["c3d_kind"], pair["c3d_channel"])
        except Exception:
            continue
        result = _best_shifted_corr_refined(
            np.asarray(t_c3d, dtype=float),
            np.asarray(y_c3d, dtype=float),
            np.asarray(t_tsv, dtype=float),
            np.asarray(y_tsv, dtype=float),
            max_shift_sec=8.0,
            coarse_step_sec=0.01,
            fine_window_sec=0.35,
            fine_step_sec=0.001,
        )
        corr = result.get("corr")
        shift_sec = result.get("shift_sec")
        if corr is None or shift_sec is None:
            continue
        abs_corr = abs(float(corr))
        candidate = {
            "quality": _raw_quality(abs_corr, abs(float(shift_sec))),
            "corr": round(float(corr), 6),
            "lag_sec": round(float(shift_sec), 6),
            "tsv_channel": pair["tsv_channel"],
            "c3d_channel": pair["c3d_channel"],
            "c3d_kind": pair["c3d_kind"],
            "label_matched": True,
            "label_match": pair["label_match"],
            "samples": int(result.get("samples") or 0),
        }
        if best is None or abs_corr > abs(float(best["corr"])):
            best = candidate
    if best is None:
        return {
            "quality": "missing",
            "corr": None,
            "lag_sec": None,
            "tsv_channel": None,
            "c3d_channel": None,
            "c3d_kind": None,
            "label_matched": False,
            "samples": 0,
        }
    return best


def _format_file_stamp(dt: Optional[datetime] = None) -> str:
    return (dt or datetime.now()).strftime("%Y%m%d_%H%M%S")


def _source_display_name(record: Dict[str, Any]) -> str:
    return Path(record["path"]).name


def _source_sync_offset(record: Dict[str, Any]) -> float:
    edges = record.get("edge_times_sec") or []
    return float(edges[0]) if edges else 0.0


def _sync_edge_skip_count(match: Dict[str, Any], source: str) -> int:
    skips = ((match.get("alignment") or {}).get("sync_edge_skips") or {})
    try:
        return max(0, int(skips.get(source) or 0))
    except Exception:
        return 0


def _source_sync_offset_for_match(match: Dict[str, Any], source: str) -> float:
    record = match.get(source) or {}
    edges = np.asarray(record.get("edge_times_sec") or [], dtype=float)
    if edges.size == 0:
        return 0.0
    skip = _sync_edge_skip_count(match, source)
    if skip < edges.size:
        return float(edges[skip])
    return float(edges[0])


def _sync_overlay_summary(match: Dict[str, Any]) -> str:
    parts = []
    for key in ("otb4", "c3d", "tsv"):
        rec = match.get(key)
        if rec:
            parts.append(f"{key.upper()}: {_source_display_name(rec)}")

    alignment = match.get("alignment") or {}
    quality = alignment.get("dedicated_sync_quality") or alignment.get("sync_triplet_quality") or "unknown"
    mean_ms = alignment.get("dedicated_sync_mean_abs_ms")
    max_ms = alignment.get("dedicated_sync_max_abs_ms")
    raw_corr = alignment.get("raw_alignment_corr")
    raw_q = alignment.get("raw_alignment_quality")
    raw_tsv = alignment.get("raw_label_match_tsv")
    raw_c3d = alignment.get("raw_label_match_c3d")
    tsv_sync_missing = quality == "missing_tsv_sync"
    raw_text = f"raw={raw_q or 'missing'}"
    if isinstance(raw_corr, (int, float)):
        raw_text += f" corr={raw_corr:.3f}"
    if raw_tsv and raw_c3d:
        raw_tsv_text = Path(str(raw_tsv)).name if "\\" in str(raw_tsv) else str(raw_tsv)
        raw_text += f" [{raw_tsv_text} -> {raw_c3d}]"
    if mean_ms is None or max_ms is None:
        metrics = f"dedicated_sync={quality} {raw_text}"
    else:
        metrics = f"dedicated_sync={quality} mean_abs={mean_ms:.3f} ms max_abs={max_ms:.3f} ms {raw_text}"
    if tsv_sync_missing:
        metrics += f" otb4/c3d={alignment.get('dedicated_sync_pair_quality') or 'n/a'}"
    return " | ".join(["Sync overlay", metrics] + parts)


def _triplet_spike_agreement(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    usable: List[Tuple[str, np.ndarray]] = []
    for rec in records:
        edges = np.asarray(rec.get("edge_times_sec") or [], dtype=float)
        if edges.size >= 2:
            usable.append((str(rec.get("kind", "unknown")), edges - edges[0]))

    if len(usable) < 2:
        return {
            "quality": "insufficient",
            "source_count": len(usable),
            "spike_count": 0,
            "mean_abs_delta_ms": None,
            "max_abs_delta_ms": None,
            "pairwise": {},
        }

    spike_count = min(len(edges) for _kind, edges in usable)
    if spike_count <= 1:
        return {
            "quality": "insufficient",
            "source_count": len(usable),
            "spike_count": spike_count,
            "mean_abs_delta_ms": None,
            "max_abs_delta_ms": None,
            "pairwise": {},
        }

    pairwise: Dict[str, Dict[str, float]] = {}
    pair_deltas: List[np.ndarray] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            name_i, a = usable[i]
            name_j, b = usable[j]
            n = min(len(a), len(b))
            delta = (a[:n] - b[:n]) * 1000.0
            pairwise[f"{name_i}_vs_{name_j}"] = {
                "mean_abs_ms": round(float(np.mean(np.abs(delta))), 6),
                "max_abs_ms": round(float(np.max(np.abs(delta))), 6),
                "count": int(n),
            }
            pair_deltas.append(delta)

    if pair_deltas:
        spread_ms = np.concatenate([np.abs(delta) for delta in pair_deltas])
        mean_abs = float(np.mean(spread_ms))
        max_abs = float(np.max(spread_ms))
    else:
        mean_abs = float("inf")
        max_abs = float("inf")

    if max_abs <= 1.0 and mean_abs <= 0.25:
        quality = "excellent"
    elif max_abs <= 5.0 and mean_abs <= 1.0:
        quality = "good"
    elif max_abs <= 20.0:
        quality = "fair"
    else:
        quality = "poor"

    return {
        "quality": quality,
        "source_count": len(usable),
        "spike_count": spike_count,
        "mean_abs_delta_ms": round(mean_abs, 6),
        "max_abs_delta_ms": round(max_abs, 6),
        "pairwise": pairwise,
    }


def _match_plot_shifts(match: Dict[str, Any]) -> Dict[str, float]:
    c3d_ref = _source_sync_offset_for_match(match, "c3d")
    shifts = {
        "c3d": -c3d_ref,
        "otb4": -_source_sync_offset_for_match(match, "otb4"),
        "tsv": -_source_sync_offset(match.get("tsv") or {}),
    }
    raw_shift = (match.get("alignment") or {}).get("raw_alignment_lag_sec")
    if raw_shift is not None:
        shifts["tsv"] = float(raw_shift) - c3d_ref
    return shifts


def _record_native_time_range(record: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    kind = str(record.get("kind", ""))
    if kind == "tsv":
        duration = record.get("duration_sec")
        if isinstance(duration, (int, float)) and np.isfinite(duration) and duration > 0:
            return 0.0, float(duration)
        return None
    sample_rate = record.get("sample_rate")
    sample_count = record.get("sample_count")
    if isinstance(sample_rate, (int, float)) and isinstance(sample_count, int) and sample_rate and sample_count > 0:
        return 0.0, float(max(sample_count - 1, 0)) / float(sample_rate)
    return None


def _build_inner_merge_alignment(match: Dict[str, Any]) -> Dict[str, Any]:
    shifts = _match_plot_shifts(match)
    source_ranges: Dict[str, Dict[str, Any]] = {}
    aligned_starts: List[float] = []
    aligned_ends: List[float] = []
    aligned_edges: Dict[str, List[float]] = {}

    for source in ("otb4", "c3d", "tsv"):
        record = match.get(source)
        if not record:
            continue
        native_range = _record_native_time_range(record)
        shift_sec = float(shifts.get(source, 0.0))
        sync_anchor_native = _source_sync_offset_for_match(match, source)
        if native_range is None:
            continue
        native_start, native_end = native_range
        aligned_start = native_start + shift_sec
        aligned_end = native_end + shift_sec
        source_ranges[source] = {
            "native_start_sec": round(native_start, 6),
            "native_end_sec": round(native_end, 6),
            "aligned_start_sec": round(aligned_start, 6),
            "aligned_end_sec": round(aligned_end, 6),
            "shift_sec": round(shift_sec, 6),
            "sync_anchor_native_sec": round(sync_anchor_native, 6),
            "sync_anchor_aligned_sec": round(sync_anchor_native + shift_sec, 6),
            "sync_channel": record.get("sync_channel"),
        }
        aligned_starts.append(aligned_start)
        aligned_ends.append(aligned_end)
        edges = np.asarray(record.get("edge_times_sec") or [], dtype=float)
        aligned_edges[source] = [round(float(v), 6) for v in (edges + shift_sec).tolist()]

    overlap_start = max(aligned_starts) if aligned_starts else None
    overlap_end = min(aligned_ends) if aligned_ends else None
    overlap_duration = None
    if overlap_start is not None and overlap_end is not None and overlap_end > overlap_start:
        overlap_duration = overlap_end - overlap_start

    alignment_channels = {
        "otb4_sync": (match.get("otb4") or {}).get("sync_channel"),
        "c3d_sync": (match.get("c3d") or {}).get("sync_channel"),
        "tsv_sync": (match.get("tsv") or {}).get("sync_channel"),
        "tsv_raw": (match.get("alignment") or {}).get("raw_label_match_tsv"),
        "c3d_raw": (match.get("alignment") or {}).get("raw_label_match_c3d"),
        "c3d_raw_kind": (match.get("alignment") or {}).get("raw_label_match_c3d_kind"),
    }

    return {
        "formula": "aligned_time_sec = source_time_sec + shift_sec",
        "shift_basis": "plot_time_shifts_sec",
        "source_time_ranges": source_ranges,
        "aligned_sync_edges_sec": aligned_edges,
        "inner_merge_start_sec": round(float(overlap_start), 6) if overlap_start is not None else None,
        "inner_merge_end_sec": round(float(overlap_end), 6) if overlap_end is not None else None,
        "inner_merge_duration_sec": round(float(overlap_duration), 6) if overlap_duration is not None else None,
        "channels": alignment_channels,
    }


def _aligned_sync_edges(match: Dict[str, Any], source: str) -> np.ndarray:
    rec = match.get(source) or {}
    edges = np.asarray(rec.get("edge_times_sec") or [], dtype=float)
    if edges.size == 0:
        return edges
    return edges + float(_match_plot_shifts(match).get(source, 0.0))


def _spike_match_count(a: np.ndarray, b: np.ndarray, tol_sec: float = 0.05) -> int:
    if a.size == 0 or b.size == 0:
        return 0
    i = 0
    j = 0
    matched = 0
    while i < len(a) and j < len(b):
        delta = float(a[i] - b[j])
        if abs(delta) <= tol_sec:
            matched += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matched


def _select_otb4_c3d_edge_alignment(match: Dict[str, Any]) -> Dict[str, Any]:
    otb_edges = np.asarray(((match.get("otb4") or {}).get("edge_times_sec") or []), dtype=float)
    c3d_edges = np.asarray(((match.get("c3d") or {}).get("edge_times_sec") or []), dtype=float)
    if otb_edges.size < 2 or c3d_edges.size < 2:
        return {
            "otb4_skip": 0,
            "c3d_skip": 0,
            "shift_sec": None,
            "mean_abs_ms": None,
            "max_abs_ms": None,
            "matched_count": 0,
            "basis": "insufficient_edges",
            "late_c3d_supported": False,
        }

    alignment = match.get("alignment") or {}
    raw_quality = str(alignment.get("raw_alignment_quality") or "")
    raw_corr = alignment.get("raw_alignment_corr")
    raw_lag = alignment.get("raw_alignment_lag_sec")
    tsv = match.get("tsv") or {}
    late_c3d_supported = bool(
        tsv
        and isinstance(raw_corr, (int, float))
        and abs(float(raw_corr)) >= 0.999
        and isinstance(raw_lag, (int, float))
        and float(raw_lag) <= -1.0
    )

    def _eval(drop_otb: int, drop_c3d: int) -> Optional[Dict[str, Any]]:
        if drop_otb >= otb_edges.size or drop_c3d >= c3d_edges.size:
            return None
        n = min(len(otb_edges) - drop_otb, len(c3d_edges) - drop_c3d)
        if n < 2:
            return None
        shift_sec = float(c3d_edges[drop_c3d] - otb_edges[drop_otb])
        delta_ms = (otb_edges[drop_otb:drop_otb + n] + shift_sec - c3d_edges[drop_c3d:drop_c3d + n]) * 1000.0
        mean_abs_ms = float(np.mean(np.abs(delta_ms)))
        max_abs_ms = float(np.max(np.abs(delta_ms)))
        lag_penalty = abs(float(raw_lag) - shift_sec) * 30.0 if isinstance(raw_lag, (int, float)) else 0.0
        drop_penalty = 15.0 * float(drop_otb) + 25.0 * float(drop_c3d)
        return {
            "otb4_skip": int(drop_otb),
            "c3d_skip": int(drop_c3d),
            "shift_sec": shift_sec,
            "mean_abs_ms": mean_abs_ms,
            "max_abs_ms": max_abs_ms,
            "matched_count": int(n),
            "score": mean_abs_ms + lag_penalty + drop_penalty,
        }

    base = _eval(0, 0)
    if base is None:
        return {
            "otb4_skip": 0,
            "c3d_skip": 0,
            "shift_sec": None,
            "mean_abs_ms": None,
            "max_abs_ms": None,
            "matched_count": 0,
            "basis": "insufficient_edges",
            "late_c3d_supported": late_c3d_supported,
        }
    best = {**base, "basis": "first_visible_sync"}
    base_is_clearly_bad = bool(
        (
            isinstance(base.get("mean_abs_ms"), (int, float))
            and float(base["mean_abs_ms"]) > 80.0
        )
        or (
            isinstance(base.get("max_abs_ms"), (int, float))
            and float(base["max_abs_ms"]) > 120.0
        )
    )
    if late_c3d_supported and base_is_clearly_bad:
        best_late = best
        # Special case: C3D started late, so OTB4 can have one or more earlier pulses
        # that are absent in the C3D file. Do not skip C3D edges in this mode.
        for drop_otb in range(1, min(3, len(otb_edges) - 1) + 1):
            cand = _eval(drop_otb, 0)
            if cand is None:
                continue
            if cand["score"] < best_late["score"]:
                best_late = {**cand, "basis": "late_c3d_raw_bridge"}
        if (
            best_late["basis"] == "late_c3d_raw_bridge"
            and best_late["mean_abs_ms"] is not None
            and base["mean_abs_ms"] is not None
            and (
                float(best_late["mean_abs_ms"]) <= 25.0
                or float(best_late["mean_abs_ms"]) + 40.0 < float(base["mean_abs_ms"])
            )
        ):
            best = best_late
    best["late_c3d_supported"] = late_c3d_supported
    return best


def _dedicated_sync_agreement(match: Dict[str, Any]) -> Dict[str, Any]:
    tsv_sync_missing = bool(match.get("tsv") and not ((match["tsv"].get("sync_present")) and match["tsv"].get("sync_channel")))
    usable: List[Tuple[str, np.ndarray]] = []
    for key in ("otb4", "c3d", "tsv"):
        rec = match.get(key)
        if rec and rec.get("sync_present") and rec.get("sync_channel"):
            edges = _aligned_sync_edges(match, key)
            skip = _sync_edge_skip_count(match, key)
            if skip > 0 and skip < edges.size:
                edges = edges[skip:]
            if edges.size >= 2:
                usable.append((key, edges))

    if tsv_sync_missing:
        quality = "missing_tsv_sync"
    elif len(usable) < 2:
        quality = "insufficient"
    else:
        quality = "unknown"

    pairwise: Dict[str, Dict[str, Any]] = {}
    deltas: List[np.ndarray] = []
    spike_count = min((len(edges) for _name, edges in usable), default=0)
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            name_i, a = usable[i]
            name_j, b = usable[j]
            n = min(len(a), len(b))
            if n == 0:
                continue
            delta = (a[:n] - b[:n]) * 1000.0
            deltas.append(delta)
            pairwise[f"{name_i}_vs_{name_j}"] = {
                "mean_abs_ms": round(float(np.mean(np.abs(delta))), 6),
                "max_abs_ms": round(float(np.max(np.abs(delta))), 6),
                "count": int(n),
                "matched_spikes_50ms": int(_spike_match_count(a, b, tol_sec=0.05)),
            }

    mean_abs = None
    max_abs = None
    pair_quality = "insufficient"
    if deltas:
        spread_ms = np.concatenate([np.abs(delta) for delta in deltas])
        mean_abs = round(float(np.mean(spread_ms)), 6)
        max_abs = round(float(np.max(spread_ms)), 6)
        if max_abs <= 1.0 and mean_abs <= 0.25:
            pair_quality = "excellent"
        elif max_abs <= 5.0 and mean_abs <= 1.0:
            pair_quality = "good"
        elif max_abs <= 20.0:
            pair_quality = "fair"
        else:
            pair_quality = "poor"
        if not tsv_sync_missing:
            quality = pair_quality

    return {
        "quality": quality,
        "pair_quality": pair_quality,
        "tsv_sync_missing": tsv_sync_missing,
        "source_count": len(usable),
        "spike_count": int(spike_count),
        "mean_abs_delta_ms": mean_abs,
        "max_abs_delta_ms": max_abs,
        "pairwise": pairwise,
    }


@dataclass
class SyncRecord:
    path: str
    kind: str
    sync_channel: Optional[str]
    sync_present: bool
    sync_quality: str
    sample_rate: Optional[float]
    sample_count: int
    threshold: Optional[float]
    edge_count: int
    edge_times_sec: List[float]
    intervals_sec: List[float]
    min_value: Optional[float]
    max_value: Optional[float]
    notes: List[str]
    timestamp: Optional[str] = None
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    duration_sec: Optional[float] = None
    channel_names: Optional[List[str]] = None
    point_channel_names: Optional[List[str]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "sync_channel": self.sync_channel,
            "sync_present": self.sync_present,
            "sync_quality": self.sync_quality,
            "sample_rate": self.sample_rate,
            "sample_count": self.sample_count,
            "threshold": self.threshold,
            "edge_count": self.edge_count,
            "edge_times_sec": self.edge_times_sec,
            "intervals_sec": self.intervals_sec,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "duration_sec": self.duration_sec,
            "channel_names": self.channel_names or [],
            "point_channel_names": self.point_channel_names or [],
        }


def _extract_otb4_record(path: Path) -> SyncRecord:
    data, _time, descs, fs, _name, _size = load_otb4_file(str(path))
    flat = []
    for d in descs:
        try:
            flat.append(str(d[0][0]))
        except Exception:
            flat.append(str(d))

    ts = _parse_timestamp_from_name(path)
    notes: List[str] = []
    try:
        _t_aux, x_aux, aux_meta = _load_otb4_aux_channel(path, device=SYNC_OTB_DEVICE, subtitle=SYNC_OTB_SUBTITLE)
    except Exception as exc:
        return SyncRecord(
            path=str(path),
            kind="otb4",
            sync_channel=None,
            sync_present=False,
            sync_quality="missing",
            sample_rate=float(fs),
            sample_count=int(data.shape[1]),
            threshold=None,
            edge_count=0,
            edge_times_sec=[],
            intervals_sec=[],
            min_value=None,
            max_value=None,
            notes=[f"Unable to read {SYNC_OTB_DEVICE} {SYNC_OTB_SUBTITLE}: {exc}"],
            timestamp=ts.isoformat(sep=" ") if ts else None,
            channel_names=flat,
        )

    x = np.asarray(x_aux, dtype=float).reshape(-1)
    sig = _orient_signal(x)
    amp = float(np.nanmax(sig))
    peaks_sec = _detect_sparse_spikes(
        x,
        float(aux_meta["sampling_frequency"]),
        prominence_grid=[max(float(np.std(sig)) * 6.0, 1e-4), max(amp * 0.15, 1e-4), max(amp * 0.25, 1e-4)],
        distance_grid_sec=[0.8, 1.0, 1.2],
    )
    intervals = np.diff(peaks_sec).tolist() if len(peaks_sec) >= 2 else []
    quality = "certain" if len(peaks_sec) >= 4 else "weak"
    notes.append(f"Selected OTB4 channel {SYNC_OTB_LABEL}.")
    notes.append("OTB4 sync extraction uses metadata-driven Syncstation AUX 2 track in volts.")
    notes.append("The generic OTB4 reader exposes duplicate control-style labels; those are ignored for sync extraction.")
    if len(peaks_sec) < 4:
        notes.append("Edge count is low; sync trace may be truncated or only partially visible.")
    channel_names = [SYNC_OTB_LABEL] + [name for name in flat if name != SYNC_OTB_LABEL]
    return SyncRecord(
        path=str(path),
        kind="otb4",
        sync_channel=SYNC_OTB_LABEL,
        sync_present=True,
        sync_quality=quality,
        sample_rate=float(aux_meta["sampling_frequency"]),
        sample_count=int(len(x)),
        threshold=None,
        edge_count=int(len(peaks_sec)),
        edge_times_sec=[float(v) for v in peaks_sec],
        intervals_sec=[float(v) for v in intervals],
        min_value=float(np.nanmin(x)),
        max_value=float(np.nanmax(x)),
        notes=notes,
        timestamp=ts.isoformat(sep=" ") if ts else None,
        channel_names=channel_names,
    )


def _extract_c3d_record(path: Path) -> SyncRecord:
    c3d = ezc3d.c3d(str(path), extract_forceplat_data=True)
    labels = list(c3d["parameters"]["ANALOG"]["LABELS"]["value"])
    point_labels = list(c3d["parameters"].get("POINT", {}).get("LABELS", {}).get("value", []))
    if "LABELS2" in c3d["parameters"].get("POINT", {}):
        point_labels += list(c3d["parameters"]["POINT"]["LABELS2"]["value"])
    rate = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])
    platform_notes: List[str] = []
    platforms = c3d["data"].get("platform", [])
    if platforms:
        cop = np.asarray(platforms[0].get("center_of_pressure"), dtype=float)
        if cop.ndim == 2 and cop.shape[0] >= 2:
            unit_pos = platforms[0].get("unit_position", "unknown")
            platform_notes.append(f"C3D force-platform CoP available: copx/copy ({unit_pos}).")
        else:
            platform_notes.append("C3D force-platform present but CoP x/y unavailable.")
    else:
        platform_notes.append("No C3D force-platform data found.")
    if SYNC_C3D_LABEL not in labels:
        ts = _parse_timestamp_from_name(path)
        return SyncRecord(
            path=str(path),
            kind="c3d",
            sync_channel=None,
            sync_present=False,
            sync_quality="missing",
            sample_rate=rate,
            sample_count=0,
            threshold=None,
            edge_count=0,
            edge_times_sec=[],
            intervals_sec=[],
            min_value=None,
            max_value=None,
            notes=["Voltage.2_Sync not found in analog labels."] + platform_notes,
            timestamp=ts.isoformat(sep=" ") if ts else None,
            channel_names=labels,
            point_channel_names=point_labels,
        )

    idx = labels.index(SYNC_C3D_LABEL)
    arr = np.asarray(c3d["data"]["analogs"], dtype=float)
    subframes, n_channels, n_frames = arr.shape
    x = np.transpose(arr, (2, 0, 1)).reshape(n_frames * subframes, n_channels)[:, idx]
    sig = _orient_signal(x)
    peaks_sec = _detect_sparse_spikes(
        x,
        rate,
        prominence_grid=[max(float(np.std(sig)) * 6.0, 1e-6), max(float(np.max(sig)) * 0.15, 1e-6), max(float(np.max(sig)) * 0.25, 1e-6)],
        distance_grid_sec=[0.8, 1.0, 1.2],
    )
    quality = "certain" if len(peaks_sec) >= 4 else "weak"
    notes = [f"Selected C3D analog label {SYNC_C3D_LABEL}."] + platform_notes
    if len(peaks_sec) < 4:
        notes.append("Edge count is low; sync trace may be truncated or only partially visible.")
    return SyncRecord(
        path=str(path),
        kind="c3d",
        sync_channel=SYNC_C3D_LABEL,
        sync_present=True,
        sync_quality=quality,
        sample_rate=rate,
        sample_count=int(len(x)),
        threshold=None,
        edge_count=int(len(peaks_sec)),
        edge_times_sec=[float(v) for v in peaks_sec],
        intervals_sec=[float(v) for v in np.diff(peaks_sec).tolist()] if len(peaks_sec) >= 2 else [],
        min_value=float(np.nanmin(x)),
        max_value=float(np.nanmax(x)),
        notes=notes,
        timestamp=None,
        channel_names=labels,
        point_channel_names=point_labels,
    )


def _read_tsv_table(path: Path) -> Tuple[List[str], pd.DataFrame]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()
    cols = header.rstrip("\r\n").split("\t") if header else []
    df = pd.read_csv(path, sep="\t", decimal=",", encoding="utf-8-sig")
    return cols, df


def _tsv_channel_base_name(name: str) -> str:
    text = str(name).strip()
    for suffix in ("[raw]", "[offset]", "[sync]"):
        if text.lower().endswith(suffix.lower()):
            return text[: -len(suffix)].strip()
    return text


def _find_tsv_raw_offset_pair(columns: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    raw_map = {_tsv_channel_base_name(col).lower(): str(col) for col in columns if str(col).lower().endswith("[raw]")}
    offset_map = {_tsv_channel_base_name(col).lower(): str(col) for col in columns if str(col).lower().endswith("[offset]")}
    for key, raw_col in raw_map.items():
        offset_col = offset_map.get(key)
        if offset_col:
            return raw_col, offset_col
    raw_col = next((str(col) for col in columns if str(col).lower().endswith("[raw]")), None)
    offset_col = next((str(col) for col in columns if str(col).lower().endswith("[offset]")), None)
    return raw_col, offset_col


def _tsv_numeric_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.asarray([], dtype=float)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _finite_count(values: np.ndarray) -> int:
    return int(np.sum(np.isfinite(np.asarray(values, dtype=float))))


def _fit_tsv_performed_model(desired: np.ndarray, offset: np.ndarray, performed: np.ndarray) -> Optional[Dict[str, Any]]:
    desired = np.asarray(desired, dtype=float)
    offset = np.asarray(offset, dtype=float)
    performed = np.asarray(performed, dtype=float)
    best: Optional[Dict[str, Any]] = None
    max_lag = max(0, min(12, len(performed) - 2))
    for lag in range(max_lag + 1):
        x_des = desired[:-lag] if lag else desired
        x_off = offset[:-lag] if lag else offset
        y = performed[lag:] if lag else performed
        mask = np.isfinite(x_des) & np.isfinite(x_off) & np.isfinite(y)
        if int(np.sum(mask)) < 10:
            continue
        A = np.column_stack([x_des[mask], x_off[mask], np.ones(int(np.sum(mask)), dtype=float)])
        coef, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
        pred = A @ coef
        rmse = float(np.sqrt(np.mean((y[mask] - pred) ** 2)))
        cur = {
            "lag": int(lag),
            "coef": [float(v) for v in coef],
            "rmse": rmse,
            "sample_count": int(np.sum(mask)),
        }
        if best is None or rmse < float(best["rmse"]):
            best = cur
    return best


def _tsv_desired_signature(df: pd.DataFrame) -> Tuple[float, float]:
    desired = _tsv_numeric_column(df, "desired")
    desired = desired[np.isfinite(desired)]
    if desired.size == 0:
        return 0.0, 0.0
    return float(np.median(desired)), float(np.max(desired) - np.min(desired))


def _nearby_tsv_reference_dirs(path: Path) -> List[Path]:
    dirs: List[Path] = [path.parent]
    for ancestor in list(path.parents)[:4]:
        try:
            children = list(ancestor.iterdir())
        except Exception:
            children = []
        for child in children:
            if child.is_dir() and child.name.lower() == "biofeedbackapp":
                dirs.append(child)
    ordered: List[Path] = []
    seen: set[str] = set()
    for item in dirs:
        key = str(item.resolve())
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


@lru_cache(maxsize=256)
def _best_tsv_performed_model(path_str: str) -> Optional[Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        return None
    _cols, target_df = _read_tsv_table(path)
    raw_col, offset_col = _find_tsv_raw_offset_pair(list(target_df.columns))
    if offset_col is None:
        return None
    target_sig = _tsv_desired_signature(target_df)
    best: Optional[Dict[str, Any]] = None
    for ref_dir in _nearby_tsv_reference_dirs(path):
        for candidate in sorted(ref_dir.glob("*.tsv")):
            if candidate == path:
                continue
            try:
                _ref_cols, ref_df = _read_tsv_table(candidate)
            except Exception:
                continue
            if "performed" not in ref_df.columns or "desired" not in ref_df.columns:
                continue
            cand_raw_col, cand_offset_col = _find_tsv_raw_offset_pair(list(ref_df.columns))
            if cand_offset_col is None:
                continue
            if raw_col and cand_raw_col and _tsv_channel_base_name(cand_raw_col).lower() != _tsv_channel_base_name(raw_col).lower():
                continue
            if _tsv_channel_base_name(cand_offset_col).lower() != _tsv_channel_base_name(offset_col).lower():
                continue
            performed = _tsv_numeric_column(ref_df, "performed")
            if _finite_count(performed) < 10:
                continue
            model = _fit_tsv_performed_model(
                _tsv_numeric_column(ref_df, "desired"),
                _tsv_numeric_column(ref_df, cand_offset_col),
                performed,
            )
            if model is None:
                continue
            cand_sig = _tsv_desired_signature(ref_df)
            similarity = abs(cand_sig[0] - target_sig[0]) + 0.5 * abs(cand_sig[1] - target_sig[1])
            choice = {
                **model,
                "offset_col": offset_col,
                "similarity": float(similarity),
                "reference_path": str(candidate),
            }
            if best is None:
                best = choice
                continue
            best_key = (float(best["similarity"]), float(best["rmse"]))
            cur_key = (float(choice["similarity"]), float(choice["rmse"]))
            if cur_key < best_key:
                best = choice
    return best


def _synth_tsv_performed_values(path: Path, df: pd.DataFrame) -> Optional[np.ndarray]:
    model = _best_tsv_performed_model(str(path))
    if not model:
        return None
    offset_col = str(model.get("offset_col") or "")
    if not offset_col or offset_col not in df.columns or "desired" not in df.columns:
        return None
    desired = _tsv_numeric_column(df, "desired")
    offset = _tsv_numeric_column(df, offset_col)
    lag = int(model.get("lag") or 0)
    coef = np.asarray(model.get("coef") or [], dtype=float)
    if coef.size != 3:
        return None
    usable = len(df) - lag if lag else len(df)
    if usable <= 0:
        return None
    out = np.full(len(df), np.nan, dtype=float)
    A = np.column_stack([desired[:usable], offset[:usable], np.ones(usable, dtype=float)])
    pred = A @ coef
    if lag:
        out[lag:] = pred
        out[:lag] = float(pred[0])
    else:
        out[:] = pred
    return out


def _extract_tsv_record(path: Path) -> SyncRecord:
    cols, df = _read_tsv_table(path)
    ts = _parse_timestamp_from_name(path)
    ts_vals = pd.to_numeric(df["ts"], errors="coerce").to_numpy(dtype=float) if not df.empty and "ts" in df.columns else None
    first_ts = float(ts_vals[0]) if ts_vals is not None and len(ts_vals) else None
    last_ts = float(ts_vals[-1]) if ts_vals is not None and len(ts_vals) else None
    duration_sec = float(pd.to_numeric(df["t_rel"], errors="coerce").iloc[-1]) if not df.empty and "t_rel" in df.columns else None
    if df.empty or not cols:
        return SyncRecord(
            path=str(path),
            kind="tsv",
            sync_channel=None,
            sync_present=False,
            sync_quality="missing",
            sample_rate=None,
            sample_count=0,
            threshold=None,
            edge_count=0,
            edge_times_sec=[],
            intervals_sec=[],
            min_value=None,
            max_value=None,
            notes=["Empty or unreadable TSV file."],
            timestamp=ts.isoformat(sep=" ") if ts else None,
            first_ts=first_ts,
            last_ts=last_ts,
            duration_sec=duration_sec,
            channel_names=cols,
        )

    sync_cols = [c for c in df.columns if str(c).endswith(SYNC_TSV_EXACT_SUFFIX)]
    if not sync_cols:
        raw_cols = [c for c in df.columns if str(c).lower().endswith("[raw]")]
        return SyncRecord(
            path=str(path),
            kind="tsv",
            sync_channel=None,
            sync_present=False,
            sync_quality="missing",
            sample_rate=None,
            sample_count=int(len(df)),
            threshold=None,
            edge_count=0,
            edge_times_sec=[],
            intervals_sec=[],
            min_value=None,
            max_value=None,
            notes=["Sync column not present in TSV header."] + ([f"Available TSV raw column(s): {', '.join(raw_cols)}"] if raw_cols else []),
            timestamp=ts.isoformat(sep=" ") if ts else None,
            first_ts=first_ts,
            last_ts=last_ts,
            duration_sec=duration_sec,
            channel_names=list(df.columns),
        )

    col = sync_cols[0]
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(x) if np.any(np.isfinite(x)) else 0.0
    x = np.nan_to_num(x, nan=med)
    minv = float(np.nanmin(x))
    maxv = float(np.nanmax(x))
    notes = [f"Selected TSV sync column {col}."]
    if maxv - minv < 0.05:
        notes.append("Low dynamic range detected; sync is likely weak or missing.")
    if "t_rel" in df.columns:
        t_rel = pd.to_numeric(df["t_rel"], errors="coerce").to_numpy(dtype=float)
        if len(t_rel) > 2 and np.all(np.isfinite(t_rel)) and np.median(np.diff(t_rel)) > 0:
            sample_rate = float(1.0 / np.median(np.diff(t_rel)))
        else:
            sample_rate = None
    else:
        t_rel = None
        sample_rate = None
    edge_axis = t_rel if t_rel is not None else np.arange(len(x), dtype=float)
    peaks_sec = _detect_tsv_spikes_with_template(x, edge_axis)
    if len(peaks_sec) < 2:
        notes.append("No reliable sync spikes found in TSV exact sync column.")
    quality = "certain" if len(peaks_sec) >= 4 and maxv - minv >= 0.01 else "weak"
    return SyncRecord(
        path=str(path),
        kind="tsv",
        sync_channel=col,
        sync_present=True,
        sync_quality=quality,
        sample_rate=sample_rate,
        sample_count=int(len(x)),
        threshold=None,
        edge_count=int(len(peaks_sec)),
        edge_times_sec=[float(v) for v in peaks_sec],
        intervals_sec=[float(v) for v in np.diff(peaks_sec).tolist()] if len(peaks_sec) >= 2 else [],
        min_value=minv,
        max_value=maxv,
        notes=notes,
        timestamp=ts.isoformat(sep=" ") if ts else None,
        first_ts=first_ts,
        last_ts=last_ts,
        duration_sec=duration_sec,
        channel_names=list(df.columns),
    )


def _apply_otb4_repairs(matches: List[Dict[str, Any]], log_lines: List[str]) -> None:
    for match in matches:
        if not (match.get("otb4") and match.get("c3d")):
            continue
        otb_path = Path(match["otb4"]["path"])
        fs = float(match["otb4"].get("sample_rate") or 2000.0)
        c3d_edges = match["c3d"].get("edge_times_sec") or []
        base_pairwise = ((_triplet_spike_agreement([match["otb4"], match["c3d"]]).get("pairwise") or {}).get("otb4_vs_c3d") or {})
        base_mean_ms = base_pairwise.get("mean_abs_ms")
        base_max_ms = base_pairwise.get("max_abs_ms")
        candidates = _detect_otb4_repair_candidates(
            otb_path,
            match["otb4"].get("edge_times_sec") or [],
            c3d_edges,
            fs,
        )
        if not candidates:
            continue

        def _zones_to_infos(zones_in: Sequence[Tuple[int, int]], shift_sec: float) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for start_index, hole_samples in zones_in:
                if int(hole_samples) <= 0:
                    continue
                out.append(
                    {
                        "start_index": int(start_index),
                        "hole_samples": int(hole_samples),
                        "start_sec": round(int(start_index) / fs, 6),
                        "end_sec": round((int(start_index) + int(hole_samples)) / fs, 6),
                        "aligned_start_sec": round(int(start_index) / fs + float(shift_sec), 6),
                        "aligned_end_sec": round((int(start_index) + int(hole_samples)) / fs + float(shift_sec), 6),
                    }
                )
            return out

        device_repairs: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            device = str(candidate.get("device") or "").strip()
            if not device or device in device_repairs:
                continue
            device_repairs[device] = {
                "applied": False,
                "repair_source": "matlab_ramp_gap_detection",
                "device": device,
                "subtitle": candidate.get("subtitle"),
                "track_offset": candidate.get("track_offset"),
                "candidate_mean_abs_ms": candidate.get("mean_abs_ms"),
                "candidate_max_abs_ms": candidate.get("max_abs_ms"),
                "matched_edges": candidate.get("matched_edges"),
                "samples_added": candidate.get("samples_added"),
                "zones": _zones_to_infos(
                    _merge_index_zones(
                        [
                            (int(zone.get("start_index") or 0), int(zone.get("hole_samples") or 0))
                            for zone in (candidate.get("zones") or [])
                            if int(zone.get("hole_samples") or 0) > 0
                        ]
                    ),
                    float(_match_plot_shifts(match).get("otb4", 0.0)),
                ),
                "hole_groups": candidate.get("hole_groups") or [],
                "jumpback_groups": candidate.get("jumpback_groups") or [],
            }
        if device_repairs:
            match.setdefault("alignment", {})["otb4_device_repairs"] = device_repairs

        def _candidate_zone_tuples(candidate_in: Dict[str, Any]) -> List[Tuple[int, int]]:
            return _merge_index_zones(
                [
                    (int(zone.get("start_index") or 0), int(zone.get("hole_samples") or 0))
                    for zone in (candidate_in.get("zones") or [])
                    if int(zone.get("hole_samples") or 0) > 0
                ]
            )

        def _summary_key(summary_in: Dict[str, Any]) -> Tuple[float, float, float, int, int]:
            mean_val = summary_in.get("mean_abs_ms")
            max_val = summary_in.get("max_abs_ms")
            ok = (
                isinstance(mean_val, (int, float))
                and isinstance(max_val, (int, float))
                and float(mean_val) <= 20.0
                and float(max_val) <= 50.0
            )
            return (
                0.0 if ok else 1.0,
                float(mean_val) if isinstance(mean_val, (int, float)) else 1e9,
                float(max_val) if isinstance(max_val, (int, float)) else 1e9,
                int(summary_in.get("otb4_skip") or 0) + int(summary_in.get("c3d_skip") or 0),
                -int(summary_in.get("matched_count") or 0),
            )

        option_rows: List[Dict[str, Any]] = []
        base_summary = _best_edge_alignment_summary(match["otb4"].get("edge_times_sec") or [], c3d_edges)
        option_rows.append(
            {
                "name": "none",
                "zones": [],
                "summary": base_summary,
                "edges": list(match["otb4"].get("edge_times_sec") or []),
                "samples_added": 0,
                "components": [],
                "device": None,
                "subtitle": None,
                "track_offset": None,
                "hole_groups": [],
                "jumpback_groups": [],
            }
        )
        for candidate in candidates:
            zones = _candidate_zone_tuples(candidate)
            option_rows.append(
                {
                    "name": f"{candidate.get('device')} {candidate.get('subtitle')}",
                    "zones": zones,
                    "summary": _best_edge_alignment_summary(
                        _detect_otb4_edges_with_zones(otb_path, zones, fs),
                        c3d_edges,
                    ),
                    "edges": _detect_otb4_edges_with_zones(otb_path, zones, fs),
                    "samples_added": int(sum(length for _start, length in zones)),
                    "components": [candidate],
                    "device": candidate.get("device"),
                    "subtitle": candidate.get("subtitle"),
                    "track_offset": candidate.get("track_offset"),
                    "hole_groups": candidate.get("hole_groups") or [],
                    "jumpback_groups": candidate.get("jumpback_groups") or [],
                }
            )
        primary = candidates[0]
        primary_zones = _candidate_zone_tuples(primary)
        for extra in candidates[1:]:
            extra_zones = _candidate_zone_tuples(extra)
            zones = _merge_index_zones(primary_zones + extra_zones)
            edges = _detect_otb4_edges_with_zones(otb_path, zones, fs)
            option_rows.append(
                {
                    "name": f"{primary.get('device')}+{extra.get('device')}",
                    "zones": zones,
                    "summary": _best_edge_alignment_summary(edges, c3d_edges),
                    "edges": edges,
                    "samples_added": int(sum(length for _start, length in zones)),
                    "components": [primary, extra],
                    "device": f"{primary.get('device')} + {extra.get('device')}",
                    "subtitle": f"{primary.get('subtitle')} + {extra.get('subtitle')}",
                    "track_offset": [primary.get("track_offset"), extra.get("track_offset")],
                    "hole_groups": list(primary.get("hole_groups") or []) + list(extra.get("hole_groups") or []),
                    "jumpback_groups": list(primary.get("jumpback_groups") or []) + list(extra.get("jumpback_groups") or []),
                }
            )

        selected = min(option_rows, key=lambda row: _summary_key(row["summary"]))
        if selected["zones"]:
            selected_zone_set = set(selected["zones"])
            for candidate in candidates:
                extra_zones = _candidate_zone_tuples(candidate)
                if not extra_zones:
                    continue
                if set(extra_zones).issubset(selected_zone_set):
                    continue
                union_zones = _merge_index_zones(selected["zones"] + extra_zones)
                union_edges = _detect_otb4_edges_with_zones(otb_path, union_zones, fs)
                union_summary = _best_edge_alignment_summary(union_edges, c3d_edges)
                union_row = {
                    "name": f"{selected['name']}+{candidate.get('device')}",
                    "zones": union_zones,
                    "summary": union_summary,
                    "edges": union_edges,
                    "samples_added": int(sum(length for _start, length in union_zones)),
                    "components": list(selected.get("components") or []) + [candidate],
                    "device": " + ".join(
                        [
                            part
                            for part in [str(selected.get("device") or "").strip(), str(candidate.get("device") or "").strip()]
                            if part
                        ]
                    ),
                    "subtitle": " + ".join(
                        [
                            part
                            for part in [str(selected.get("subtitle") or "").strip(), str(candidate.get("subtitle") or "").strip()]
                            if part
                        ]
                    ),
                    "track_offset": [selected.get("track_offset"), candidate.get("track_offset")],
                    "hole_groups": list(selected.get("hole_groups") or []) + list(candidate.get("hole_groups") or []),
                    "jumpback_groups": list(selected.get("jumpback_groups") or []) + list(candidate.get("jumpback_groups") or []),
                }
                if _summary_key(union_summary) < _summary_key(selected["summary"]):
                    selected = union_row
        repair_block_reason = _repair_application_block_reason(
            base_summary,
            selected,
            sample_count=int(match["otb4"].get("sample_count") or 0),
        )
        if repair_block_reason:
            selected = option_rows[0]
        shift = float(_match_plot_shifts(match).get("otb4", 0.0))
        repair = {
            "applied": bool(selected["zones"]),
            "repair_source": "matlab_ramp_gap_detection",
            "device": selected.get("device"),
            "subtitle": selected.get("subtitle"),
            "track_offset": selected.get("track_offset"),
            "base_mean_abs_ms": base_mean_ms,
            "base_max_abs_ms": base_max_ms,
            "candidate_mean_abs_ms": ((candidates[0] if candidates else {}).get("mean_abs_ms")),
            "candidate_max_abs_ms": ((candidates[0] if candidates else {}).get("max_abs_ms")),
            "repaired_mean_abs_ms": selected["summary"].get("mean_abs_ms") if selected["zones"] else None,
            "repaired_max_abs_ms": selected["summary"].get("max_abs_ms") if selected["zones"] else None,
            "matched_edges": selected["summary"].get("matched_count"),
            "samples_added": selected["samples_added"],
            "zones": _zones_to_infos(selected["zones"], shift),
            "hole_groups": selected.get("hole_groups") or [],
            "jumpback_groups": selected.get("jumpback_groups") or [],
            "repaired_edge_times_sec": selected["edges"] if selected["zones"] else [],
            "selected_option": selected["name"],
            "selected_otb4_skip": selected["summary"].get("otb4_skip"),
            "selected_c3d_skip": selected["summary"].get("c3d_skip"),
            "selected_shift_sec": selected["summary"].get("shift_sec"),
            "selected_matched_count": selected["summary"].get("matched_count"),
            "component_devices": [str(comp.get("device") or "") for comp in selected.get("components") or []],
            "component_subtitles": [str(comp.get("subtitle") or "") for comp in selected.get("components") or []],
            "block_reason": repair_block_reason,
        }
        match.setdefault("alignment", {})["otb4_repair"] = repair
        if not repair["applied"]:
            best_candidate = candidates[0]
            best_candidate_zones = _candidate_zone_tuples(best_candidate)
            match["otb4"].setdefault("notes", []).append(
                f"OTB4 gap candidate detected from {best_candidate['device']} {best_candidate['subtitle']}: "
                f"{len(best_candidate_zones)} zones, +{best_candidate.get('samples_added')} samples, not auto-applied."
            )
            log_lines.append(
                f"[repair] match {match['match_id']:03d}: detected OTB4 gap candidate {candidates[0]['device']} {candidates[0]['subtitle']} "
                f"with {len(candidates[0].get('zones') or [])} zones and +{repair['samples_added']} samples, but kept mean_abs {candidates[0].get('mean_abs_ms')} ms "
                f"vs base {base_mean_ms} ms so it was not applied."
            )
            if repair_block_reason:
                match["otb4"].setdefault("notes", []).append(f"OTB4 repair blocked: {repair_block_reason}.")
                log_lines.append(
                    f"[repair] match {match['match_id']:03d}: blocked repair candidate because {repair_block_reason}."
                )
            continue
        match["otb4"]["edge_times_sec"] = list(repair["repaired_edge_times_sec"])
        match["otb4"]["edge_count"] = len(match["otb4"]["edge_times_sec"])
        match["otb4"]["intervals_sec"] = [float(v) for v in np.diff(np.asarray(match["otb4"]["edge_times_sec"], dtype=float)).tolist()] if len(match["otb4"]["edge_times_sec"]) >= 2 else []
        match["otb4"]["sample_count"] = int(match["otb4"].get("sample_count") or 0) + int(repair["samples_added"] or 0)
        match["otb4"].setdefault("notes", []).append(
            f"OTB4 repair applied from {repair['device']} {repair['subtitle']}: +{repair['samples_added']} NaN gap samples."
        )

        final_edge_align = {
            **selected["summary"],
            "basis": "repair_gap_alignment",
            "late_c3d_supported": bool(((match.get("alignment") or {}).get("otb4_c3d_edge_alignment") or {}).get("late_c3d_supported")),
        }
        match["alignment"]["sync_edge_skips"] = {
            "otb4": int(final_edge_align.get("otb4_skip") or 0),
            "c3d": int(final_edge_align.get("c3d_skip") or 0),
            "tsv": int((((match.get("alignment") or {}).get("sync_edge_skips") or {}).get("tsv") or 0)),
        }
        match["alignment"]["otb4_c3d_edge_alignment"] = final_edge_align
        repair["zones"] = _zones_to_infos(selected["zones"], float(_match_plot_shifts(match).get("otb4", 0.0)))

        raw_result = {
            "quality": (match.get("alignment") or {}).get("raw_alignment_quality"),
            "corr": (match.get("alignment") or {}).get("raw_alignment_corr"),
            "lag_sec": (match.get("alignment") or {}).get("raw_alignment_lag_sec"),
        }
        match["alignment"]["raw_alignment_quality"] = _effective_raw_quality(raw_result, final_edge_align)
        dedicated = _dedicated_sync_agreement(match)
        match["alignment"]["dedicated_sync_quality"] = dedicated.get("quality")
        match["alignment"]["dedicated_sync_pair_quality"] = dedicated.get("pair_quality")
        match["alignment"]["dedicated_sync_mean_abs_ms"] = dedicated.get("mean_abs_delta_ms")
        match["alignment"]["dedicated_sync_max_abs_ms"] = dedicated.get("max_abs_delta_ms")
        match["alignment"]["dedicated_sync_spike_count"] = dedicated.get("spike_count")
        match["alignment"]["dedicated_sync_pairwise"] = dedicated.get("pairwise")
        match["alignment"]["plot_time_shifts_sec"] = _match_plot_shifts(match)
        match["alignment"]["inner_merge"] = _build_inner_merge_alignment(match)
        match["alignment"]["sync_triplet_quality"] = dedicated.get("quality")
        match["alignment"]["sync_triplet_spike_mean_abs_ms"] = dedicated.get("mean_abs_delta_ms")
        match["alignment"]["sync_triplet_spike_max_abs_ms"] = dedicated.get("max_abs_delta_ms")
        match["alignment"]["sync_triplet_spike_count"] = dedicated.get("spike_count")
        match["alignment"]["sync_triplet_pairwise"] = dedicated.get("pairwise")
        log_lines.append(
            f"[repair] match {match['match_id']:03d}: applied OTB4 repair using {repair['device']} {repair['subtitle']} "
            f"+{repair['samples_added']} samples, mean_abs {base_mean_ms} -> {repair['repaired_mean_abs_ms']} ms, "
            f"max_abs {base_max_ms} -> {repair['repaired_max_abs_ms']} ms."
        )


def _pair_records(left: List[SyncRecord], right: List[SyncRecord], skip_penalty: float = 0.25) -> List[Tuple[SyncRecord, SyncRecord, float]]:
    if not left or not right:
        return []
    n_left = len(left)
    n_right = len(right)
    cost = np.full((n_left, n_right), 1e6, dtype=float)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            cur_cost = _pair_cost_value(a.intervals_sec, b.intervals_sec, weak_pair_cost=float(skip_penalty))
            if cur_cost is None:
                continue
            cost[i, j] = float(cur_cost)

    # Preserve acquisition order: files were recorded sequentially, so pairings should not cross in time.
    dp = np.full((n_left + 1, n_right + 1), np.inf, dtype=float)
    move: List[List[Optional[Tuple[str, int, int]]]] = [[None] * (n_right + 1) for _ in range(n_left + 1)]
    dp[0, 0] = 0.0
    for i in range(n_left + 1):
        for j in range(n_right + 1):
            cur = float(dp[i, j])
            if not np.isfinite(cur):
                continue
            if i < n_left and cur + float(skip_penalty) < dp[i + 1, j]:
                dp[i + 1, j] = cur + float(skip_penalty)
                move[i + 1][j] = ("skip_left", i, j)
            if j < n_right and cur + float(skip_penalty) < dp[i, j + 1]:
                dp[i, j + 1] = cur + float(skip_penalty)
                move[i][j + 1] = ("skip_right", i, j)
            if i < n_left and j < n_right and cur + float(cost[i, j]) < dp[i + 1, j + 1]:
                dp[i + 1, j + 1] = cur + float(cost[i, j])
                move[i + 1][j + 1] = ("pair", i, j)

    i = n_left
    j = n_right
    out: List[Tuple[SyncRecord, SyncRecord, float]] = []
    while i > 0 or j > 0:
        step = move[i][j]
        if step is None:
            break
        kind, prev_i, prev_j = step
        if kind == "pair":
            out.append((left[prev_i], right[prev_j], float(cost[prev_i, prev_j])))
        i, j = prev_i, prev_j
    out.reverse()
    return out


def _summarize_pair_cost(cost: float) -> str:
    if cost < 0.02:
        return "certain"
    if cost < 0.05:
        return "probable"
    return "uncertain"


def _parse_record_dt(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _c3d_review_sort_key(match: Dict[str, Any]) -> Tuple[int, str]:
    c3d_path = Path(str((match.get("c3d") or {}).get("path") or ""))
    m = re.match(r"file_(\d+)$", c3d_path.stem, flags=re.IGNORECASE)
    if m:
        return (int(m.group(1)), c3d_path.name.lower())
    return (10**9, c3d_path.name.lower())


def _iter_source_files(folder: Path, suffix: str, excluded_dir_names: Optional[Sequence[str]] = None) -> List[Path]:
    excluded = {name.lower() for name in (excluded_dir_names or ("matched",))}
    results: List[Path] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d.lower() not in excluded]
        for file_name in files:
            if file_name.lower().endswith(suffix.lower()):
                results.append(Path(root) / file_name)
    results.sort()
    return results


def _pair_cost_value(
    left_intervals: Sequence[float],
    right_intervals: Sequence[float],
    *,
    weak_pair_cost: float = 0.25,
) -> Optional[float]:
    if not left_intervals or not right_intervals:
        if not left_intervals and not right_intervals:
            return float(weak_pair_cost)
        return None
    aa = np.asarray(left_intervals, dtype=float)
    bb = np.asarray(right_intervals, dtype=float)
    n = min(len(aa), len(bb))
    if n == 0:
        return None
    aa = aa[:n]
    bb = bb[:n]
    ma = np.median(aa) if np.median(aa) > 0 else 1.0
    mb = np.median(bb) if np.median(bb) > 0 else 1.0
    aa = aa / ma
    bb = bb / mb
    base = float(np.sqrt(np.mean((aa - bb) ** 2)))
    base += 0.01 * abs(len(left_intervals) - len(right_intervals))
    return base


def _infer_root_filename_clock_offset_sec(folder: Path) -> Optional[float]:
    samples: List[float] = []
    for suffix in (".tsv", ".otb4"):
        for path in _iter_source_files(folder, suffix, excluded_dir_names=("matched", "accepted", "__pycache__", ".git")):
            delta = _filename_clock_offset_sec(path)
            if delta is None:
                continue
            # Keep only plausible timezone-like offsets and ignore small write delays.
            if abs(delta) <= 18.0 * 3600.0:
                samples.append(float(delta))
    if not samples:
        return None
    return float(np.median(np.asarray(samples, dtype=float)))


def _auto_accept_match(match: Dict[str, Any]) -> bool:
    alignment = match.get("alignment") or {}
    raw_quality = alignment.get("raw_alignment_quality")
    raw_corr = alignment.get("raw_alignment_corr")
    edge_align = alignment.get("otb4_c3d_edge_alignment") or {}
    pairwise = alignment.get("dedicated_sync_pairwise") or {}
    otb_c3d = pairwise.get("otb4_vs_c3d") or pairwise.get("c3d_vs_otb4") or {}
    otb_c3d_count = int(otb_c3d.get("count") or 0)
    otb_c3d_matched = int(otb_c3d.get("matched_spikes_50ms") or 0)
    otb_c3d_mean = otb_c3d.get("mean_abs_ms")
    otb_c3d_max = otb_c3d.get("max_abs_ms")
    otb_c3d_sync_ok = False
    if otb_c3d_count > 0:
        if otb_c3d_matched >= otb_c3d_count:
            otb_c3d_sync_ok = True
        elif isinstance(otb_c3d_mean, (int, float)) and isinstance(otb_c3d_max, (int, float)):
            otb_c3d_sync_ok = float(otb_c3d_mean) <= 20.0 and float(otb_c3d_max) <= 50.0
    raw_ok = raw_quality == "excellent" or (
        raw_quality == "good" and isinstance(raw_corr, (int, float)) and abs(float(raw_corr)) >= 0.995
    )
    if not raw_ok and edge_align.get("basis") == "late_c3d_raw_bridge" and isinstance(raw_corr, (int, float)):
        raw_ok = abs(float(raw_corr)) >= 0.999
    return (
        match.get("certainty") == "certain"
        and raw_ok
        and otb_c3d_sync_ok
    )


def _apply_review_defaults(match: Dict[str, Any]) -> None:
    review = dict(match.get("review") or {})
    auto_accept = bool(review.get("auto_accept")) if "auto_accept" in review else _auto_accept_match(match)
    if "user_decision" not in review:
        review["user_decision"] = None
    if "user_touched" not in review:
        review["user_touched"] = False
    review["auto_accept"] = auto_accept
    review["final_accept"] = bool(review["user_decision"]) if review["user_decision"] is not None else auto_accept
    review["reviewed"] = review["user_decision"] is not None
    review["status_key"] = _review_status_key_from_review(review)
    review["status_color"] = _review_status_color_from_review(review)
    review["status_text"] = _review_status_label_from_review(review)
    review["decision_source"] = "user" if review["user_decision"] is not None else "automatic"
    match["review"] = review
    match["accepted"] = review["final_accept"]


def _match_tsv_records(
    matches: List[Dict[str, Any]],
    tsv_records: List[SyncRecord],
    log_lines: List[str],
    *,
    filename_clock_offset_sec: float = 0.0,
) -> List[Dict[str, Any]]:
    candidates: List[SyncRecord] = [
        rec
        for rec in tsv_records
        if rec.first_ts is not None
        and rec.last_ts is not None
        and rec.duration_sec is not None
        and rec.duration_sec >= 5.0
        and (rec.sync_present or any(str(c).lower().endswith("[raw]") for c in (rec.channel_names or [])))
    ]
    used: set[str] = set()
    matched_tsv: List[Dict[str, Any]] = []
    for match in matches:
        if not (match.get("otb4") and match.get("c3d")):
            continue
        otb_dt = _parse_record_dt(match["otb4"].get("timestamp"))
        if otb_dt is None:
            continue
        otb_end = otb_dt.timestamp() + float(filename_clock_offset_sec or 0.0)
        otb_duration = float(match["otb4"].get("sample_count", 0) or 0.0) / float(match["otb4"].get("sample_rate") or 1.0)
        otb_start = otb_end - otb_duration
        c3d_duration = float(match["c3d"].get("sample_count", 0) or 0.0) / float(match["c3d"].get("sample_rate") or 1.0)
        otb_edges = list(match["otb4"].get("edge_times_sec") or [])
        c3d_edges = list(match["c3d"].get("edge_times_sec") or [])
        if otb_edges and c3d_edges:
            pair_span = float(max(otb_edges[-1], c3d_edges[-1]) - min(otb_edges[0], c3d_edges[0]))
        else:
            pair_span = float(max(otb_duration, c3d_duration))
        template_ref = []
        n_ref = min(len(otb_edges), len(c3d_edges))
        if n_ref >= 2:
            otb_rel = np.asarray(otb_edges[:n_ref], dtype=float)
            c3d_rel = np.asarray(c3d_edges[:n_ref], dtype=float)
            template_ref = (((otb_rel - otb_rel[0]) + (c3d_rel - c3d_rel[0])) / 2.0).tolist()

        best: Optional[Tuple[float, SyncRecord, float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = None
        best_diag: Optional[Tuple[SyncRecord, float, Optional[float], Optional[float], Optional[float], str]] = None
        for rec in candidates:
            if rec.path in used:
                continue
            last_ts = rec.last_ts
            first_ts = rec.first_ts
            name_dt = _parse_record_dt(rec.timestamp)
            if last_ts is None:
                continue
            name_ts = (name_dt.timestamp() + float(filename_clock_offset_sec or 0.0)) if name_dt is not None else None
            if name_ts is not None and abs(name_ts - otb_start) > 180.0 and abs(last_ts - otb_end) > 180.0:
                continue
            end_anchor = name_ts if name_ts is not None else otb_end
            start_anchor = end_anchor - (rec.duration_sec or 0.0)
            start_delta = abs((first_ts if first_ts is not None else start_anchor) - otb_start)
            end_delta = abs(last_ts - otb_end)
            filename_delta = abs(end_anchor - otb_start) if name_dt is not None else 0.0
            duration_penalty = 0.1 * abs((rec.duration_sec or 0.0) - pair_span)
            rec_json = rec.to_json()
            raw_result = _best_tsv_raw_alignment(rec_json, match["c3d"])
            raw_corr = raw_result.get("corr")
            diag_corr = abs(float(raw_corr)) if isinstance(raw_corr, (int, float)) else None
            diag_lag = abs(float(raw_result.get("lag_sec") or 0.0)) if raw_result.get("lag_sec") is not None else None
            diag_score = start_delta + end_delta + duration_penalty + 0.01 * filename_delta
            if (
                best_diag is None
                or (diag_corr is not None and best_diag[2] is not None and diag_corr > best_diag[2])
                or (diag_corr is not None and best_diag[2] is None)
            ):
                best_diag = (rec, diag_score, diag_corr, diag_lag, filename_delta, raw_result.get("quality") or "missing")
            if raw_corr is None or abs(float(raw_corr)) < 0.95:
                continue

            if rec.sync_present and rec.sync_channel and template_ref:
                try:
                    _cols, df = _read_tsv_table(Path(rec.path))
                    x = pd.to_numeric(df[rec.sync_channel], errors="coerce").to_numpy(dtype=float)
                    t_rel = pd.to_numeric(df["t_rel"], errors="coerce").to_numpy(dtype=float)
                    refined_peaks = _detect_tsv_spikes_with_template(x, t_rel, np.diff(template_ref))
                    if len(refined_peaks) >= 2:
                        rec_json["edge_times_sec"] = [float(v) for v in refined_peaks]
                        rec_json["edge_count"] = len(refined_peaks)
                        rec_json["intervals_sec"] = [float(v) for v in np.diff(refined_peaks).tolist()]
                except Exception:
                    pass

            temp_match = {
                **match,
                "tsv": rec_json,
                "alignment": {
                    **(match.get("alignment") or {}),
                    "raw_alignment_lag_sec": raw_result.get("lag_sec"),
                    "raw_alignment_corr": raw_result.get("corr"),
                    "raw_alignment_quality": raw_result.get("quality"),
                },
            }
            edge_align = _select_otb4_c3d_edge_alignment(temp_match)
            temp_match["alignment"]["sync_edge_skips"] = {
                "otb4": int(edge_align.get("otb4_skip") or 0),
                "c3d": int(edge_align.get("c3d_skip") or 0),
                "tsv": 0,
            }
            temp_match["alignment"]["otb4_c3d_edge_alignment"] = edge_align
            temp_match["alignment"]["raw_alignment_quality"] = _effective_raw_quality(raw_result, edge_align)
            dedicated = _dedicated_sync_agreement(temp_match)
            basis = "dedicated_sync+raw" if rec.sync_present and rec.sync_channel else "raw_only"
            raw_penalty = (1.0 - abs(float(raw_corr))) * 40.0 + 0.15 * abs(float(raw_result.get("lag_sec") or 0.0))
            sync_penalty = 0.5
            if rec.sync_present and rec.sync_channel:
                mean_abs = dedicated.get("mean_abs_delta_ms")
                if mean_abs is None:
                    sync_penalty = 1.5
                else:
                    sync_penalty = min(float(mean_abs) / 200.0, 3.0)
            score = start_delta + end_delta + duration_penalty + 0.01 * filename_delta + raw_penalty + sync_penalty
            if best is None or score < best[0]:
                best = (score, rec, filename_delta, rec_json, raw_result, dedicated)
        if best is None:
            reason = "No TSV candidate passed the time-window and raw-match filters."
            if best_diag is not None:
                rec, _diag_score, diag_corr, diag_lag, filename_delta, diag_quality = best_diag
                reason = (
                    f"No TSV candidate passed the raw-match threshold. "
                    f"Best nearby TSV was {Path(rec.path).name} "
                    f"(raw={diag_quality}, abs_corr={diag_corr if diag_corr is not None else 'n/a'}, "
                    f"abs_lag_sec={diag_lag if diag_lag is not None else 'n/a'}, "
                    f"filename_delta_sec={filename_delta if filename_delta is not None else 'n/a'})."
                )
            match["alignment"]["tsv_match_status"] = "missing"
            match["alignment"]["tsv_skip_reason"] = reason
            log_lines.append(f"[tsv-match] match {match['match_id']:03d}: {reason}")
            continue
        score, rec, filename_delta, rec_json, raw_result, dedicated = best
        if score > 8.0:
            reason = (
                f"Best TSV {Path(rec.path).name} rejected because combined score={score:.3f} exceeded 8.000 "
                f"(raw={raw_result.get('quality')}, corr={raw_result.get('corr')}, lag={raw_result.get('lag_sec')}, "
                f"dedicated_sync={dedicated.get('quality')})."
            )
            match["alignment"]["tsv_match_status"] = "rejected"
            match["alignment"]["tsv_skip_reason"] = reason
            log_lines.append(
                f"[tsv-match] match {match['match_id']:03d}: {reason}"
            )
            continue
        used.add(rec.path)
        match["tsv"] = rec_json
        match["alignment"]["tsv_match_status"] = "matched"
        match["alignment"]["tsv_skip_reason"] = None
        match["alignment"]["tsv_start_ts"] = rec.first_ts
        match["alignment"]["tsv_end_ts"] = rec.last_ts
        match["alignment"]["tsv_duration_sec"] = rec.duration_sec
        match["alignment"]["tsv_end_minus_otb_sec"] = round(rec.last_ts - otb_end, 6) if rec.last_ts is not None else None
        match["alignment"]["tsv_start_minus_otb_start_sec"] = round(rec.first_ts - otb_start, 6) if rec.first_ts is not None else None
        match["alignment"]["tsv_score"] = round(score, 6)
        match["alignment"]["tsv_alignment_basis"] = "dedicated_sync+raw" if rec.sync_present and rec.sync_channel else "raw_only"
        match["alignment"]["sync_edge_skips"] = temp_match["alignment"].get("sync_edge_skips") or {"otb4": 0, "c3d": 0, "tsv": 0}
        match["alignment"]["otb4_c3d_edge_alignment"] = temp_match["alignment"].get("otb4_c3d_edge_alignment") or {}
        match["alignment"]["raw_alignment_quality"] = _effective_raw_quality(raw_result, match["alignment"].get("otb4_c3d_edge_alignment"))
        match["alignment"]["raw_alignment_corr"] = raw_result.get("corr")
        match["alignment"]["raw_alignment_lag_sec"] = raw_result.get("lag_sec")
        match["alignment"]["raw_alignment_samples"] = raw_result.get("samples")
        match["alignment"]["raw_label_match_tsv"] = raw_result.get("tsv_channel")
        match["alignment"]["raw_label_match_c3d"] = raw_result.get("c3d_channel")
        match["alignment"]["raw_label_match_c3d_kind"] = raw_result.get("c3d_kind")
        match["alignment"]["raw_label_based"] = raw_result.get("label_matched")
        final_edge_align = _select_otb4_c3d_edge_alignment(match)
        match["alignment"]["sync_edge_skips"] = {
            "otb4": int(final_edge_align.get("otb4_skip") or 0),
            "c3d": int(final_edge_align.get("c3d_skip") or 0),
            "tsv": 0,
        }
        match["alignment"]["otb4_c3d_edge_alignment"] = final_edge_align
        match["alignment"]["raw_alignment_quality"] = _effective_raw_quality(raw_result, final_edge_align)
        dedicated = _dedicated_sync_agreement(match)
        match["alignment"]["dedicated_sync_quality"] = dedicated.get("quality")
        match["alignment"]["dedicated_sync_pair_quality"] = dedicated.get("pair_quality")
        match["alignment"]["dedicated_sync_mean_abs_ms"] = dedicated.get("mean_abs_delta_ms")
        match["alignment"]["dedicated_sync_max_abs_ms"] = dedicated.get("max_abs_delta_ms")
        match["alignment"]["dedicated_sync_spike_count"] = dedicated.get("spike_count")
        match["alignment"]["dedicated_sync_pairwise"] = dedicated.get("pairwise")
        match["alignment"]["plot_time_shifts_sec"] = _match_plot_shifts(match)
        match["alignment"]["inner_merge"] = _build_inner_merge_alignment(match)
        match["alignment"]["sync_triplet_quality"] = match["alignment"]["dedicated_sync_quality"]
        match["alignment"]["sync_triplet_spike_mean_abs_ms"] = match["alignment"]["dedicated_sync_mean_abs_ms"]
        match["alignment"]["sync_triplet_spike_max_abs_ms"] = match["alignment"]["dedicated_sync_max_abs_ms"]
        match["alignment"]["sync_triplet_spike_count"] = match["alignment"]["dedicated_sync_spike_count"]
        match["alignment"]["sync_triplet_pairwise"] = match["alignment"]["dedicated_sync_pairwise"]
        match["alignment"]["sync_triplet_waveform_quality"] = match["alignment"]["raw_alignment_quality"]
        match["alignment"]["sync_triplet_raw_corr_c3d_tsv"] = raw_result.get("corr")
        match["alignment"]["sync_triplet_waveform_pairwise"] = {
            "c3d_vs_tsv_raw": {"corr": raw_result.get("corr"), "lag_sec": raw_result.get("lag_sec")}
        }
        log_lines.append(
            f"[tsv-match] match {match['match_id']:03d}: {Path(rec.path).name} score={score:.3f} "
            f"start_delta={match['alignment']['tsv_start_minus_otb_start_sec']:.3f}s "
            f"end_delta={match['alignment']['tsv_end_minus_otb_sec']:.3f}s "
            f"filename_delta={filename_delta:.3f}s "
            f"basis={match['alignment']['tsv_alignment_basis']} "
            f"raw={match['alignment']['raw_alignment_quality']} corr={raw_result.get('corr')} lag={raw_result.get('lag_sec')}s "
            f"dedicated_sync={dedicated.get('quality')} mean_abs={dedicated.get('mean_abs_delta_ms')}ms max_abs={dedicated.get('max_abs_delta_ms')}ms"
        )
        if raw_result.get("tsv_channel") and raw_result.get("c3d_channel"):
            log_lines.append(
                f"[raw-match] match {match['match_id']:03d}: TSV {raw_result['tsv_channel']} <-> C3D {raw_result['c3d_channel']} "
                f"kind={raw_result.get('c3d_kind')} corr={raw_result.get('corr')} lag={raw_result.get('lag_sec')} "
                f"label_match={raw_result.get('label_match')}"
            )
        else:
            log_lines.append(f"[raw-match] match {match['match_id']:03d}: no exact TSV [raw] to C3D match found.")
        if dedicated.get("pairwise"):
            log_lines.append(f"[sync-match] match {match['match_id']:03d}: {json.dumps(dedicated['pairwise'], ensure_ascii=False)}")
        if (match["alignment"].get("otb4_c3d_edge_alignment") or {}).get("basis") == "late_c3d_raw_bridge":
            info = match["alignment"]["otb4_c3d_edge_alignment"]
            log_lines.append(
                f"[sync-bridge] match {match['match_id']:03d}: raw C3D/TSV supports late C3D start; "
                f"skip otb4={info.get('otb4_skip')} c3d={info.get('c3d_skip')} "
                f"shift={info.get('shift_sec')}s mean_abs={info.get('mean_abs_ms')}ms max_abs={info.get('max_abs_ms')}ms"
            )
        if match["alignment"]["raw_alignment_quality"] in {"good", "excellent", "fair"} and dedicated.get("pair_quality") == "poor":
            log_lines.append(
                f"[diagnostic] match {match['match_id']:03d}: raw shape agrees better than sync pulses. "
                f"This suggests a truncated or missing sync spike in one recording rather than a better cross-file reassignment."
            )
        matched_tsv.append(rec_json)
    return matched_tsv


def _build_pair_match_from_records(
    otb4_record: Dict[str, Any],
    c3d_record: Dict[str, Any],
    *,
    match_id: int,
) -> Dict[str, Any]:
    pair_cost = _pair_cost_value(
        otb4_record.get("intervals_sec") or [],
        c3d_record.get("intervals_sec") or [],
        weak_pair_cost=0.25,
    )
    if pair_cost is None:
        pair_cost = 1.0
    certainty = _summarize_pair_cost(float(pair_cost))
    otb_edges = list(otb4_record.get("edge_times_sec") or [])
    c3d_edges = list(c3d_record.get("edge_times_sec") or [])
    overlap_start = 0.0
    overlap_end = 0.0
    if otb_edges and c3d_edges:
        overlap_start = max(float(otb_edges[0]), float(c3d_edges[0]))
        overlap_end = min(float(otb_edges[-1]), float(c3d_edges[-1]))
    alignment = {
        "sync_channel": "shared TTL",
        "overlap_start_sec": overlap_start,
        "overlap_end_sec": overlap_end,
    }
    triplet = _triplet_spike_agreement([otb4_record, c3d_record])
    alignment["sync_triplet_quality"] = triplet["quality"]
    alignment["sync_triplet_spike_mean_abs_ms"] = triplet["mean_abs_delta_ms"]
    alignment["sync_triplet_spike_max_abs_ms"] = triplet["max_abs_delta_ms"]
    alignment["sync_triplet_spike_count"] = triplet["spike_count"]
    alignment["sync_triplet_pairwise"] = triplet["pairwise"]
    return {
        "match_id": int(match_id),
        "certainty": certainty,
        "pair_cost": float(pair_cost),
        "otb4": dict(otb4_record),
        "c3d": dict(c3d_record),
        "tsv": None,
        "alignment": alignment,
    }


def _evaluate_manual_tsv_selection(
    match: Dict[str, Any],
    tsv_record: Dict[str, Any],
    *,
    filename_clock_offset_sec: float,
) -> Optional[Dict[str, Any]]:
    otb_dt = _parse_record_dt((match.get("otb4") or {}).get("timestamp"))
    if otb_dt is None:
        return None
    otb_end = otb_dt.timestamp() + float(filename_clock_offset_sec or 0.0)
    otb_duration = float((match.get("otb4") or {}).get("sample_count", 0) or 0.0) / float((match.get("otb4") or {}).get("sample_rate") or 1.0)
    otb_start = otb_end - otb_duration
    c3d_duration = float((match.get("c3d") or {}).get("sample_count", 0) or 0.0) / float((match.get("c3d") or {}).get("sample_rate") or 1.0)
    otb_edges = list((match.get("otb4") or {}).get("edge_times_sec") or [])
    c3d_edges = list((match.get("c3d") or {}).get("edge_times_sec") or [])
    if otb_edges and c3d_edges:
        pair_span = float(max(otb_edges[-1], c3d_edges[-1]) - min(otb_edges[0], c3d_edges[0]))
    else:
        pair_span = float(max(otb_duration, c3d_duration))
    template_ref: List[float] = []
    n_ref = min(len(otb_edges), len(c3d_edges))
    if n_ref >= 2:
        otb_rel = np.asarray(otb_edges[:n_ref], dtype=float)
        c3d_rel = np.asarray(c3d_edges[:n_ref], dtype=float)
        template_ref = (((otb_rel - otb_rel[0]) + (c3d_rel - c3d_rel[0])) / 2.0).tolist()

    rec_json = dict(tsv_record)
    last_ts = rec_json.get("last_ts")
    first_ts = rec_json.get("first_ts")
    duration_sec = rec_json.get("duration_sec")
    if last_ts is None or duration_sec is None:
        return None
    name_dt = _parse_record_dt(rec_json.get("timestamp"))
    name_ts = (name_dt.timestamp() + float(filename_clock_offset_sec or 0.0)) if name_dt is not None else None
    end_anchor = name_ts if name_ts is not None else otb_end
    start_anchor = end_anchor - float(duration_sec or 0.0)
    start_delta = abs((float(first_ts) if first_ts is not None else start_anchor) - otb_start)
    end_delta = abs(float(last_ts) - otb_end)
    filename_delta = abs(end_anchor - otb_start) if name_dt is not None else 0.0
    duration_penalty = 0.1 * abs(float(duration_sec or 0.0) - pair_span)

    raw_result = _best_tsv_raw_alignment(rec_json, match["c3d"])
    raw_corr = raw_result.get("corr")

    if rec_json.get("sync_present") and rec_json.get("sync_channel") and template_ref:
        try:
            _cols, df = _read_tsv_table(Path(rec_json["path"]))
            x = pd.to_numeric(df[rec_json["sync_channel"]], errors="coerce").to_numpy(dtype=float)
            t_rel = pd.to_numeric(df["t_rel"], errors="coerce").to_numpy(dtype=float)
            refined_peaks = _detect_tsv_spikes_with_template(x, t_rel, np.diff(template_ref))
            if len(refined_peaks) >= 2:
                rec_json["edge_times_sec"] = [float(v) for v in refined_peaks]
                rec_json["edge_count"] = len(refined_peaks)
                rec_json["intervals_sec"] = [float(v) for v in np.diff(refined_peaks).tolist()]
        except Exception:
            pass

    temp_match = {
        **match,
        "tsv": rec_json,
        "alignment": {
            **(match.get("alignment") or {}),
            "raw_alignment_lag_sec": raw_result.get("lag_sec"),
            "raw_alignment_corr": raw_result.get("corr"),
            "raw_alignment_quality": raw_result.get("quality"),
        },
    }
    edge_align = _select_otb4_c3d_edge_alignment(temp_match)
    temp_match["alignment"]["sync_edge_skips"] = {
        "otb4": int(edge_align.get("otb4_skip") or 0),
        "c3d": int(edge_align.get("c3d_skip") or 0),
        "tsv": 0,
    }
    temp_match["alignment"]["otb4_c3d_edge_alignment"] = edge_align
    temp_match["alignment"]["raw_alignment_quality"] = _effective_raw_quality(raw_result, edge_align)
    dedicated = _dedicated_sync_agreement(temp_match)
    corr_mag = abs(float(raw_corr)) if isinstance(raw_corr, (int, float)) else 0.0
    raw_penalty = (1.0 - corr_mag) * 40.0 + 0.15 * abs(float(raw_result.get("lag_sec") or 0.0))
    sync_penalty = 0.5
    if rec_json.get("sync_present") and rec_json.get("sync_channel"):
        mean_abs = dedicated.get("mean_abs_delta_ms")
        if mean_abs is None:
            sync_penalty = 1.5
        else:
            sync_penalty = min(float(mean_abs) / 200.0, 3.0)
    score = start_delta + end_delta + duration_penalty + 0.01 * filename_delta + raw_penalty + sync_penalty
    return {
        "rec_json": rec_json,
        "raw_result": raw_result,
        "dedicated": dedicated,
        "score": float(score),
        "filename_delta": float(filename_delta),
        "otb_end": float(otb_end),
        "otb_start": float(otb_start),
        "start_delta": float(start_delta),
        "end_delta": float(end_delta),
    }


def _apply_manual_tsv_selection(
    match: Dict[str, Any],
    tsv_record: Dict[str, Any],
    *,
    filename_clock_offset_sec: float,
) -> bool:
    evaluated = _evaluate_manual_tsv_selection(match, tsv_record, filename_clock_offset_sec=filename_clock_offset_sec)
    if evaluated is None:
        return False
    rec_json = evaluated["rec_json"]
    raw_result = evaluated["raw_result"]
    dedicated = evaluated["dedicated"]
    score = evaluated["score"]
    filename_delta = evaluated["filename_delta"]
    otb_end = evaluated["otb_end"]
    otb_start = evaluated["otb_start"]
    rec_last_ts = rec_json.get("last_ts")
    rec_first_ts = rec_json.get("first_ts")
    match["tsv"] = rec_json
    alignment = match.setdefault("alignment", {})
    alignment["manual_selection"] = True
    alignment["tsv_match_status"] = "manual"
    alignment["tsv_skip_reason"] = "Manual TSV/OTB4 selection pending user review."
    alignment["tsv_start_ts"] = rec_first_ts
    alignment["tsv_end_ts"] = rec_last_ts
    alignment["tsv_duration_sec"] = rec_json.get("duration_sec")
    alignment["tsv_end_minus_otb_sec"] = round(float(rec_last_ts) - otb_end, 6) if rec_last_ts is not None else None
    alignment["tsv_start_minus_otb_start_sec"] = round(float(rec_first_ts) - otb_start, 6) if rec_first_ts is not None else None
    alignment["tsv_score"] = round(float(score), 6)
    alignment["tsv_alignment_basis"] = "manual_selection"
    alignment["raw_alignment_corr"] = raw_result.get("corr")
    alignment["raw_alignment_lag_sec"] = raw_result.get("lag_sec")
    alignment["raw_alignment_samples"] = raw_result.get("samples")
    alignment["raw_label_match_tsv"] = raw_result.get("tsv_channel")
    alignment["raw_label_match_c3d"] = raw_result.get("c3d_channel")
    alignment["raw_label_match_c3d_kind"] = raw_result.get("c3d_kind")
    alignment["raw_label_based"] = raw_result.get("label_matched")
    final_edge_align = _select_otb4_c3d_edge_alignment(match)
    alignment["sync_edge_skips"] = {
        "otb4": int(final_edge_align.get("otb4_skip") or 0),
        "c3d": int(final_edge_align.get("c3d_skip") or 0),
        "tsv": 0,
    }
    alignment["otb4_c3d_edge_alignment"] = final_edge_align
    alignment["raw_alignment_quality"] = _effective_raw_quality(raw_result, final_edge_align)
    dedicated = _dedicated_sync_agreement(match)
    alignment["dedicated_sync_quality"] = dedicated.get("quality")
    alignment["dedicated_sync_pair_quality"] = dedicated.get("pair_quality")
    alignment["dedicated_sync_mean_abs_ms"] = dedicated.get("mean_abs_delta_ms")
    alignment["dedicated_sync_max_abs_ms"] = dedicated.get("max_abs_delta_ms")
    alignment["dedicated_sync_spike_count"] = dedicated.get("spike_count")
    alignment["dedicated_sync_pairwise"] = dedicated.get("pairwise")
    alignment["plot_time_shifts_sec"] = _match_plot_shifts(match)
    alignment["inner_merge"] = _build_inner_merge_alignment(match)
    alignment["sync_triplet_quality"] = alignment["dedicated_sync_quality"]
    alignment["sync_triplet_spike_mean_abs_ms"] = alignment["dedicated_sync_mean_abs_ms"]
    alignment["sync_triplet_spike_max_abs_ms"] = alignment["dedicated_sync_max_abs_ms"]
    alignment["sync_triplet_spike_count"] = alignment["dedicated_sync_spike_count"]
    alignment["sync_triplet_pairwise"] = alignment["dedicated_sync_pairwise"]
    alignment["sync_triplet_waveform_quality"] = alignment["raw_alignment_quality"]
    alignment["sync_triplet_raw_corr_c3d_tsv"] = raw_result.get("corr")
    alignment["sync_triplet_waveform_pairwise"] = {
        "c3d_vs_tsv_raw": {"corr": raw_result.get("corr"), "lag_sec": raw_result.get("lag_sec")}
    }
    return True


def analyze_folder(folder: Path) -> Dict[str, Any]:
    folder = folder.resolve()
    log_lines: List[str] = []
    log_lines.append(f"[{_now_iso()}] Scan start: {folder}")
    filename_clock_offset_sec = _infer_root_filename_clock_offset_sec(folder) or 0.0
    log_lines.append(
        f"[timing] filename clock offset={filename_clock_offset_sec:.3f}s ({filename_clock_offset_sec / 3600.0:.6f}h) "
        f"derived from file modification times."
    )

    otb4_records: List[SyncRecord] = []
    c3d_records: List[SyncRecord] = []
    tsv_records: List[SyncRecord] = []
    unmatched: List[Dict[str, Any]] = []

    for path in _iter_source_files(folder, ".otb4", excluded_dir_names=("matched", "accepted", "__pycache__", ".git")):
        rec = _extract_otb4_record(path)
        otb4_records.append(rec)
        log_lines.append(
            f"[otb4] {path.name}: sync={rec.sync_present} quality={rec.sync_quality} "
            f"channel={rec.sync_channel} edges={rec.edge_count}"
        )
        for note in rec.notes:
            log_lines.append(f"  - {note}")

    for path in _iter_source_files(folder, ".c3d", excluded_dir_names=("matched", "accepted", "__pycache__", ".git")):
        rec = _extract_c3d_record(path)
        c3d_records.append(rec)
        log_lines.append(
            f"[c3d] {path.name}: sync={rec.sync_present} quality={rec.sync_quality} "
            f"channel={rec.sync_channel} edges={rec.edge_count}"
        )
        for note in rec.notes:
            log_lines.append(f"  - {note}")

    for path in _iter_source_files(folder, ".tsv", excluded_dir_names=("matched", "accepted", "__pycache__", ".git")):
        rec = _extract_tsv_record(path)
        tsv_records.append(rec)
        log_lines.append(
            f"[tsv] {path.name}: sync={rec.sync_present} quality={rec.sync_quality} "
            f"channel={rec.sync_channel} edges={rec.edge_count}"
        )
        for note in rec.notes:
            log_lines.append(f"  - {note}")

    otb4_to_c3d = _pair_records(otb4_records, c3d_records)
    otb4_to_c3d.sort(key=lambda t: t[0].path)

    pair_matches: List[Dict[str, Any]] = []
    pair_id = 1
    for otb, c3d, cost in otb4_to_c3d:
        certainty = _summarize_pair_cost(cost)
        alignment = {
            "sync_channel": "shared TTL",
            "overlap_start_sec": max(
                otb.edge_times_sec[0] if otb.edge_times_sec else 0.0,
                c3d.edge_times_sec[0] if c3d.edge_times_sec else 0.0,
            ),
            "overlap_end_sec": min(
                otb.edge_times_sec[-1] if otb.edge_times_sec else 0.0,
                c3d.edge_times_sec[-1] if c3d.edge_times_sec else 0.0,
            ),
        }
        triplet = _triplet_spike_agreement([otb.to_json(), c3d.to_json()])
        alignment["sync_triplet_quality"] = triplet["quality"]
        alignment["sync_triplet_spike_mean_abs_ms"] = triplet["mean_abs_delta_ms"]
        alignment["sync_triplet_spike_max_abs_ms"] = triplet["max_abs_delta_ms"]
        alignment["sync_triplet_spike_count"] = triplet["spike_count"]
        alignment["sync_triplet_pairwise"] = triplet["pairwise"]
        log_lines.append(
            f"[pair] {pair_id:03d} otb4={Path(otb.path).name} c3d={Path(c3d.path).name} "
            f"cost={cost:.6f} certainty={certainty} spikes={triplet['quality']} "
            f"mean_abs={triplet['mean_abs_delta_ms']}ms max_abs={triplet['max_abs_delta_ms']}ms"
        )
        pair_matches.append(
            {
                "match_id": pair_id,
                "certainty": certainty,
                "pair_cost": cost,
                "otb4": otb.to_json(),
                "c3d": c3d.to_json(),
                "tsv": None,
                "alignment": alignment,
            }
        )
        pair_id += 1

    _apply_otb4_repairs(pair_matches, log_lines)
    matched_tsv = _match_tsv_records(
        pair_matches,
        tsv_records,
        log_lines,
        filename_clock_offset_sec=filename_clock_offset_sec,
    )
    pair_matches.sort(key=_c3d_review_sort_key)
    for idx, match in enumerate(pair_matches, start=1):
        match["match_id"] = idx
        _apply_review_defaults(match)
    certain_triplets = [match for match in pair_matches if match.get("tsv")]
    pair_only_matches = [match for match in pair_matches if not match.get("tsv")]
    used_otb_paths = {str((match.get("otb4") or {}).get("path") or "") for match in pair_matches}
    used_c3d_paths = {str((match.get("c3d") or {}).get("path") or "") for match in pair_matches}
    startup_messages: List[str] = []
    if pair_only_matches:
        startup_messages.append(
            f"{len(pair_only_matches)} certain OTB4/C3D pair(s) are included in review without a TSV match."
        )
        for match in pair_only_matches:
            reason = (match.get("alignment") or {}).get("tsv_skip_reason") or "No TSV reason recorded."
            startup_messages.append(f"{Path(match['c3d']['path']).name}: {reason}")
    for rec in tsv_records:
        if rec.path not in {item["path"] for item in matched_tsv}:
            if rec.sync_quality == "certain" and rec.edge_count >= 4:
                log_lines.append(
                    f"[tsv-candidate] {Path(rec.path).name}: sync present but not selected as a certain match."
                )
            else:
                log_lines.append(
                    f"[tsv-missing] {Path(rec.path).name}: sync channel missing or too weak for a certain match."
                )

    summary = {
        "otb4_count": len(otb4_records),
        "c3d_count": len(c3d_records),
        "tsv_count": len(tsv_records),
        "pair_count": len(pair_matches),
        "certain_pair_count": sum(1 for match in pair_matches if match.get("certainty") == "certain"),
        "certain_tsv_count": len(matched_tsv),
        "certain_triplet_count": len(certain_triplets),
        "auto_accept_count": sum(1 for match in certain_triplets if (match.get("review") or {}).get("auto_accept")),
        "filename_clock_offset_sec": round(float(filename_clock_offset_sec), 6),
    }
    log_lines.append(f"[summary] {json.dumps(summary, ensure_ascii=False)}")
    log_lines.append(f"[{_now_iso()}] Scan complete.")
    mapping = {
        "source_root": str(folder),
        "created_at": _now_iso(),
        "summary": summary,
        "matches": pair_matches,
        "triplet_matches": certain_triplets,
        "pair_only_matches": pair_only_matches,
        "tsv_candidates": matched_tsv,
        "unmatched": unmatched,
        "startup_messages": startup_messages,
        "filename_clock_offset_sec": round(float(filename_clock_offset_sec), 6),
        "source_records": {
            "otb4": [rec.to_json() for rec in otb4_records],
            "c3d": [rec.to_json() for rec in c3d_records],
            "tsv": [rec.to_json() for rec in tsv_records],
        },
        "unmatched_c3d_records": [rec.to_json() for rec in c3d_records if str(rec.path) not in used_c3d_paths],
        "unmatched_otb4_records": [rec.to_json() for rec in otb4_records if str(rec.path) not in used_otb_paths],
    }
    return {"mapping": mapping, "log_lines": log_lines}


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_repair_jsons(export_dir: Path, mapping: Dict[str, Any]) -> None:
    repair_dir = export_dir / "matched"
    repair_dir.mkdir(exist_ok=True, parents=True)
    for existing in repair_dir.glob("*_repair.json"):
        try:
            existing.unlink()
        except Exception:
            pass
    for match in mapping.get("matches", []):
        alignment = match.get("alignment") or {}
        repair = alignment.get("otb4_repair")
        device_repairs = alignment.get("otb4_device_repairs") or {}
        if not isinstance(repair, dict) and not isinstance(device_repairs, dict):
            continue
        path = (repair.get("json_path") if isinstance(repair, dict) else None) or ((match.get("export_plan") or {}).get("repair_json") or {}).get("path")
        if not path:
            continue
        out_path = Path(path)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        payload = {
            **(repair if isinstance(repair, dict) else {}),
            "device_repairs": device_repairs if isinstance(device_repairs, dict) else {},
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _clean_output_dir(path: Path) -> None:
    path.mkdir(exist_ok=True, parents=True)
    for existing in list(path.iterdir()):
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink()


def _clean_output_dir_except(path: Path, keep_suffixes: Sequence[str]) -> None:
    path.mkdir(exist_ok=True, parents=True)
    keep = {suffix.lower() for suffix in keep_suffixes}
    for existing in list(path.iterdir()):
        if existing.is_dir():
            shutil.rmtree(existing)
        elif existing.suffix.lower() not in keep:
            existing.unlink()


def _planned_copy_name(match: Dict[str, Any], key: str, saved_stamp: str) -> Optional[str]:
    item = match.get(key)
    if not item:
        return None
    src = Path(item["path"])
    return f"{_match_base_name(match, saved_stamp)}{src.suffix.lower()}"


def _tsv_sidecar_json_path(match: Dict[str, Any]) -> Optional[Path]:
    tsv = match.get("tsv") or {}
    tsv_path = tsv.get("path")
    if not tsv_path:
        return None
    sidecar = Path(str(tsv_path)).with_suffix(".json")
    return sidecar if sidecar.exists() else None


def _repair_json_name(match: Dict[str, Any], saved_stamp: str) -> str:
    return f"{_match_base_name(match, saved_stamp)}_repair.json"


def _mat_export_name(match: Dict[str, Any], saved_stamp: str) -> str:
    return f"{_match_base_name(match, saved_stamp)}.mat"


def _match_base_name(match: Dict[str, Any], saved_stamp: str) -> str:
    stamp = saved_stamp
    tsv = match.get("tsv") or {}
    tsv_path = tsv.get("path")
    if tsv_path:
        tsv_dt = _parse_timestamp_from_name(Path(str(tsv_path)))
        if tsv_dt is not None:
            stamp = tsv_dt.strftime("%Y%m%d_%H%M%S")
    c3d = match.get("c3d") or {}
    c3d_path = c3d.get("path")
    if c3d_path:
        stem = Path(str(c3d_path)).stem
        if stem:
            return f"{stem}_{stamp}"
    return f"m{int(match['match_id']):03d}_{stamp}"


def _match_export_plan(match: Dict[str, Any], export_dir: Path, saved_stamp: str) -> Dict[str, Any]:
    mat_name = _mat_export_name(match, saved_stamp)
    mat_path = export_dir / "matched" / mat_name
    pipe_mat_path = _pipe_mat_path(mat_path)
    pipe_json_path = _pipe_json_path(pipe_mat_path)
    repair_json_name = _repair_json_name(match, saved_stamp)
    repair_dir_path = export_dir / "matched" / repair_json_name
    plan: Dict[str, Any] = {
        "matched": {},
        "mat": {
            "filename": mat_name,
            "path": str(mat_path),
        },
        "pipe_mat": {
            "filename": pipe_mat_path.name,
            "path": str(pipe_mat_path),
        },
        "pipe_json": {
            "filename": pipe_json_path.name,
            "path": str(pipe_json_path),
        },
        "repair_json": {
            "filename": repair_json_name,
            "path": str(repair_dir_path),
        },
    }
    for key in ["otb4", "c3d", "tsv"]:
        item = match.get(key)
        if not item:
            continue
        src = Path(item["path"])
        matched_name = _planned_copy_name(match, key, saved_stamp)
        plan["matched"][key] = {
            "original_path": str(src),
            "original_filename": src.name,
            "renamed_filename": matched_name,
            "renamed_path": str(export_dir / "matched" / matched_name) if matched_name else None,
        }
    sidecar_json = _tsv_sidecar_json_path(match)
    if sidecar_json is not None:
        matched_json_name = f"{_match_base_name(match, saved_stamp)}_traceBioRTFB.json"
        plan["matched"]["tsv_json"] = {
            "original_path": str(sidecar_json),
            "original_filename": sidecar_json.name,
            "renamed_filename": matched_json_name,
            "renamed_path": str(export_dir / "matched" / matched_json_name),
        }
    if (match.get("alignment") or {}).get("otb4_repair"):
        plan["matched"]["repair_json"] = {
            "original_path": str(repair_dir_path),
            "original_filename": repair_json_name,
            "renamed_filename": repair_json_name,
            "renamed_path": str(export_dir / "matched" / repair_json_name),
        }
    return plan


def _attach_export_plan(match: Dict[str, Any], export_dir: Path, saved_stamp: str) -> None:
    match["saved_at"] = saved_stamp
    match["export_plan"] = _match_export_plan(match, export_dir, saved_stamp)
    repair = (match.get("alignment") or {}).get("otb4_repair")
    if isinstance(repair, dict):
        repair["json_path"] = ((match.get("export_plan") or {}).get("repair_json") or {}).get("path")
    for key in ["otb4", "c3d", "tsv"]:
        item = match.get(key)
        if not item:
            continue
        src = Path(item["path"])
        item["original_filename"] = src.name
        item["planned_matched_filename"] = (
            ((match.get("export_plan") or {}).get("matched") or {}).get(key, {}).get("renamed_filename")
        )


def _has_user_input(mapping: Dict[str, Any]) -> bool:
    for match in mapping.get("matches", []):
        review = match.get("review") or {}
        if review.get("user_decision") is not None or review.get("user_touched"):
            return True
    return False


def _review_complete(mapping: Dict[str, Any]) -> bool:
    reviewed, total = _review_progress(mapping)
    return total > 0 and reviewed == total


def _copy_match_group(match: Dict[str, Any], target_dir: Path, saved_stamp: str) -> Dict[str, Optional[str]]:
    copied = {"otb4": None, "c3d": None, "tsv": None, "tsv_json": None, "repair_json": None}
    plan_group = ((match.get("export_plan") or {}).get("matched") or {})
    for key in ["otb4", "c3d", "tsv", "tsv_json", "repair_json"]:
        if key in {"otb4", "c3d", "tsv"}:
            item = match.get(key)
            if not item:
                continue
            src = Path(item["path"])
        else:
            entry = plan_group.get(key) or {}
            src_path = entry.get("original_path")
            if not src_path or not Path(src_path).exists():
                continue
            src = Path(src_path)
        entry = plan_group.get(key) or {}
        filename = entry.get("renamed_filename")
        if not filename:
            continue
        dst = target_dir / filename
        try:
            if src.resolve() != dst.resolve():
                _safe_copy(src, dst)
        except Exception:
            _safe_copy(src, dst)
        copied[key] = str(dst)
    return copied


def _write_review_outputs(export_dir: Path, mapping: Dict[str, Any], saved_stamp: str) -> Tuple[Path, Path]:
    export_dir = export_dir.resolve()
    matched_dir = export_dir / "matched"
    legacy_accepted_dir = export_dir / "accepted"
    if legacy_accepted_dir.exists():
        shutil.rmtree(legacy_accepted_dir, ignore_errors=True)
    if not _review_complete(mapping):
        matched_dir.mkdir(exist_ok=True, parents=True)
    else:
        matched_dir.mkdir(exist_ok=True, parents=True)
        for match in mapping.get("matches", []):
            if not (match.get("review") or {}).get("final_accept") or not match.get("tsv"):
                continue
            copied = _copy_match_group(match, matched_dir, saved_stamp)
            match.setdefault("review_outputs", {})["matched_copied"] = copied
            repair = (match.get("alignment") or {}).get("otb4_repair")
            if isinstance(repair, dict) and copied.get("repair_json"):
                matched_repair_path = str(copied["repair_json"])
                repair["json_path"] = matched_repair_path
                export_plan = match.setdefault("export_plan", {})
                if isinstance(export_plan.get("repair_json"), dict):
                    export_plan["repair_json"]["path"] = matched_repair_path
                matched_group = export_plan.setdefault("matched", {})
                if isinstance(matched_group.get("repair_json"), dict):
                    matched_group["repair_json"]["renamed_path"] = matched_repair_path

    accepted_matches = [
        match for match in mapping.get("matches", [])
        if (match.get("review") or {}).get("final_accept") and match.get("tsv")
    ]
    accepted_mapping = {
        "source_root": mapping["source_root"],
        "created_at": mapping["created_at"],
        "saved_at": saved_stamp,
        "summary": {
            **mapping.get("summary", {}),
            "accepted_count": len(accepted_matches),
            "reviewed_count": sum(1 for match in mapping.get("matches", []) if (match.get("review") or {}).get("reviewed")),
        },
        "matches": accepted_matches,
    }
    accepted_json_path = export_dir / "accepted_matches.json"
    accepted_json_path.write_text(json.dumps(accepted_mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return accepted_json_path, matched_dir


def export_results(source_folder: Path, export_dir: Path, mapping: Dict[str, Any], log_lines: Sequence[str]) -> Tuple[Path, Path, Path]:
    export_dir = export_dir.resolve()
    matched_dir = export_dir / "matched"
    legacy_accepted_dir = export_dir / "accepted"
    if matched_dir.exists():
        shutil.rmtree(matched_dir, ignore_errors=True)
    if legacy_accepted_dir.exists():
        shutil.rmtree(legacy_accepted_dir, ignore_errors=True)
    saved_stamp = _format_file_stamp()

    mapping_path = export_dir / "matched_files.json"
    log_path = export_dir / "alignment.log"

    safe_mapping = {
        "source_root": mapping["source_root"],
        "created_at": mapping["created_at"],
        "summary": mapping["summary"],
        "matches": [],
        "triplet_matches": [],
        "pair_only_matches": [],
        "tsv_candidates": mapping.get("tsv_candidates", []),
        "unmatched": mapping.get("unmatched", []),
        "startup_messages": list(mapping.get("startup_messages", [])),
        "source_records": mapping.get("source_records", {}),
        "unmatched_c3d_records": mapping.get("unmatched_c3d_records", []),
        "unmatched_otb4_records": mapping.get("unmatched_otb4_records", []),
    }

    for match in mapping["matches"]:
        out_match = dict(match)
        _apply_review_defaults(out_match)
        _attach_export_plan(out_match, export_dir, saved_stamp)
        safe_mapping["matches"].append(out_match)
        if out_match.get("tsv"):
            safe_mapping["triplet_matches"].append(out_match)
        else:
            safe_mapping["pair_only_matches"].append(out_match)

    _ensure_unique_source_assignments(safe_mapping["matches"])
    _refresh_mapping_summary(safe_mapping)
    safe_mapping["saved_at"] = saved_stamp
    safe_mapping["export_root"] = str(export_dir)
    _write_repair_jsons(export_dir, safe_mapping)
    mapping_path.write_text(json.dumps(safe_mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_text(log_path, log_lines)
    _write_review_outputs(export_dir, safe_mapping, saved_stamp)
    return mapping_path, log_path, matched_dir


def run_scan(folder: Path, export_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    result = analyze_folder(folder)
    return export_results(folder, export_dir or folder, result["mapping"], result["log_lines"])


# ---------------- Viewer ----------------
def _load_mapping(mapping_path: Path) -> Dict[str, Any]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    mapping["export_root"] = str(mapping_path.parent.resolve())
    return mapping


def _load_otb4_channel(path: Path, channel_name: str, repair_zones: Optional[Sequence[Tuple[int, int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if channel_name == SYNC_OTB_LABEL:
        t, x, _meta = _load_otb4_aux_channel(path, device=SYNC_OTB_DEVICE, subtitle=SYNC_OTB_SUBTITLE)
        if repair_zones:
            x = _insert_nan_holes_signal(x, repair_zones)
            t = np.arange(len(x), dtype=float) / float(_meta["sampling_frequency"])
        return t, x
    data, _time, descs, fs, _name, _size = load_otb4_file(str(path))
    flat = []
    for d in descs:
        try:
            flat.append(str(d[0][0]))
        except Exception:
            flat.append(str(d))
    idx = flat.index(channel_name)
    x = np.asarray(data[idx], dtype=float).reshape(-1)
    if repair_zones:
        x = _insert_nan_holes_signal(x, repair_zones)
    t = np.arange(len(x), dtype=float) / float(fs)
    return t, x


def _c3d_metadata(path: Path) -> Dict[str, Any]:
    c3d = ezc3d.c3d(str(path))
    analog_labels = list(c3d["parameters"]["ANALOG"]["LABELS"]["value"])
    point_labels = list(c3d["parameters"].get("POINT", {}).get("LABELS", {}).get("value", []))
    if "LABELS2" in c3d["parameters"].get("POINT", {}):
        point_labels += list(c3d["parameters"]["POINT"]["LABELS2"]["value"])
    analog_rate = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])
    point_rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0]) if "POINT" in c3d["parameters"] else None
    return {
        "c3d": c3d,
        "analog_labels": analog_labels,
        "point_labels": point_labels,
        "analog_rate": analog_rate,
        "point_rate": point_rate,
    }


def _load_c3d_analog_channel(path: Path, channel_name: str) -> Tuple[np.ndarray, np.ndarray]:
    meta = _c3d_metadata(path)
    labels = meta["analog_labels"]
    idx = labels.index(channel_name)
    arr = np.asarray(meta["c3d"]["data"]["analogs"], dtype=float)
    subframes, n_channels, n_frames = arr.shape
    x = np.transpose(arr, (2, 0, 1)).reshape(n_frames * subframes, n_channels)[:, idx]
    fs = float(meta["analog_rate"])
    t = np.arange(len(x), dtype=float) / fs
    return t, x


def _load_c3d_point_channel(path: Path, channel_name: str) -> Tuple[np.ndarray, np.ndarray]:
    meta = _c3d_metadata(path)
    labels = meta["point_labels"]
    if "." not in channel_name:
        raise ValueError(f"Point channel must include a coordinate suffix: {channel_name}")
    base, coord = channel_name.rsplit(".", 1)
    idx = labels.index(base)
    coord_idx = {"x": 0, "y": 1, "z": 2, "residual": 3}.get(coord.lower())
    if coord_idx is None:
        raise ValueError(f"Unsupported C3D point coordinate: {coord}")
    arr = np.asarray(meta["c3d"]["data"]["points"], dtype=float)
    if arr.ndim != 3 or arr.shape[0] < 4:
        raise ValueError("Unexpected C3D point data shape.")
    x = arr[coord_idx, idx, :]
    fs = float(meta["point_rate"] or meta["analog_rate"])
    t = np.arange(len(x), dtype=float) / fs
    return t, x


def _load_tsv_channel(path: Path, channel_name: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep="\t", decimal=",", encoding="utf-8-sig")
    if channel_name in df.columns:
        x = pd.to_numeric(df[channel_name], errors="coerce").to_numpy(dtype=float)
    elif channel_name.strip().lower() == "performed":
        synth = _synth_tsv_performed_values(path, df)
        if synth is None:
            raise KeyError(f"Missing TSV channel: {channel_name}")
        x = np.asarray(synth, dtype=float)
    else:
        raise KeyError(f"Missing TSV channel: {channel_name}")
    if channel_name.strip().lower() == "performed" and not np.any(np.isfinite(x)):
        synth = _synth_tsv_performed_values(path, df)
        if synth is not None and np.any(np.isfinite(synth)):
            x = np.asarray(synth, dtype=float)
    med = np.nanmedian(x) if np.any(np.isfinite(x)) else 0.0
    x = np.nan_to_num(x, nan=med)
    if "t_rel" in df.columns:
        t = pd.to_numeric(df["t_rel"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(t)) or len(t) != len(x):
            t = np.arange(len(x), dtype=float)
    else:
        t = np.arange(len(x), dtype=float)
    return t, x


def _channel_list_for_record(record: Dict[str, Any]) -> List[str]:
    names = record.get("channel_names") or []
    if record["kind"] == "tsv":
        return [n for n in names if n.lower() not in {"ts", "t_rel"}]
    return names


def _channel_options_for_record(record: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    options: List[Tuple[str, Dict[str, Any]]] = []
    kind = record["kind"]
    if kind == "otb4":
        for ch in _channel_list_for_record(record):
            options.append((f"OTB4: {ch}", {"source": "otb4", "channel": ch, "kind": "sync" if ch == record.get("sync_channel") else "otb4"}))
    elif kind == "c3d":
        for ch in ("copx", "copy"):
            options.append((f"C3D CoP: {ch}", {"source": "c3d", "channel": ch, "kind": "cop"}))
        for ch in record.get("channel_names") or []:
            options.append((f"C3D analog: {ch}", {"source": "c3d", "channel": ch, "kind": "analog"}))
        for ch in record.get("point_channel_names") or []:
            for coord in ("x", "y", "z", "residual"):
                options.append((f"C3D point: {ch}.{coord}", {"source": "c3d", "channel": f"{ch}.{coord}", "kind": "point"}))
    elif kind == "tsv":
        for ch in _channel_list_for_record(record):
            options.append((f"TSV: {ch}", {"source": "tsv", "channel": ch, "kind": "tsv"}))
    return options


def _sync_series_specs(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for source in ("otb4", "c3d", "tsv"):
        rec = match.get(source)
        if rec and rec.get("sync_present") and rec.get("sync_channel"):
            specs.append({"source": source, "kind": "sync", "channel": rec["sync_channel"]})
    return specs


def _raw_series_specs(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    alignment = match.get("alignment") or {}
    tsv_channel = alignment.get("raw_label_match_tsv")
    c3d_channel = alignment.get("raw_label_match_c3d")
    c3d_kind = alignment.get("raw_label_match_c3d_kind") or "analog"
    specs: List[Dict[str, Any]] = []
    if c3d_channel:
        specs.append({"source": "c3d", "kind": "raw", "channel": c3d_channel, "c3d_kind": c3d_kind})
    if tsv_channel:
        specs.append({"source": "tsv", "kind": "raw", "channel": tsv_channel})
    return specs


def _load_series_for_spec(match: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    source = spec["source"]
    if source == "otb4":
        device = _otb_label_device(spec["channel"])
        t, y = _load_otb4_channel(
            Path(match[source]["path"]),
            spec["channel"],
            repair_zones=_repair_zones_for_match(match, device=device),
        )
    elif source == "c3d":
        t, y = _load_c3d_series(Path(match[source]["path"]), spec.get("c3d_kind") or spec.get("kind") or "analog", spec["channel"])
    elif source == "tsv":
        t, y = _load_tsv_channel(Path(match[source]["path"]), spec["channel"])
    else:
        raise ValueError(source)
    shift = float(_match_plot_shifts(match).get(source, 0.0))
    return t + shift, y


def _mat_field_name(label: str, used: set[str]) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(label).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "field"
    if text[0].isdigit():
        text = f"f_{text}"
    base = text[:63]
    candidate = base
    suffix = 1
    while candidate in used:
        suffix_text = f"_{suffix}"
        candidate = f"{base[: max(1, 63 - len(suffix_text))]}{suffix_text}"
        suffix += 1
    used.add(candidate)
    return candidate


def _crop_aligned_series(t: np.ndarray, y: np.ndarray, start_sec: float, end_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(t) & (t >= start_sec) & (t <= end_sec)
    if not np.any(mask):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return np.asarray(t[mask], dtype=float), np.asarray(y[mask], dtype=float)


def _relative_time(t: np.ndarray, start_sec: float) -> np.ndarray:
    return np.asarray(t, dtype=float) - float(start_sec)


def _build_otb4_mat_struct(match: Dict[str, Any], start_sec: float, end_sec: float) -> Dict[str, Any]:
    record = match["otb4"]
    path = Path(record["path"])
    data, _time, descs, fs, _name, _size = load_otb4_file(str(path))
    repair_zones = _repair_zones_for_match(match)
    if repair_zones:
        data = _insert_nan_holes_matrix(np.asarray(data, dtype=float), repair_zones)
    labels: List[str] = []
    for d in descs:
        try:
            labels.append(str(d[0][0]))
        except Exception:
            labels.append(str(d))
    used: set[str] = set()
    label_map: Dict[str, str] = {}
    out: Dict[str, Any] = {}
    shift = float(_match_plot_shifts(match).get("otb4", 0.0))
    base_t = np.arange(np.asarray(data).shape[1], dtype=float) / float(fs) + shift
    frames = None
    for idx, label in enumerate(labels):
        y = np.asarray(data[idx], dtype=float).reshape(-1)
        t_crop, y_crop = _crop_aligned_series(base_t, y, start_sec, end_sec)
        if t_crop.size == 0:
            continue
        if frames is None:
            frames = np.arange(t_crop.size, dtype=np.int64)
            out["time"] = _relative_time(t_crop, start_sec)
            out["frames"] = frames
        field = _mat_field_name(label, used)
        out[field] = y_crop
        label_map[field] = label
    if SYNC_OTB_LABEL not in labels:
        t_sync, y_sync = _load_otb4_channel(path, SYNC_OTB_LABEL, repair_zones=repair_zones)
        t_sync = t_sync + shift
        t_crop, y_crop = _crop_aligned_series(t_sync, y_sync, start_sec, end_sec)
        if t_crop.size:
            if "time" not in out:
                out["time"] = _relative_time(t_crop, start_sec)
                out["frames"] = np.arange(t_crop.size, dtype=np.int64)
            field = _mat_field_name(SYNC_OTB_LABEL, used)
            out[field] = y_crop
            label_map[field] = SYNC_OTB_LABEL
    out["label_map"] = label_map
    out["sample_rate"] = float(record.get("sample_rate") or fs)
    return out


def _build_c3d_mat_struct(match: Dict[str, Any], start_sec: float, end_sec: float) -> Dict[str, Any]:
    record = match["c3d"]
    path = Path(record["path"])
    meta = _c3d_metadata(path)
    shift = float(_match_plot_shifts(match).get("c3d", 0.0))
    out: Dict[str, Any] = {"analog": {}, "point": {}, "cop": {}}

    analog_used: set[str] = set()
    analog_map: Dict[str, str] = {}
    for label in record.get("channel_names") or []:
        t, y = _load_c3d_analog_channel(path, label)
        t = t + shift
        t_crop, y_crop = _crop_aligned_series(t, y, start_sec, end_sec)
        if t_crop.size == 0:
            continue
        if "time" not in out["analog"]:
            out["analog"]["time"] = _relative_time(t_crop, start_sec)
            out["analog"]["frames"] = np.arange(t_crop.size, dtype=np.int64)
        field = _mat_field_name(label, analog_used)
        out["analog"][field] = y_crop
        analog_map[field] = label
    out["analog"]["label_map"] = analog_map
    out["analog"]["sample_rate"] = float(meta["analog_rate"])

    point_used: set[str] = set()
    point_map: Dict[str, str] = {}
    for label in record.get("point_channel_names") or []:
        tx, x = _load_c3d_point_channel(path, f"{label}.x")
        ty, y = _load_c3d_point_channel(path, f"{label}.y")
        tz, z = _load_c3d_point_channel(path, f"{label}.z")
        tx = tx + shift
        mask = np.isfinite(tx) & (tx >= start_sec) & (tx <= end_sec)
        if not np.any(mask):
            continue
        t_crop = np.asarray(tx[mask], dtype=float)
        xyz = np.column_stack([x[mask], y[mask], z[mask]])
        if "time" not in out["point"]:
            out["point"]["time"] = _relative_time(t_crop, start_sec)
            out["point"]["frames"] = np.arange(t_crop.size, dtype=np.int64)
        field = _mat_field_name(label, point_used)
        out["point"][field] = xyz
        point_map[field] = label
    out["point"]["label_map"] = point_map
    if meta.get("point_rate") is not None:
        out["point"]["sample_rate"] = float(meta["point_rate"])

    cop_used: set[str] = set()
    cop_map: Dict[str, str] = {}
    for label in ("copx", "copy", "copz"):
        try:
            t, y = _load_c3d_cop_channel(path, label)
        except Exception:
            continue
        t = t + shift
        t_crop, y_crop = _crop_aligned_series(t, y, start_sec, end_sec)
        if t_crop.size == 0:
            continue
        if "time" not in out["cop"]:
            out["cop"]["time"] = _relative_time(t_crop, start_sec)
            out["cop"]["frames"] = np.arange(t_crop.size, dtype=np.int64)
        field = _mat_field_name(label, cop_used)
        out["cop"][field] = y_crop
        cop_map[field] = label
    out["cop"]["label_map"] = cop_map
    return out


def _build_tsv_mat_struct(match: Dict[str, Any], start_sec: float, end_sec: float) -> Dict[str, Any]:
    record = match["tsv"]
    path = Path(record["path"])
    df = pd.read_csv(path, sep="\t", decimal=",", encoding="utf-8-sig")
    used: set[str] = set()
    label_map: Dict[str, str] = {}
    out: Dict[str, Any] = {}
    channels = _channel_list_for_record(record)
    if "t_rel" in df.columns:
        base_t = pd.to_numeric(df["t_rel"], errors="coerce").to_numpy(dtype=float)
    else:
        base_t = np.arange(len(df), dtype=float)
    shift = float(_match_plot_shifts(match).get("tsv", 0.0))
    base_t = base_t + shift
    for label in channels:
        y = pd.to_numeric(df[label], errors="coerce").to_numpy(dtype=float)
        y = np.nan_to_num(y, nan=np.nanmedian(y) if np.any(np.isfinite(y)) else 0.0)
        t_crop, y_crop = _crop_aligned_series(base_t, y, start_sec, end_sec)
        if t_crop.size == 0:
            continue
        if "time" not in out:
            out["time"] = _relative_time(t_crop, start_sec)
            out["frames"] = np.arange(t_crop.size, dtype=np.int64)
        field = _mat_field_name(label, used)
        out[field] = y_crop
        label_map[field] = label
    out["label_map"] = label_map
    return out


def _mat_safe(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _mat_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mat_safe(v) for v in value]
    return value


def _merge_info_path(mat_path: Path) -> Path:
    return mat_path.with_name(f"{mat_path.stem}_mergeInfo.txt")


def _write_merge_info_txt(mat_path: Path, match: Dict[str, Any]) -> Path:
    alignment = match.get("alignment") or {}
    inner_merge = alignment.get("inner_merge") or {}
    lines = [
        f"match_id={int(match['match_id']):03d}",
        f"mat_file={mat_path.name}",
        f"certainty={match.get('certainty')}",
        f"saved_at={match.get('saved_at') or ''}",
        f"inner_merge_start_sec={inner_merge.get('inner_merge_start_sec')}",
        f"inner_merge_end_sec={inner_merge.get('inner_merge_end_sec')}",
        f"inner_merge_duration_sec={inner_merge.get('inner_merge_duration_sec')}",
        f"otb4_file={Path(match['otb4']['path']).name if match.get('otb4') else ''}",
        f"otb4_path={Path(match['otb4']['path']).resolve() if match.get('otb4') else ''}",
        f"c3d_file={Path(match['c3d']['path']).name if match.get('c3d') else ''}",
        f"c3d_path={Path(match['c3d']['path']).resolve() if match.get('c3d') else ''}",
        f"tsv_file={Path(match['tsv']['path']).name if match.get('tsv') else ''}",
        f"tsv_path={Path(match['tsv']['path']).resolve() if match.get('tsv') else ''}",
        f"otb4_sync={match.get('otb4', {}).get('sync_channel') if match.get('otb4') else ''}",
        f"c3d_sync={match.get('c3d', {}).get('sync_channel') if match.get('c3d') else ''}",
        f"tsv_sync={match.get('tsv', {}).get('sync_channel') if match.get('tsv') else ''}",
        f"raw_tsv={alignment.get('raw_label_match_tsv')}",
        f"raw_c3d={alignment.get('raw_label_match_c3d')}",
        f"raw_alignment_quality={alignment.get('raw_alignment_quality')}",
        f"raw_alignment_corr={alignment.get('raw_alignment_corr')}",
        f"otb4_c3d_sync_pair={json.dumps((alignment.get('dedicated_sync_pairwise') or {}).get('otb4_vs_c3d') or {}, ensure_ascii=False)}",
        f"repair_json_path={((match.get('export_plan') or {}).get('repair_json') or {}).get('path') or ''}",
    ]
    txt_path = _merge_info_path(mat_path)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path


def export_match_mat(match: Dict[str, Any], export_root: Path) -> Path:
    inner_merge = ((match.get("alignment") or {}).get("inner_merge") or {})
    start_sec = inner_merge.get("inner_merge_start_sec")
    end_sec = inner_merge.get("inner_merge_end_sec")
    if not isinstance(start_sec, (int, float)) or not isinstance(end_sec, (int, float)) or end_sec <= start_sec:
        raise ValueError("Common matched time window is unavailable.")
    saved_stamp = match.get("saved_at") or _format_file_stamp()
    file_name = (((match.get("export_plan") or {}).get("mat") or {}).get("filename") or _mat_export_name(match, saved_stamp))
    out_path = Path(export_root) / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    otb4_struct = _build_otb4_mat_struct(match, float(start_sec), float(end_sec))
    c3d_struct = _build_c3d_mat_struct(match, float(start_sec), float(end_sec))
    tsv_struct = _build_tsv_mat_struct(match, float(start_sec), float(end_sec)) if match.get("tsv") else {}
    mat_dict = {
        "meta": {
            "match_id": int(match["match_id"]),
            "exported_at": _now_iso(),
            "time_window_sec": np.array([float(start_sec), float(end_sec)], dtype=float),
            "duration_sec": float(end_sec - start_sec),
            "common_period_aligned": True,
            "aligned_to_single_common_timebase": False,
            "per_source_timebases": True,
            "resampled": False,
            "certainty": str(match.get("certainty") or ""),
            "original_files": {
                key: {
                    "path": str(match[key]["path"]),
                    "filename": str(Path(match[key]["path"]).name),
                }
                for key in ("otb4", "c3d", "tsv")
                if match.get(key)
            },
            "renamed_files": {
                key: value.get("renamed_filename")
                for key, value in (((match.get("export_plan") or {}).get("matched") or {}).items())
            },
        },
        "alignment": {
            "plot_time_shifts_sec": (match.get("alignment") or {}).get("plot_time_shifts_sec") or {},
            "inner_merge": inner_merge,
            "otb4_repair": (match.get("alignment") or {}).get("otb4_repair") or {},
        },
        "otb4": otb4_struct,
        "c3d_analog": c3d_struct["analog"],
        "c3d_point": c3d_struct["point"],
        "c3d_cop": c3d_struct["cop"],
        "tsv": tsv_struct,
    }
    savemat(str(out_path), _mat_safe(mat_dict), long_field_names=True, do_compression=True)
    _write_merge_info_txt(out_path, match)
    return out_path


def _export_match_mat_task(args: Tuple[Dict[str, Any], str]) -> Dict[str, str]:
    match, export_root = args
    export_root_path = Path(export_root)
    base_mat = export_match_mat(match, export_root_path)
    pipe_mat, pipe_json = export_match_pipe_bundle(match, export_root_path)
    return {
        "mat_export": str(base_mat),
        "pipe_mat_export": str(pipe_mat),
        "pipe_json_export": str(pipe_json),
    }


def _pipe_mat_path(mat_path: Path) -> Path:
    return mat_path.with_name(f"{mat_path.stem}_4pipe.mat")


def _pipe_json_path(pipe_mat_path: Path) -> Path:
    return pipe_mat_path.with_suffix(".json")


def _description_texts(descs: Sequence[Any]) -> List[str]:
    labels: List[str] = []
    for d in descs:
        try:
            labels.append(str(d[0][0]))
        except Exception:
            labels.append(str(d))
    return labels


def _infer_otb_grids(labels: Sequence[str]) -> List[Dict[str, Any]]:
    grid_pattern = re.compile(r"HD(\d{2})MM(\d{2})(\d{2})")
    muscle_pattern = re.compile(r"\[MUSCLE:(.*?)\]")
    grids: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for idx, label in enumerate(labels):
        m = grid_pattern.search(label)
        if m:
            scale, rows, cols = map(int, m.groups())
            muscle_match = muscle_pattern.search(label)
            muscle = muscle_match.group(1).strip() if muscle_match else None
            if (
                current is not None
                and current["scale"] == scale
                and current["rows"] == rows
                and current["cols"] == cols
                and current.get("muscle") == muscle
                and idx == current["emg_indices"][-1] + 1
            ):
                current["emg_indices"].append(idx)
            else:
                current = {
                    "grid_uid": str(uuid.uuid4()),
                    "grid_key": f"{scale}mm_{rows}x{cols}",
                    "scale": scale,
                    "rows": rows,
                    "cols": cols,
                    "muscle": muscle,
                    "emg_indices": [idx],
                    "ref_indices": [],
                }
                grids.append(current)
        elif current is not None:
            current["ref_indices"].append(idx)
    for grid in grids:
        block = grid["emg_indices"] + grid["ref_indices"]
        grid["block_end"] = max(block) if block else max(grid["emg_indices"])
    return grids


def _otb_track_is_emg(track: Dict[str, Any]) -> bool:
    if bool(track.get("IsControl")):
        return False
    if str(track.get("Device") or "").strip() == "Syncstation":
        return False
    subtitle = str(track.get("SubTitle") or "").strip().upper()
    desc_name = str(track.get("DescriptionName") or "").strip().upper()
    grid_name = str((track.get("GridInfo") or {}).get("Name") or "").strip().upper()
    candidate = grid_name or desc_name or subtitle
    return candidate.startswith("HD") or candidate == "CDE-C"


def _infer_otb_emg_groups(otb_path: Path) -> List[Dict[str, Any]]:
    tmpdir, tracks = _extract_otb4_tracks(otb_path)
    try:
        groups: List[Dict[str, Any]] = []
        signal_paths = sorted({str(track.get("SignalStreamPath") or "") for track in tracks})
        for signal_path in signal_paths:
            if not signal_path:
                continue
            for offset, track in _otb4_track_offsets(tracks, signal_path):
                if not _otb_track_is_emg(track):
                    continue
                nchan = int(track.get("NumberOfChannels") or 0)
                if nchan <= 0:
                    continue
                device = str(track.get("Device") or "").strip()
                grid_info = track.get("GridInfo") or {}
                grid_name = str(grid_info.get("Name") or track.get("DescriptionName") or track.get("SubTitle") or "").strip()
                rows = int(grid_info.get("NRow") or 1)
                cols = int(grid_info.get("NColumn") or nchan)
                ied = int(grid_info.get("IED") or 0)
                muscle = str(track.get("Muscle") or "").strip() or None
                if grid_name.upper().startswith("HD"):
                    grid_key = f"{ied}mm_{rows}x{cols}"
                else:
                    grid_key = f"{device}_{grid_name}".replace(" ", "_")
                indices = list(range(offset, offset + nchan))
                if (
                    groups
                    and groups[-1]["device"] == device
                    and groups[-1]["grid_key"] == grid_key
                    and groups[-1]["emg_indices"][-1] + 1 == indices[0]
                ):
                    groups[-1]["emg_indices"].extend(indices)
                else:
                    groups.append(
                        {
                            "grid_uid": str(uuid.uuid4()),
                            "grid_key": grid_key,
                            "device": device,
                            "label": grid_name,
                            "rows": rows,
                            "cols": cols,
                            "scale": ied,
                            "muscle": muscle,
                            "emg_indices": indices,
                            "ref_indices": [],
                        }
                    )
        return groups
    finally:
        _release_otb4_tmpdir(tmpdir)


def _resample_to_target_time(target_t: np.ndarray, source_t: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_t, dtype=float)
    source_y = np.asarray(source_y, dtype=float)
    mask = np.isfinite(source_t) & np.isfinite(source_y)
    if np.count_nonzero(mask) < 2:
        fill = float(np.nanmean(source_y[mask])) if np.count_nonzero(mask) else 0.0
        return np.full(target_t.shape, fill, dtype=float)
    source_t = source_t[mask]
    source_y = source_y[mask]
    order = np.argsort(source_t)
    source_t = source_t[order]
    source_y = source_y[order]
    uniq_t, uniq_idx = np.unique(source_t, return_index=True)
    uniq_y = source_y[uniq_idx]
    if uniq_t.size < 2:
        return np.full(target_t.shape, float(uniq_y[0]), dtype=float)
    return np.interp(target_t, uniq_t, uniq_y, left=float(uniq_y[0]), right=float(uniq_y[-1]))


def _preferred_tsv_path_columns(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    names = record.get("channel_names") or []
    performed = next((c for c in names if c.strip().lower() == "performed"), None)
    if performed is None:
        raw_col, offset_col = _find_tsv_raw_offset_pair([str(c) for c in names])
        if raw_col and offset_col:
            performed = "performed"
    original = next((c for c in names if c.strip().lower() == "desired"), None)
    return performed, original


def _build_pipe_extra_specs(match: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    specs: List[Tuple[str, Dict[str, Any]]] = []
    tsv_record = match.get("tsv") or {}
    c3d_record = match.get("c3d") or {}
    performed_col, original_col = _preferred_tsv_path_columns(tsv_record)
    if performed_col:
        specs.append(("Performed Path", {"source": "tsv", "channel": performed_col}))
    if original_col:
        specs.append(("Original Path", {"source": "tsv", "channel": original_col}))

    for ch in _channel_list_for_record(tsv_record):
        if ch in {performed_col, original_col}:
            continue
        specs.append((f"TSV {ch}", {"source": "tsv", "channel": ch}))

    for ch in c3d_record.get("channel_names") or []:
        specs.append((f"C3D Analog {ch}", {"source": "c3d", "channel": ch, "kind": "analog"}))
    for ch in ("copx", "copy", "copz"):
        try:
            _load_c3d_cop_channel(Path(c3d_record["path"]), ch)
        except Exception:
            continue
        specs.append((f"C3D CoP {ch}", {"source": "c3d", "channel": ch, "kind": "cop"}))
    for ch in c3d_record.get("point_channel_names") or []:
        for coord in ("x", "y", "z"):
            specs.append((f"C3D Point {ch}.{coord}", {"source": "c3d", "channel": f"{ch}.{coord}", "kind": "point"}))
    return specs


def _load_aligned_series(match: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    source = spec["source"]
    channel = spec["channel"]
    if source == "tsv":
        t, y = _load_tsv_channel(Path(match["tsv"]["path"]), channel)
    elif source == "c3d":
        kind = spec.get("kind") or "analog"
        t, y = _load_c3d_series(Path(match["c3d"]["path"]), kind, channel)
    else:
        raise ValueError(source)
    shift = float(_match_plot_shifts(match).get(source, 0.0))
    return t + shift, y


def _build_common_otb_target(match: Dict[str, Any]) -> Dict[str, Any]:
    inner_merge = ((match.get("alignment") or {}).get("inner_merge") or {})
    start_sec = inner_merge.get("inner_merge_start_sec")
    end_sec = inner_merge.get("inner_merge_end_sec")
    if not isinstance(start_sec, (int, float)) or not isinstance(end_sec, (int, float)) or end_sec <= start_sec:
        raise ValueError("Common matched time window is unavailable.")

    otb_path = Path(match["otb4"]["path"])
    otb_data, otb_time, descs, fs, _name, _size = load_otb4_file(str(otb_path))
    otb_data = np.asarray(otb_data, dtype=float)
    labels = _description_texts(descs)
    full_len = _max_otb_repaired_length(match, int(otb_data.shape[1]), labels)
    otb_shift = float(_match_plot_shifts(match).get("otb4", 0.0))
    aligned_otb_time = np.arange(full_len, dtype=float) / float(fs) + otb_shift
    mask = np.isfinite(aligned_otb_time) & (aligned_otb_time >= float(start_sec)) & (aligned_otb_time <= float(end_sec))
    if not np.any(mask):
        raise ValueError("No OTB4 samples in common matched time window.")
    target_t = aligned_otb_time[mask]
    return {
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "fs": float(fs),
        "path": otb_path,
        "data": otb_data,
        "labels": labels,
        "full_len": int(full_len),
        "mask": mask,
        "target_t": target_t,
        "out_time": target_t - float(start_sec),
    }


def _build_common_aligned_mat_struct(match: Dict[str, Any]) -> Dict[str, Any]:
    bundle = _build_common_otb_target(match)
    target_t = np.asarray(bundle["target_t"], dtype=float)
    out_time = np.asarray(bundle["out_time"], dtype=float)
    fs = float(bundle["fs"])
    aligned: Dict[str, Any] = {
        "time": out_time,
        "frames": np.arange(target_t.size, dtype=np.int64),
        "sample_rate": fs,
        "otb4": {},
        "c3d_analog": {},
        "c3d_point": {},
        "c3d_cop": {},
        "tsv": {},
    }

    otb_used: set[str] = set()
    otb_map: Dict[str, str] = {}
    for idx, label in enumerate(bundle["labels"]):
        device = _otb_label_device(label)
        row_zones = _repair_zones_for_match(match, device=device)
        row = np.asarray(bundle["data"][idx], dtype=float).reshape(-1)
        if row_zones:
            row = _insert_nan_holes_signal(row, row_zones)
        if row.size < bundle["full_len"]:
            row = np.pad(row, (0, int(bundle["full_len"] - row.size)), constant_values=np.nan)
        elif row.size > bundle["full_len"]:
            row = row[: int(bundle["full_len"])]
        field = _mat_field_name(label, otb_used)
        aligned["otb4"][field] = row[bundle["mask"]]
        otb_map[field] = label
    aligned["otb4"]["label_map"] = otb_map

    c3d_record = match.get("c3d") or {}
    analog_used: set[str] = set()
    analog_map: Dict[str, str] = {}
    for label in c3d_record.get("channel_names") or []:
        t, y = _load_c3d_analog_channel(Path(c3d_record["path"]), label)
        t = t + float(_match_plot_shifts(match).get("c3d", 0.0))
        field = _mat_field_name(label, analog_used)
        aligned["c3d_analog"][field] = _resample_to_target_time(target_t, t, y)
        analog_map[field] = label
    aligned["c3d_analog"]["label_map"] = analog_map

    cop_used: set[str] = set()
    cop_map: Dict[str, str] = {}
    for label in ("copx", "copy", "copz"):
        try:
            t, y = _load_c3d_cop_channel(Path(c3d_record["path"]), label)
        except Exception:
            continue
        t = t + float(_match_plot_shifts(match).get("c3d", 0.0))
        field = _mat_field_name(label, cop_used)
        aligned["c3d_cop"][field] = _resample_to_target_time(target_t, t, y)
        cop_map[field] = label
    aligned["c3d_cop"]["label_map"] = cop_map

    point_used: set[str] = set()
    point_map: Dict[str, str] = {}
    for label in c3d_record.get("point_channel_names") or []:
        base_field = _mat_field_name(label, point_used)
        point_map[base_field] = label
        for coord in ("x", "y", "z"):
            t, y = _load_c3d_point_channel(Path(c3d_record["path"]), f"{label}.{coord}")
            t = t + float(_match_plot_shifts(match).get("c3d", 0.0))
            aligned["c3d_point"][f"{base_field}_{coord}"] = _resample_to_target_time(target_t, t, y)
    aligned["c3d_point"]["label_map"] = point_map

    tsv_record = match.get("tsv") or {}
    tsv_used: set[str] = set()
    tsv_map: Dict[str, str] = {}
    for label in _channel_list_for_record(tsv_record):
        t, y = _load_tsv_channel(Path(tsv_record["path"]), label)
        t = t + float(_match_plot_shifts(match).get("tsv", 0.0))
        field = _mat_field_name(label, tsv_used)
        aligned["tsv"][field] = _resample_to_target_time(target_t, t, y)
        tsv_map[field] = label
    aligned["tsv"]["label_map"] = tsv_map
    return aligned


def _build_pipe_bundle(match: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]:
    bundle = _build_common_otb_target(match)
    otb_path = Path(bundle["path"])
    otb_data = np.asarray(bundle["data"], dtype=float)
    labels = list(bundle["labels"])
    fs = float(bundle["fs"])
    path_grids = _infer_otb_grids(labels)
    emg_groups = _infer_otb_emg_groups(otb_path)
    mask = np.asarray(bundle["mask"], dtype=bool)
    target_t = np.asarray(bundle["target_t"], dtype=float)
    out_time = np.asarray(bundle["out_time"], dtype=float)
    target_fs = float(fs)
    data_out: List[np.ndarray] = []
    desc_out: List[str] = []
    json_means: Dict[str, Any] = {"filename": _pipe_mat_path(Path((((match.get("export_plan") or {}).get("mat") or {}).get("path") or "out.mat"))).name}

    grid_by_end = {grid["block_end"]: grid for grid in path_grids}
    original_to_output: Dict[int, int] = {}
    for idx, label in enumerate(labels):
        device = _otb_label_device(label)
        row_zones = _repair_zones_for_match(match, device=device)
        row = np.asarray(otb_data[idx], dtype=float).reshape(-1)
        if row_zones:
            row = _insert_nan_holes_signal(row, row_zones)
        if row.size < bundle["full_len"]:
            row = np.pad(row, (0, int(bundle["full_len"] - row.size)), constant_values=np.nan)
        elif row.size > bundle["full_len"]:
            row = row[: int(bundle["full_len"])]
        col = row[mask].astype(float, copy=True)
        data_out.append(col)
        out_idx = len(data_out) - 1
        desc_out.append(label)
        original_to_output[idx] = out_idx
        if idx in grid_by_end:
            if "Performed Path" in [d for d in desc_out[-2:]] or "Original Path" in [d for d in desc_out[-2:]]:
                continue
            performed_col, original_col = _preferred_tsv_path_columns(match.get("tsv") or {})
            if performed_col:
                t, y = _load_aligned_series(match, {"source": "tsv", "channel": performed_col})
                data_out.append(_resample_to_target_time(target_t, t, y))
                desc_out.append("Performed Path")
            if original_col:
                t, y = _load_aligned_series(match, {"source": "tsv", "channel": original_col})
                data_out.append(_resample_to_target_time(target_t, t, y))
                desc_out.append("Original Path")

    for label, spec in _build_pipe_extra_specs(match):
        if label in {"Performed Path", "Original Path"}:
            continue
        t, y = _load_aligned_series(match, spec)
        data_out.append(_resample_to_target_time(target_t, t, y))
        desc_out.append(label)

    data_matrix = np.column_stack(data_out).astype(np.float64, copy=False)
    for grid in emg_groups:
        entries: List[Dict[str, Any]] = []
        for original_idx in grid["emg_indices"]:
            out_idx = original_to_output[original_idx]
            mean_before = float(np.nanmean(data_matrix[:, out_idx]))
            data_matrix[:, out_idx] = data_matrix[:, out_idx] - mean_before
            mean_after = float(np.nanmean(data_matrix[:, out_idx]))
            entries.append(
                {
                    "channel_index": int(out_idx),
                    "method": "mean",
                    "mean_before": mean_before,
                    "mean_after": mean_after,
                }
            )
        json_means[grid["grid_uid"]] = entries

    desc_array = np.asarray(desc_out, dtype=object).reshape(-1, 1)
    return data_matrix, out_time, desc_array, target_fs, json_means


def export_match_pipe_bundle(match: Dict[str, Any], export_root: Path) -> Tuple[Path, Path]:
    saved_stamp = match.get("saved_at") or _format_file_stamp()
    base_mat_name = (((match.get("export_plan") or {}).get("mat") or {}).get("filename") or _mat_export_name(match, saved_stamp))
    base_mat_path = Path(export_root) / base_mat_name
    pipe_mat_path = _pipe_mat_path(base_mat_path)
    pipe_json_path = _pipe_json_path(pipe_mat_path)
    pipe_mat_path.parent.mkdir(parents=True, exist_ok=True)

    data, time, description, fs, json_means = _build_pipe_bundle(match)
    savemat(
        str(pipe_mat_path),
        {
            "Data": data,
            "Time": np.asarray(time, dtype=float),
            "Description": description,
            "SamplingFrequency": np.asarray([[float(fs)]], dtype=float),
        },
        do_compression=True,
        long_field_names=True,
    )
    pipe_json_path.write_text(json.dumps(json_means, indent=2), encoding="utf-8")
    return pipe_mat_path, pipe_json_path


def _series_label(spec: Dict[str, Any]) -> str:
    source = spec["source"]
    channel = spec["channel"]
    prefix = {"otb4": "OTB4", "c3d": "C3D", "tsv": "TSV"}.get(source, source.upper())
    if spec.get("kind") == "point":
        return f"{prefix} point: {channel}"
    if spec.get("kind") == "cop":
        return f"{prefix} CoP: {channel}"
    if spec.get("kind") == "sync":
        return f"{prefix} sync: {channel}"
    if spec.get("kind") == "raw":
        return f"{prefix} raw: {channel}"
    return f"{prefix}: {channel}"


def _series_color(spec: Dict[str, Any]) -> str:
    if spec.get("kind") == "sync":
        return {"otb4": "#0B84A5", "c3d": "#F6C85F", "tsv": "#6F4E7C"}.get(spec["source"], "#666666")
    if spec.get("kind") == "point":
        return "#D45087"
    if spec.get("kind") == "cop":
        return "#3A86FF"
    if spec.get("kind") == "raw":
        return {"c3d": "#005F73", "tsv": "#EE6C4D"}.get(spec["source"], "#A23B72")
    return {"otb4": "#00A6D6", "c3d": "#7F7F7F", "tsv": "#8E7DBE"}.get(spec["source"], "#444444")


def _overlay_series_color(spec: Dict[str, Any], idx: int, total: int) -> str:
    if total <= 1:
        return _series_color(spec)
    palette = [
        "#0B84A5",
        "#F28E2B",
        "#59A14F",
        "#E15759",
        "#4E79A7",
        "#EDC948",
        "#B07AA1",
        "#76B7B2",
        "#FF9DA7",
        "#9C755F",
    ]
    return palette[idx % len(palette)]


def _raw_overlay_summary(match: Dict[str, Any]) -> str:
    alignment = match.get("alignment") or {}
    raw_tsv = alignment.get("raw_label_match_tsv") or "missing"
    raw_c3d = alignment.get("raw_label_match_c3d") or "missing"
    quality = alignment.get("raw_alignment_quality") or "missing"
    corr = alignment.get("raw_alignment_corr")
    lag_sec = alignment.get("raw_alignment_lag_sec")
    extra = f"quality={quality}"
    if isinstance(corr, (int, float)):
        extra += f" corr={corr:.3f}"
    if isinstance(lag_sec, (int, float)):
        extra += f" lag={lag_sec:.3f}s"
    return f"Matched raw overlay | {extra} | C3D: {raw_c3d} | TSV: {raw_tsv}"


def _sync_annotation_text(match: Dict[str, Any], dedicated: Dict[str, Any]) -> str:
    alignment = match.get("alignment") or {}
    lines = [f"basis: {alignment.get('tsv_alignment_basis') or 'raw_only'}"]
    if dedicated.get("mean_abs_delta_ms") is not None:
        lines.append(
            f"dedicated sync: {dedicated['quality']}  mean={dedicated['mean_abs_delta_ms']:.3f} ms  "
            f"max={dedicated['max_abs_delta_ms']:.3f} ms  n={dedicated['spike_count']}"
        )
    else:
        lines.append(f"dedicated sync: {dedicated['quality']}")
    if dedicated.get("tsv_sync_missing"):
        lines.append(f"TSV dedicated sync missing; OTB4/C3D pair quality={dedicated.get('pair_quality')}")
    pair_bits = []
    for key, info in (dedicated.get("pairwise") or {}).items():
        pair_bits.append(f"{key} {info.get('matched_spikes_50ms', 0)}/{info.get('count', 0)} spikes<=50ms")
    if pair_bits:
        lines.append(" | ".join(pair_bits))
    raw_corr = alignment.get("raw_alignment_corr")
    if isinstance(raw_corr, (int, float)):
        lines.append(f"used raw alignment: {alignment.get('raw_alignment_quality')}  corr={raw_corr:.3f}  lag={alignment.get('raw_alignment_lag_sec'):.3f}s")
    return "\n".join(lines)


def _sync_status_text(match: Dict[str, Any], dedicated: Dict[str, Any]) -> str:
    alignment = match.get("alignment") or {}
    raw_corr = alignment.get("raw_alignment_corr")
    raw_text = f"raw {alignment.get('raw_alignment_quality')}"
    if isinstance(raw_corr, (int, float)):
        raw_text += f" corr {raw_corr:.3f}"
    if dedicated.get("tsv_sync_missing"):
        return f"Dedicated sync missing in TSV | OTB4/C3D {dedicated.get('pair_quality')} | {raw_text}"
    return f"Dedicated sync {dedicated.get('quality')} | {raw_text}"


def _raw_annotation_text(match: Dict[str, Any]) -> str:
    alignment = match.get("alignment") or {}
    corr = alignment.get("raw_alignment_corr")
    lag_sec = alignment.get("raw_alignment_lag_sec")
    if not isinstance(corr, (int, float)) or not isinstance(lag_sec, (int, float)):
        return "raw alignment unavailable"
    return (
        f"raw alignment: {alignment.get('raw_alignment_quality')}  "
        f"corr={corr:.6f}  lag={lag_sec:.3f}s  plotted_range=[0,1]\n"
        f"C3D {alignment.get('raw_label_match_c3d')}  <->  TSV {alignment.get('raw_label_match_tsv')}"
    )


def _raw_status_text(match: Dict[str, Any]) -> str:
    alignment = match.get("alignment") or {}
    corr = alignment.get("raw_alignment_corr")
    if isinstance(corr, (int, float)):
        return f"Raw overlay {alignment.get('raw_alignment_quality')} | corr {corr:.6f} | lag {alignment.get('raw_alignment_lag_sec'):.3f}s | plotted [0,1]"
    return "Raw overlay unavailable."


def _match_files_line(match: Dict[str, Any]) -> str:
    return (
        "Merged files: "
        f"OTB4={Path(str((match.get('otb4') or {}).get('path') or '')).name or 'missing'} | "
        f"C3D={Path(str((match.get('c3d') or {}).get('path') or '')).name or 'missing'} | "
        f"TSV={Path(str((match.get('tsv') or {}).get('path') or '')).name if match.get('tsv') else 'missing'}"
    )


def _pairwise_sync_metric_text(dedicated: Dict[str, Any], plotted_sources: Sequence[str]) -> str:
    pairwise = dedicated.get("pairwise") or {}
    labels: List[str] = []
    for idx, left in enumerate(plotted_sources):
        for right in plotted_sources[idx + 1:]:
            info = pairwise.get(f"{left}_vs_{right}") or pairwise.get(f"{right}_vs_{left}") or {}
            if not info:
                continue
            labels.append(
                f"{left.upper()}/{right.upper()} {int(info.get('matched_spikes_50ms') or 0)}/{int(info.get('count') or 0)}<=50ms"
            )
    return " | ".join(labels)


def _plot_source_files_text(match: Dict[str, Any], row_role: str, series_specs: Sequence[Dict[str, Any]]) -> str:
    if row_role == "sync":
        source_order = [source for source in ("otb4", "c3d", "tsv") if match.get(source) and any(spec.get("source") == source for spec in series_specs)]
    elif row_role == "raw":
        source_order = [source for source in ("c3d", "tsv") if match.get(source) and any(spec.get("source") == source for spec in series_specs)]
    else:
        source_order = []
        for spec in series_specs:
            source = str(spec.get("source") or "")
            if source in {"otb4", "c3d", "tsv"} and source not in source_order and match.get(source):
                source_order.append(source)
    if not source_order:
        return "Files in this plot: none"
    parts = [f"{source.upper()}={Path(str((match.get(source) or {}).get('path') or '')).name}" for source in source_order]
    return "Files in this plot: " + " | ".join(parts)


def _sync_plot_metric_text(match: Dict[str, Any], series_specs: Sequence[Dict[str, Any]]) -> str:
    dedicated = _dedicated_sync_agreement(match)
    plotted_sources = [str(spec.get("source") or "") for spec in series_specs if spec.get("kind") == "sync"]
    plotted_sources = [source for idx, source in enumerate(plotted_sources) if source and source not in plotted_sources[:idx]]
    parts = [f"Sync evidence: {dedicated.get('quality') or 'missing'}"]
    mean_abs = dedicated.get("mean_abs_delta_ms")
    max_abs = dedicated.get("max_abs_delta_ms")
    spike_count = dedicated.get("spike_count")
    if isinstance(mean_abs, (int, float)):
        parts.append(f"mean={float(mean_abs):.3f} ms")
    if isinstance(max_abs, (int, float)):
        parts.append(f"max={float(max_abs):.3f} ms")
    if isinstance(spike_count, (int, float)):
        parts.append(f"n={int(spike_count)}")
    pair_text = _pairwise_sync_metric_text(dedicated, plotted_sources)
    if pair_text:
        parts.append(pair_text)
    return " | ".join(parts)


def _raw_plot_metric_text(match: Dict[str, Any]) -> str:
    alignment = match.get("alignment") or {}
    corr = alignment.get("raw_alignment_corr")
    lag_sec = alignment.get("raw_alignment_lag_sec")
    quality = alignment.get("raw_alignment_quality") or "missing"
    if not isinstance(corr, (int, float)) or not isinstance(lag_sec, (int, float)):
        return "Raw evidence: unavailable"
    return (
        "Raw evidence: "
        f"{quality} | corr={float(corr):.6f} | lag={float(lag_sec):.3f}s | "
        f"C3D={alignment.get('raw_label_match_c3d') or 'missing'} | TSV={alignment.get('raw_label_match_tsv') or 'missing'}"
    )


def _plot_decision_context_text(match: Dict[str, Any], row_role: str, series_specs: Sequence[Dict[str, Any]]) -> str:
    kinds = {str(spec.get("kind") or "") for spec in series_specs}
    lines: List[str] = []
    if row_role == "sync" or "sync" in kinds:
        lines.append(_sync_plot_metric_text(match, series_specs))
    if row_role == "raw" or "raw" in kinds:
        lines.append(_raw_plot_metric_text(match))
    if not lines:
        lines.append("Decision evidence: this row is contextual only; sync/raw overlays carry the acceptance metrics.")
    return "\n".join(lines)


def _repair_gap_regions(match: Dict[str, Any]) -> List[Tuple[float, float]]:
    repair = _repair_info_for_match(match) or {}
    regions: List[Tuple[float, float]] = []
    for container in [repair, *((_device_repairs_for_match(match) or {}).values())]:
        for zone in container.get("zones") or []:
            start = zone.get("aligned_start_sec")
            end = zone.get("aligned_end_sec")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
                regions.append((float(start), float(end)))
    regions.sort()
    merged: List[Tuple[float, float]] = []
    for start, end in regions:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _repair_gap_style(match: Dict[str, Any]) -> Tuple[Tuple[int, int, int, int], str]:
    repair = _repair_info_for_match(match)
    if repair and repair.get("applied"):
        return (255, 214, 214, 28), "#E7A5A5"
    return (255, 240, 199, 28), "#D1A24A"


def _repair_status_text(match: Dict[str, Any]) -> str:
    repair = _repair_info_for_match(match)
    device_repairs = _device_repairs_for_match(match)
    if not repair and not device_repairs:
        return "OTB4 repair: no gap zones detected."
    device_count = len([value for value in device_repairs.values() if (value.get("zones") or [])])
    if not repair:
        return f"OTB4 device-specific gap candidates detected on {device_count} device(s)."
    if not repair.get("applied"):
        return (
            f"OTB4 gap candidate via {repair.get('device')} {repair.get('subtitle')} | "
            f"gaps {len(repair.get('zones') or [])} | +{int(repair.get('samples_added') or 0)} samples | "
            f"mean {repair.get('base_mean_abs_ms')} -> {repair.get('candidate_mean_abs_ms')} ms | "
            f"device candidates {device_count} | not auto-applied"
        )
    return (
        f"OTB4 repair applied via {repair.get('device')} {repair.get('subtitle')} | "
        f"gaps {len(repair.get('zones') or [])} | +{int(repair.get('samples_added') or 0)} samples | "
        f"mean {repair.get('base_mean_abs_ms')} -> {repair.get('repaired_mean_abs_ms')} ms | "
        f"device candidates {device_count}"
    )


def _y_axis_label(row_role: str, series_specs: Sequence[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    if row_role == "raw":
        return "Normalized value", None
    if len(series_specs) == 1 and series_specs[0].get("kind") in {"cop", "raw"}:
        return "Position", "m"
    return "Value", None


AUTO_ACCEPT_COLOR = "#CFEFD2"
USER_ACCEPT_COLOR = "#1E6B2D"
AUTO_REJECT_COLOR = "#F2CACA"
USER_REJECT_COLOR = "#8B1E1E"
NEUTRAL_BUTTON_COLOR = "#D9D9D9"


def _review_status_key_from_review(review: Dict[str, Any]) -> str:
    if review.get("user_decision") is True:
        return "user_accept"
    if review.get("user_decision") is False:
        return "user_reject"
    return "auto_accept" if review.get("auto_accept") else "auto_reject"


def _review_status_label_from_review(review: Dict[str, Any]) -> str:
    return {
        "auto_accept": "AUTO-ACCEPT",
        "user_accept": "USER-ACCEPT",
        "auto_reject": "AUTO-REJECT",
        "user_reject": "USER-REJECT",
    }[_review_status_key_from_review(review)]


def _review_status_key(match: Dict[str, Any]) -> str:
    review = match.get("review") or {}
    return _review_status_key_from_review(review)


def _review_status_label(match: Dict[str, Any]) -> str:
    review = match.get("review") or {}
    return _review_status_label_from_review(review)


def _decision_reason_text(match: Dict[str, Any]) -> str:
    alignment = match.get("alignment") or {}
    review = match.get("review") or {}
    pairwise = alignment.get("dedicated_sync_pairwise") or {}
    otb_c3d = pairwise.get("otb4_vs_c3d") or pairwise.get("c3d_vs_otb4") or {}
    raw_quality = alignment.get("raw_alignment_quality") or "missing"
    raw_corr = alignment.get("raw_alignment_corr")
    raw_lag = alignment.get("raw_alignment_lag_sec")
    edge_align = alignment.get("otb4_c3d_edge_alignment") or {}
    lines: List[str] = []
    lines.append(f"Automatic suggestion: {_review_status_label(match)}")
    lines.append(f"Decision source: {review.get('decision_source') or 'automatic'}")
    lines.append(f"Triplet certainty: {match.get('certainty')}")
    if match.get("manual_selection") or alignment.get("manual_selection"):
        lines.append("Manual assembly: this candidate was created by user-selected C3D/OTB4/TSV sources and should be confirmed visually.")
    if alignment.get("manual_reassignment"):
        lines.append(f"Manual reassignment: {alignment.get('manual_reassignment_note') or 'A source file was reassigned and this candidate now needs confirmation.'}")

    if isinstance(raw_corr, (int, float)) and isinstance(raw_lag, (int, float)):
        lines.append(
            f"Raw C3D/TSV match: {raw_quality} | corr={float(raw_corr):.6f} | lag={float(raw_lag):.3f}s"
        )
    else:
        lines.append(f"Raw C3D/TSV match: {raw_quality}")

    if otb_c3d:
        mean_ms = otb_c3d.get("mean_abs_ms")
        max_ms = otb_c3d.get("max_abs_ms")
        matched = int(otb_c3d.get("matched_spikes_50ms") or 0)
        count = int(otb_c3d.get("count") or 0)
        lines.append(
            f"OTB4/C3D sync: mean={mean_ms} ms | max={max_ms} ms | spikes within 50 ms={matched}/{count}"
        )
        if edge_align.get("basis") == "late_c3d_raw_bridge":
            lines.append(
                f"Sync bridge used: late C3D start inferred from raw alignment; skipped {int(edge_align.get('otb4_skip') or 0)} leading OTB4 sync pulse(s)."
            )
    else:
        lines.append("OTB4/C3D sync: unavailable")

    if match.get("tsv"):
        tsv_sync = pairwise.get("otb4_vs_tsv") or pairwise.get("c3d_vs_tsv") or {}
        if alignment.get("dedicated_sync_quality") == "missing_tsv_sync":
            lines.append("TSV dedicated sync channel is missing, so the decision relies on raw C3D/TSV agreement plus OTB4/C3D sync.")
        elif tsv_sync:
            lines.append(
                f"TSV dedicated sync check: OTB4/TSV spikes within 50 ms={int((pairwise.get('otb4_vs_tsv') or {}).get('matched_spikes_50ms') or 0)}/{int((pairwise.get('otb4_vs_tsv') or {}).get('count') or 0)} | "
                f"C3D/TSV spikes within 50 ms={int((pairwise.get('c3d_vs_tsv') or {}).get('matched_spikes_50ms') or 0)}/{int((pairwise.get('c3d_vs_tsv') or {}).get('count') or 0)}"
            )

    auto_accept = bool(review.get("auto_accept"))
    if auto_accept:
        lines.append("Why accepted: the match is certain, the raw C3D/TSV channel agrees strongly, and the OTB4/C3D sync pulses are within the auto-accept tolerance.")
    else:
        fail_reasons: List[str] = []
        if match.get("certainty") != "certain":
            fail_reasons.append("pairing certainty is below 'certain'")
        raw_ok = raw_quality == "excellent" or (
            raw_quality == "good" and isinstance(raw_corr, (int, float)) and abs(float(raw_corr)) >= 0.995
        )
        if not raw_ok:
            fail_reasons.append("raw C3D/TSV agreement did not reach the auto-accept threshold")
        if otb_c3d:
            mean_ms = otb_c3d.get("mean_abs_ms")
            max_ms = otb_c3d.get("max_abs_ms")
            matched = int(otb_c3d.get("matched_spikes_50ms") or 0)
            count = int(otb_c3d.get("count") or 0)
            sync_ok = False
            if count > 0 and matched >= count:
                sync_ok = True
            elif isinstance(mean_ms, (int, float)) and isinstance(max_ms, (int, float)):
                sync_ok = float(mean_ms) <= 20.0 and float(max_ms) <= 50.0
            if not sync_ok:
                fail_reasons.append("OTB4/C3D sync stayed outside the auto-accept limit (mean <= 20 ms and max <= 50 ms, or all spikes matched)")
        if alignment.get("dedicated_sync_quality") not in {"missing_tsv_sync", None}:
            otb_tsv = pairwise.get("otb4_vs_tsv") or {}
            c3d_tsv = pairwise.get("c3d_vs_tsv") or {}
            if int(otb_tsv.get("matched_spikes_50ms") or 0) == 0 or int(c3d_tsv.get("matched_spikes_50ms") or 0) == 0:
                fail_reasons.append("TSV dedicated sync spikes disagree strongly with OTB4/C3D")
        if fail_reasons:
            lines.append("Why rejected: " + " ".join(fail_reasons))

    lines.append("Guide: accept if the sync overlay and raw overlay look physically consistent across the green common time window; reject if the overlays require implausible pulse pairing or obvious mis-timing.")
    return "\n".join(lines)


def _decision_reason_rich_text(match: Dict[str, Any]) -> str:
    green = "#1F5E2E"
    red = "#7A1F1F"
    parts: List[str] = []
    for line in _decision_reason_text(match).splitlines():
        escaped = html.escape(line)
        color: Optional[str] = None
        if line.startswith("Automatic suggestion:"):
            if "ACCEPT" in line:
                color = green
            elif "REJECT" in line:
                color = red
        elif line.startswith("Why accepted:"):
            color = green
        elif line.startswith("Why rejected:"):
            color = red
        if color:
            parts.append(f'<span style="color: {color}; font-weight: 700;">{escaped}</span>')
        else:
            parts.append(escaped)
    return "<br>".join(parts)


def _review_status_color_from_review(review: Dict[str, Any]) -> str:
    return {
        "auto_accept": AUTO_ACCEPT_COLOR,
        "user_accept": USER_ACCEPT_COLOR,
        "auto_reject": AUTO_REJECT_COLOR,
        "user_reject": USER_REJECT_COLOR,
    }[_review_status_key_from_review(review)]


def _review_status_color(match: Dict[str, Any]) -> str:
    return _review_status_color_from_review(match.get("review") or {})


def _review_status_text_color(match: Dict[str, Any], selected: bool = False) -> str:
    if selected:
        return "#000000"
    return "#FFFFFF" if _review_status_key(match) in {"user_accept", "user_reject"} else "#000000"


def _style_button(button, bg_color: str, fg_color: str) -> None:
    button.setStyleSheet(
        f"QPushButton {{ background-color: {bg_color}; color: {fg_color}; font-weight: 600; padding: 6px 10px; }}"
    )


def _review_item_text(match: Dict[str, Any]) -> str:
    status = _review_status_label(match)
    quality = (match.get("alignment") or {}).get("dedicated_sync_quality")
    raw_q = (match.get("alignment") or {}).get("raw_alignment_quality")
    manual = " | MANUAL" if (
        match.get("manual_selection")
        or ((match.get("alignment") or {}).get("manual_selection"))
        or ((match.get("alignment") or {}).get("manual_reassignment"))
    ) else ""
    text = f"{match['match_id']:03d} | {status}{manual} | sync={quality} | raw={raw_q} | {Path(match['otb4']['path']).name}"
    if match.get("tsv"):
        text += f" | {Path(match['tsv']['path']).name}"
    else:
        text += " | TSV missing"
    return text


def _save_mapping_bundle(mapping_path: Path, mapping: Dict[str, Any]) -> None:
    saved_at = mapping.get("saved_at") or _format_file_stamp()
    mapping["saved_at"] = saved_at
    export_root = Path(mapping.get("export_root") or mapping_path.parent)
    for match in mapping.get("matches", []):
        _attach_export_plan(match, export_root, saved_at)
    _write_repair_jsons(export_root, mapping)
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_review_outputs(export_root, mapping, saved_at)
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def _review_progress(mapping: Dict[str, Any]) -> Tuple[int, int]:
    matches = mapping.get("matches", [])
    reviewed = sum(1 for match in matches if (match.get("review") or {}).get("reviewed"))
    return reviewed, len(matches)


def _refresh_mapping_summary(mapping: Dict[str, Any]) -> None:
    matches = mapping.get("matches", []) or []
    summary = mapping.setdefault("summary", {})
    summary["pair_count"] = len(matches)
    summary["certain_pair_count"] = sum(1 for match in matches if match.get("certainty") == "certain")
    summary["certain_triplet_count"] = sum(1 for match in matches if match.get("tsv"))
    summary["final_accept_count"] = sum(1 for match in matches if (match.get("review") or {}).get("final_accept"))
    source_records = mapping.get("source_records", {}) or {}
    used_paths = {
        key: {
            _match_source_path(match, key)
            for match in matches
            if _match_source_path(match, key)
        }
        for key in ("otb4", "c3d", "tsv")
    }
    summary["unmatched_c3d_count"] = sum(
        1 for rec in (source_records.get("c3d", []) or [])
        if str(rec.get("path") or "") and str(rec.get("path") or "") not in used_paths["c3d"]
    )
    summary["unmatched_otb4_count"] = sum(
        1 for rec in (source_records.get("otb4", []) or [])
        if str(rec.get("path") or "") and str(rec.get("path") or "") not in used_paths["otb4"]
    )
    summary["unmatched_tsv_count"] = sum(
        1 for rec in (source_records.get("tsv", []) or [])
        if str(rec.get("path") or "") and str(rec.get("path") or "") not in used_paths["tsv"]
    )


def _match_source_path(match: Dict[str, Any], key: str) -> str:
    return str(((match.get(key) or {}).get("path")) or "")


def _duplicate_source_assignments(matches: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, List[int]]]:
    dupes: Dict[str, Dict[str, List[int]]] = {"otb4": {}, "c3d": {}, "tsv": {}}
    for key in ("otb4", "c3d", "tsv"):
        seen: Dict[str, List[int]] = {}
        for idx, match in enumerate(matches):
            path = _match_source_path(match, key)
            if not path:
                continue
            seen.setdefault(path, []).append(idx)
        dupes[key] = {path: rows for path, rows in seen.items() if len(rows) > 1}
    return dupes


def _duplicate_assignment_lines(matches: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    dupes = _duplicate_source_assignments(matches)
    for key, mapping in dupes.items():
        for path, rows in mapping.items():
            refs = ", ".join(f"{int(matches[row].get('match_id') or row + 1):03d}" for row in rows)
            lines.append(f"{key}: {Path(path).name} used by matches {refs}")
    return lines


def _ensure_unique_source_assignments(matches: Sequence[Dict[str, Any]]) -> None:
    problem_lines = _duplicate_assignment_lines(matches)
    if problem_lines:
        raise RuntimeError("Duplicate source assignments detected:\n" + "\n".join(problem_lines))


_PICKER_CANCEL = object()
_PICKER_KEEP_CURRENT = object()
_PICKER_CLEAR = object()


class PlotRowWidget:
    def __init__(self, viewer, title: str, series_specs: Optional[List[Dict[str, Any]]] = None, row_role: str = "custom"):
        from PySide6 import QtWidgets
        import pyqtgraph as pg

        self.viewer = viewer
        self.title = title
        self.row_role = row_role
        self.series_specs: List[Dict[str, Any]] = list(series_specs or [])
        self.match: Optional[Dict[str, Any]] = None
        self.widget = QtWidgets.QFrame()
        self.widget.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout = QtWidgets.QVBoxLayout(self.widget)
        top = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel(title)
        self.label.setWordWrap(True)
        top.addWidget(self.label, 1)
        self.add_row_btn = QtWidgets.QPushButton("+ Row")
        self.add_overlay_btn = QtWidgets.QPushButton("+ Overlay")
        top.addWidget(self.add_row_btn)
        top.addWidget(self.add_overlay_btn)
        layout.addLayout(top)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.plot)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("QLabel { color: #404040; padding-top: 4px; }")
        layout.addWidget(self.status)
        self.add_row_btn.clicked.connect(self._request_new_row)
        self.add_overlay_btn.clicked.connect(self._request_overlay)

    def _request_new_row(self):
        if self.viewer is not None:
            self.viewer.add_plot_row()

    def _request_overlay(self):
        if self.viewer is not None:
            self.viewer.add_series_to_row(self)

    def set_match(self, match: Dict[str, Any]) -> None:
        self.match = match
        if self.row_role == "sync":
            self.series_specs = _sync_series_specs(match)
        elif self.row_role == "raw":
            self.series_specs = _raw_series_specs(match)
        self.refresh()

    def refresh(self) -> None:
        from PySide6 import QtCore
        import pyqtgraph as pg

        self.plot.clear()
        self.plot.addLegend()
        if self.match is None:
            self.label.setText(self.title)
            self.status.setText("No match selected.")
            return

        if self.row_role == "sync":
            text = "Dedicated sync overlay | light-green band = common matched time window"
            if _repair_gap_regions(self.match):
                text += " | pink/amber bands = OTB4 repaired or detected gap zones"
            self.label.setText(text)
        elif self.row_role == "raw":
            text = "Matched raw overlay | light-green band = common matched time window"
            if _repair_gap_regions(self.match):
                text += " | pink/amber bands = OTB4 repaired or detected gap zones"
            self.label.setText(text)
        else:
            series_names = ", ".join(_series_label(spec) for spec in self.series_specs) if self.series_specs else "No channels selected"
            self.label.setText(f"{self.title} | {series_names}")

        self.status.setText("")
        x_min: Optional[float] = None
        x_max: Optional[float] = None
        y_max: Optional[float] = None
        missing: List[str] = []
        plotted = 0
        for spec in self.series_specs:
            try:
                t, y = _load_series_for_spec(self.match, spec)
            except Exception as exc:
                missing.append(f"{_series_label(spec)} unavailable: {exc}")
                continue
            if len(t) == 0:
                missing.append(f"{_series_label(spec)} unavailable: empty data")
                continue
            if self.row_role == "raw":
                y = _normalize_unit_interval(y)
            color = _overlay_series_color(spec, plotted, len(self.series_specs))
            self.plot.plot(t, y, pen=pg.mkPen(color=color, width=1.4), name=_series_label(spec))
            plotted += 1
            lo = float(np.nanmin(t))
            hi = float(np.nanmax(t))
            x_min = lo if x_min is None else min(x_min, lo)
            x_max = hi if x_max is None else max(x_max, hi)
            y_hi = float(np.nanmax(y))
            y_max = y_hi if y_max is None else max(y_max, y_hi)

        footer_lines: List[str] = []
        if plotted:
            self.plot.setLabel("bottom", "Aligned time", units="s")
            y_label, y_units = _y_axis_label(self.row_role, self.series_specs)
            self.plot.setLabel("left", y_label, units=y_units)
            if x_min is not None and x_max is not None and np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                self.plot.setXRange(x_min, x_max, padding=0.03)
            else:
                self.plot.autoRange()
            inner_merge = ((self.match.get("alignment") or {}).get("inner_merge") or {})
            merge_start = inner_merge.get("inner_merge_start_sec")
            merge_end = inner_merge.get("inner_merge_end_sec")
            if isinstance(merge_start, (int, float)) and isinstance(merge_end, (int, float)) and merge_end > merge_start:
                region = pg.LinearRegionItem(values=(merge_start, merge_end), movable=False, brush=(190, 240, 190, 28))
                region.setZValue(-20)
                for line in region.lines:
                    line.setPen(pg.mkPen(color="#9FD89F", width=1.0, style=QtCore.Qt.PenStyle.DashLine))
                self.plot.addItem(region)
            gap_brush, gap_line = _repair_gap_style(self.match)
            for gap_start, gap_end in _repair_gap_regions(self.match):
                region = pg.LinearRegionItem(values=(gap_start, gap_end), movable=False, brush=gap_brush)
                region.setZValue(-19)
                for line in region.lines:
                    line.setPen(pg.mkPen(color=gap_line, width=1.0, style=QtCore.Qt.PenStyle.DashLine))
                self.plot.addItem(region)
            footer_lines.append(_plot_source_files_text(self.match, self.row_role, self.series_specs))
            footer_lines.append(_plot_decision_context_text(self.match, self.row_role, self.series_specs))
        else:
            footer_lines.append("No channels plotted.")

        if missing:
            footer_lines.append("Unavailable: " + " | ".join(missing))
        self.status.setText("\n".join(line for line in footer_lines if line))


class ViewerWindow:
    def __init__(self, mapping_path: Path, matched_dir: Path):
        from PySide6 import QtCore, QtGui, QtWidgets

        self.mapping_path = mapping_path
        self.matched_dir = matched_dir
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Sync Alignment Review")
        self.window.resize(1700, 1000)
        self.window.closeEvent = self._close_event
        self.window.resizeEvent = self._resize_event
        self._windowed_geometry = None
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { selection-background-color: rgba(0,0,0,0); selection-color: black; }"
            "QListWidget::item { padding: 6px; }"
            "QListWidget::item:selected:active { background-color: rgba(0,0,0,0); color: black; border: 2px solid black; }"
            "QListWidget::item:selected:!active { background-color: rgba(0,0,0,0); color: black; border: 2px solid black; }"
        )
        palette = self.list_widget.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(0, 0, 0, 0))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#000000"))
        self.list_widget.setPalette(palette)
        left_layout.addWidget(self.list_widget, 0)
        self.scan_notes = QtWidgets.QLabel("")
        self.scan_notes.setWordWrap(True)
        self.scan_notes.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.scan_notes.setStyleSheet("QLabel { background: #f4f4f4; border: 1px solid #d0d0d0; padding: 8px; }")
        left_layout.addWidget(self.scan_notes, 0)
        left_layout.addStretch(1)
        self.match_report_group = QtWidgets.QGroupBox("Match Report")
        self.match_report_layout = QtWidgets.QVBoxLayout(self.match_report_group)
        self.match_report_layout.setContentsMargins(8, 10, 8, 8)
        self.match_report_layout.setSpacing(6)
        self.sync_info = QtWidgets.QLabel("")
        self.sync_info.setWordWrap(True)
        self.sync_info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.match_report_layout.addWidget(self.sync_info, 0)
        self.raw_info = QtWidgets.QLabel("")
        self.raw_info.setWordWrap(True)
        self.raw_info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.match_report_layout.addWidget(self.raw_info, 0)
        self.decision_info = QtWidgets.QLabel("")
        self.decision_info.setWordWrap(True)
        self.decision_info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.match_report_layout.addWidget(self.decision_info, 0)
        self.progress = QtWidgets.QLabel("")
        self.progress.setWordWrap(True)
        self.match_report_layout.addWidget(self.progress, 0)
        self.task_progress = QtWidgets.QProgressBar()
        self.task_progress.setRange(0, 1)
        self.task_progress.setValue(0)
        self.task_progress.setFormat("Idle")
        self.match_report_layout.addWidget(self.task_progress, 0)
        left_layout.addWidget(self.match_report_group, 0)
        self.detail = QtWidgets.QScrollArea()
        self.detail.setWidgetResizable(True)
        self.detail_inner = QtWidgets.QWidget()
        self.detail_layout = QtWidgets.QVBoxLayout(self.detail_inner)
        self.detail_layout.setContentsMargins(8, 8, 8, 8)
        self.detail_layout.setSpacing(10)
        self.detail.setWidget(self.detail_inner)
        layout.addWidget(left_panel, 1)
        layout.addWidget(self.detail, 4)

        self.mapping = _load_mapping(mapping_path)
        self.matches = self.mapping.get("matches", [])
        self.source_records = self.mapping.get("source_records", {}) or {}
        self.unmatched_c3d_records = list(self.mapping.get("unmatched_c3d_records", []) or [])
        self.unmatched_otb4_records = list(self.mapping.get("unmatched_otb4_records", []) or [])
        for match in self.matches:
            _apply_review_defaults(match)
        self.current_match: Optional[Dict[str, Any]] = None
        self.plot_rows: List[PlotRowWidget] = []
        self._completed_notified = False
        self.completion_message: Optional[str] = None
        self._busy = False
        self._recompute_source_availability()

        self.header = QtWidgets.QLabel("")
        self.header.setWordWrap(True)
        self.detail_layout.addWidget(self.header)
        self.match_files_info = QtWidgets.QLabel("")
        self.match_files_info.setWordWrap(False)
        self.match_files_info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail_layout.addWidget(self.match_files_info)

        review_controls = QtWidgets.QHBoxLayout()
        self.accept_btn = QtWidgets.QPushButton("Accept -> Next")
        self.reject_btn = QtWidgets.QPushButton("Reject -> Next")
        self.reset_decision_btn = QtWidgets.QPushButton("Reset To Auto")
        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.next_btn = QtWidgets.QPushButton("Next")
        review_controls.addWidget(self.accept_btn)
        review_controls.addWidget(self.reject_btn)
        review_controls.addWidget(self.reset_decision_btn)
        review_controls.addStretch(1)
        review_controls.addWidget(self.prev_btn)
        review_controls.addWidget(self.next_btn)
        self.detail_layout.addLayout(review_controls)

        self.accept_btn.clicked.connect(lambda: self._decide_and_advance(True))
        self.reject_btn.clicked.connect(lambda: self._decide_and_advance(False))
        self.reset_decision_btn.clicked.connect(self.reset_user_decision)
        self.prev_btn.clicked.connect(lambda: self._move_selection(-1))
        self.next_btn.clicked.connect(lambda: self._move_selection(1))

        controls = QtWidgets.QHBoxLayout()
        self.add_row_btn = QtWidgets.QPushButton("+ Row")
        self.manual_match_btn = QtWidgets.QPushButton("Create Manual Match")
        self.edit_match_btn = QtWidgets.QPushButton("Set/Change Match Files")
        self.export_mat_btn = QtWidgets.QPushButton("Export accepted to .mat")
        self.parallel_export_chk = QtWidgets.QCheckBox("Parallel export")
        self.parallel_export_chk.setChecked(True)
        self.worker_spin = QtWidgets.QSpinBox()
        self.worker_spin.setRange(1, 64)
        self.worker_spin.setValue(6)
        self.worker_spin.setEnabled(True)
        self.reset_btn = QtWidgets.QPushButton("Reset")
        controls.addWidget(self.add_row_btn)
        controls.addWidget(self.manual_match_btn)
        controls.addWidget(self.edit_match_btn)
        controls.addWidget(self.export_mat_btn)
        controls.addWidget(self.parallel_export_chk)
        controls.addWidget(self.worker_spin)
        controls.addWidget(self.reset_btn)
        self.detail_layout.addLayout(controls)
        self.add_row_btn.clicked.connect(self.add_plot_row)
        self.manual_match_btn.clicked.connect(self.add_manual_match)
        self.edit_match_btn.clicked.connect(self.edit_current_match)
        self.export_mat_btn.clicked.connect(self.export_current_mat)
        self.parallel_export_chk.toggled.connect(self.worker_spin.setEnabled)
        self.reset_btn.clicked.connect(self.reset_rows)
        self.fullscreen_shortcut = QtGui.QShortcut(QtGui.QKeySequence("F12"), self.window)
        self.fullscreen_shortcut.activated.connect(self._toggle_fullscreen)
        self._apply_left_panel_sizing()

        self._refresh_match_list()
        self.list_widget.currentItemChanged.connect(self._on_select)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)
            current = self.list_widget.currentItem()
            if current is not None:
                self._on_select(current, None)

    def _refresh_match_list(self, selected_row: Optional[int] = None, preserve_selection: bool = True) -> None:
        from PySide6 import QtWidgets, QtGui, QtCore

        current_match_id = self.current_match.get("match_id") if self.current_match else None
        self.list_widget.clear()
        target_row = 0 if selected_row is None else max(0, int(selected_row))
        for row, match in enumerate(self.matches):
            item = QtWidgets.QListWidgetItem(_review_item_text(match))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            item.setBackground(QtGui.QColor(_review_status_color(match)))
            item.setForeground(QtGui.QColor(_review_status_text_color(match, selected=False)))
            self.list_widget.addItem(item)
            if selected_row is None and preserve_selection and current_match_id is not None and match.get("match_id") == current_match_id:
                target_row = row
        if self.list_widget.count():
            target_row = max(0, min(self.list_widget.count() - 1, target_row))
            was_blocked = self.list_widget.blockSignals(True)
            self.list_widget.setCurrentRow(target_row)
            self.list_widget.blockSignals(was_blocked)
            self.current_match = self.matches[target_row]
        else:
            self.current_match = None
        self._apply_list_item_styles()

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        enabled = not self._busy
        for widget in [
            self.list_widget,
            self.accept_btn,
            self.reject_btn,
            self.reset_decision_btn,
            self.prev_btn,
            self.next_btn,
            self.add_row_btn,
            self.manual_match_btn,
            self.edit_match_btn,
            self.export_mat_btn,
            self.parallel_export_chk,
            self.worker_spin,
            self.reset_btn,
        ]:
            widget.setEnabled(enabled)
        if self.parallel_export_chk.isChecked():
            self.worker_spin.setEnabled(enabled)
        for row in self.plot_rows:
            row.add_row_btn.setEnabled(enabled)
            row.add_overlay_btn.setEnabled(enabled)

    def _show_copyable_message(self, title: str, text: str) -> None:
        from PySide6 import QtGui, QtWidgets

        dialog = QtWidgets.QDialog(self.window)
        dialog.setWindowTitle(title)
        dialog.resize(1100, 480)
        layout = QtWidgets.QVBoxLayout(dialog)
        edit = QtWidgets.QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(text)
        layout.addWidget(edit)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        copy_btn = QtWidgets.QPushButton("Copy to Clipboard")
        close_btn = QtWidgets.QPushButton("Close")
        buttons.addWidget(copy_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        copy_btn.clicked.connect(lambda: QtGui.QGuiApplication.clipboard().setText(edit.toPlainText()))
        close_btn.clicked.connect(dialog.accept)
        dialog.exec()

    def _show_startup_messages(self) -> None:
        messages = [str(msg) for msg in (self.mapping.get("startup_messages") or []) if str(msg).strip()]
        if not messages:
            return
        text = "Scan notes for this review:\n\n" + "\n".join(f"- {msg}" for msg in messages)
        self._show_copyable_message("Scan Notes", text)

    def _apply_list_item_styles(self) -> None:
        from PySide6 import QtGui

        current_row = self.list_widget.currentRow()
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is None:
                continue
            match = self.matches[row]
            font = item.font()
            font.setBold(row == current_row)
            item.setFont(font)
            item.setBackground(QtGui.QColor(_review_status_color(match)))
            item.setForeground(QtGui.QColor(_review_status_text_color(match, selected=(row == current_row))))

    def _apply_left_panel_sizing(self) -> None:
        target_height = max(360, int(self.window.height() * 0.58))
        self.list_widget.setMinimumHeight(target_height)
        self.list_widget.setMaximumHeight(target_height)

    def _resize_event(self, event) -> None:
        from PySide6 import QtWidgets

        self._apply_left_panel_sizing()
        QtWidgets.QMainWindow.resizeEvent(self.window, event)

    def _toggle_fullscreen(self) -> None:
        if self.window.isFullScreen():
            self.window.showNormal()
            if self._windowed_geometry is not None:
                self.window.setGeometry(self._windowed_geometry)
        else:
            self._windowed_geometry = self.window.geometry()
            self.window.showFullScreen()

    def _update_header(self) -> None:
        if self.current_match is None:
            self.header.setText("")
            self.progress.setText("")
            self.scan_notes.setText("")
            self.match_files_info.setText("")
            self.sync_info.setText("")
            self.raw_info.setText("")
            self.decision_info.setText("")
            return
        alignment = self.current_match.get("alignment") or {}
        inner_merge = alignment.get("inner_merge") or {}
        review = self.current_match.get("review") or {}
        merge_start = inner_merge.get("inner_merge_start_sec")
        merge_end = inner_merge.get("inner_merge_end_sec")
        merge_duration = inner_merge.get("inner_merge_duration_sec")
        merge_text = "Common time window unavailable."
        if isinstance(merge_start, (int, float)) and isinstance(merge_end, (int, float)) and isinstance(merge_duration, (int, float)):
            merge_text = f"Common time window: {merge_start:.3f}s to {merge_end:.3f}s ({merge_duration:.3f}s)"
        self.header.setText(
            f"{_review_status_label(self.current_match)} | decision source: {review.get('decision_source')} | "
            f"certainty={self.current_match.get('certainty')} | exportable={'yes' if self.current_match.get('tsv') else 'no'}\n"
            f"{merge_text}"
        )
        reviewed_count, total_count = _review_progress(self.mapping)
        user_accept_count = sum(1 for match in self.matches if (match.get("review") or {}).get("user_decision") is True)
        user_reject_count = sum(1 for match in self.matches if (match.get("review") or {}).get("user_decision") is False)
        free_c3d = len(self.unmatched_c3d_records)
        free_otb4 = len(self.unmatched_otb4_records)
        free_tsv = sum(
            1
            for rec in (self.source_records.get("tsv", []) or [])
            if str(rec.get("path") or "") not in self._source_usage_map("tsv")
        )
        self.progress.setText(
            f"Reviewed {reviewed_count}/{total_count}. User accepted={user_accept_count}, user rejected={user_reject_count}. "
            f"Unassociated files: C3D={free_c3d}, OTB4={free_otb4}, TSV={free_tsv}. "
            f"Auto-accept suggests matches with excellent raw agreement and strong OTB4/C3D sync."
        )
        notes = [str(msg) for msg in (self.mapping.get("startup_messages") or []) if str(msg).strip()]
        self.scan_notes.setText(
            ("Scan notes:\n" + "\n".join(f"- {msg}" for msg in notes)) if notes else "Scan notes: none."
        )
        self.match_files_info.setText(_match_files_line(self.current_match))
        dedicated = _dedicated_sync_agreement(self.current_match)
        self.sync_info.setText(f"Sync: {_sync_status_text(self.current_match, dedicated)}\n{_repair_status_text(self.current_match)}")
        self.raw_info.setText(f"Raw: {_raw_status_text(self.current_match)}")
        if not self.current_match.get("tsv"):
            reason = (alignment.get("tsv_skip_reason") or "No TSV match reason recorded.")
            self.raw_info.setText(f"TSV missing: {reason}")
        self.decision_info.setText(_decision_reason_rich_text(self.current_match))
        self._update_review_controls()

    def _current_index(self) -> int:
        item = self.list_widget.currentItem()
        return self.list_widget.row(item) if item is not None else -1

    def _move_selection(self, delta: int) -> None:
        if self._busy or not self.list_widget.count():
            return
        idx = self._current_index()
        if idx < 0:
            idx = 0
        idx = max(0, min(self.list_widget.count() - 1, idx + delta))
        self.list_widget.setCurrentRow(idx)

    def _save_reviews(self) -> None:
        _ensure_unique_source_assignments(self.matches)
        self.mapping["matches"] = self.matches
        self.mapping["unmatched_c3d_records"] = self.unmatched_c3d_records
        self.mapping["source_records"] = self.source_records
        self.mapping["unmatched_otb4_records"] = self.unmatched_otb4_records
        _refresh_mapping_summary(self.mapping)
        reviewed_count, total_count = _review_progress(self.mapping)
        self.mapping.setdefault("summary", {})["reviewed_count"] = reviewed_count
        self.mapping.setdefault("summary", {})["final_accept_count"] = sum(
            1 for match in self.matches if (match.get("review") or {}).get("final_accept")
        )
        _save_mapping_bundle(self.mapping_path, self.mapping)

    def _set_progress(self, value: int, maximum: int, text_format: str) -> None:
        maximum = max(1, int(maximum))
        value = max(0, min(int(value), maximum))
        self.task_progress.setRange(0, maximum)
        self.task_progress.setValue(value)
        self.task_progress.setFormat(text_format)
        self.app.processEvents()

    def _update_review_controls(self) -> None:
        review = (self.current_match or {}).get("review") or {}
        key = _review_status_key_from_review(review) if review else None
        accept_bg = NEUTRAL_BUTTON_COLOR
        reject_bg = NEUTRAL_BUTTON_COLOR
        accept_fg = "#000000"
        reject_fg = "#000000"
        if key == "auto_accept":
            accept_bg = AUTO_ACCEPT_COLOR
        elif key == "user_accept":
            accept_bg = USER_ACCEPT_COLOR
            accept_fg = "#FFFFFF"
        elif key == "auto_reject":
            reject_bg = AUTO_REJECT_COLOR
        elif key == "user_reject":
            reject_bg = USER_REJECT_COLOR
            reject_fg = "#FFFFFF"
        _style_button(self.accept_btn, accept_bg, accept_fg)
        _style_button(self.reject_btn, reject_bg, reject_fg)
        _style_button(self.reset_decision_btn, "#E6E6E6", "#000000")
        self.reset_decision_btn.setEnabled(self.current_match is not None and review.get("user_decision") is not None)

    def _persist_refresh_and_maybe_finish(self, selected_row: Optional[int] = None) -> None:
        from PySide6 import QtWidgets

        self._set_busy(True)
        try:
            self._set_progress(0, 1, "Writing review outputs")
            try:
                self._save_reviews()
            except Exception as exc:
                self._set_progress(0, 1, "Save failed")
                QtWidgets.QMessageBox.critical(self.window, "Save Failed", str(exc))
                return
            self._refresh_match_list(selected_row=selected_row)
            if self.current_match is not None and self.plot_rows:
                self._refresh_rows()
            self._update_header()
            self._maybe_finish_review()
            self._set_progress(1, 1, "Ready")
        finally:
            self._set_busy(False)
            self._update_review_controls()

    def _decide_and_advance(self, accepted: bool) -> None:
        if self.current_match is None or self._busy:
            return
        current_row = self._current_index()
        next_row = current_row
        if current_row >= 0 and self.list_widget.count():
            next_row = min(self.list_widget.count() - 1, current_row + 1)
        self.set_user_decision(accepted, selected_row=current_row)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(next_row)

    def set_user_decision(self, accepted: bool, selected_row: Optional[int] = None) -> None:
        if self.current_match is None or self._busy:
            return
        review = self.current_match.setdefault("review", {})
        review["user_decision"] = bool(accepted)
        review["user_touched"] = True
        _apply_review_defaults(self.current_match)
        self._persist_refresh_and_maybe_finish(selected_row=selected_row)

    def reset_user_decision(self) -> None:
        if self.current_match is None or self._busy:
            return
        current_row = self._current_index()
        review = self.current_match.setdefault("review", {})
        review["user_decision"] = None
        review["user_touched"] = True
        _apply_review_defaults(self.current_match)
        self._persist_refresh_and_maybe_finish(selected_row=current_row)

    def _maybe_finish_review(self) -> None:
        from PySide6 import QtWidgets

        reviewed_count, total_count = _review_progress(self.mapping)
        if total_count > 0 and reviewed_count == total_count and not self._completed_notified:
            self._completed_notified = True
            export_root = Path(self.mapping.get("export_root") or self.mapping_path.parent)
            accepted_json = export_root / "accepted_matches.json"
            self.completion_message = (
                f"Review complete.\n\n"
                f"Wrote:\n- {self.mapping_path}\n- {export_root / 'alignment.log'}\n- {export_root / 'matched'}\n"
                f"- {accepted_json}\n\n"
                f"Use 'matched_files.json' for all reviewed candidates and 'accepted_matches.json' for final accepted matches."
            )
            QtWidgets.QMessageBox.information(self.window, "Complete", self.completion_message)

    def _clear_rows(self) -> None:
        for row in self.plot_rows:
            row.widget.setParent(None)
            row.widget.deleteLater()
        self.plot_rows = []
        gc.collect()

    def _ensure_default_rows(self) -> None:
        if self.current_match is None or self.plot_rows:
            return
        sync_row = PlotRowWidget(self, "Dedicated sync overlay", _sync_series_specs(self.current_match), row_role="sync")
        raw_row = PlotRowWidget(self, "Matched raw overlay", _raw_series_specs(self.current_match), row_role="raw")
        self.plot_rows.extend([sync_row, raw_row])
        self.detail_layout.addWidget(sync_row.widget)
        self.detail_layout.addWidget(raw_row.widget)

    def reset_rows(self) -> None:
        if self.current_match is None or self._busy:
            return
        self._clear_rows()
        self._ensure_default_rows()
        self._refresh_rows()

    def _source_usage_map(self, key: str, *, exclude_rows: Optional[Sequence[int]] = None) -> Dict[str, List[int]]:
        excluded = {int(row) for row in (exclude_rows or [])}
        usage: Dict[str, List[int]] = {}
        for row, match in enumerate(self.matches):
            if row in excluded:
                continue
            path = _match_source_path(match, key)
            if not path:
                continue
            usage.setdefault(path, []).append(row)
        return usage

    def _match_id_for_row(self, row: int) -> int:
        return int(self.matches[row].get("match_id") or row + 1)

    def _find_row_by_match_id(self, match_id: int) -> Optional[int]:
        for row, match in enumerate(self.matches):
            if int(match.get("match_id") or row + 1) == int(match_id):
                return row
        return None

    def _current_match_row(self) -> Optional[int]:
        if self.current_match is None:
            return None
        return self._find_row_by_match_id(int(self.current_match.get("match_id") or -1))

    def _match_brief_label(self, row: int) -> str:
        match = self.matches[row]
        otb_name = Path(str((match.get("otb4") or {}).get("path") or "")).name
        c3d_name = Path(str((match.get("c3d") or {}).get("path") or "")).name
        tsv_name = Path(str((match.get("tsv") or {}).get("path") or "")).name if match.get("tsv") else "no TSV"
        return f"{int(match.get('match_id') or row + 1):03d} ({otb_name} | {c3d_name} | {tsv_name})"

    def _record_usage_text(
        self,
        source_key: str,
        record: Dict[str, Any],
        *,
        exclude_rows: Optional[Sequence[int]] = None,
        current_record: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        path = str(record.get("path") or "")
        current_path = str((current_record or {}).get("path") or "")
        if current_path and path == current_path:
            rows = self._source_usage_map(source_key, exclude_rows=exclude_rows).get(path, [])
            if not rows:
                return "CURRENT", 0
            refs = ", ".join(self._match_brief_label(row) for row in rows)
            return f"CURRENT | ALSO USED by {refs}", 2
        rows = self._source_usage_map(source_key, exclude_rows=exclude_rows).get(path, [])
        if not rows:
            return "UNASSOCIATED", 1
        refs = ", ".join(self._match_brief_label(row) for row in rows)
        return f"USED by {refs}", 2

    def _recompute_source_availability(self) -> None:
        self.unmatched_c3d_records = [
            dict(rec) for rec in (self.source_records.get("c3d", []) or [])
            if str(rec.get("path") or "") not in self._source_usage_map("c3d")
        ]
        self.unmatched_otb4_records = [
            dict(rec) for rec in (self.source_records.get("otb4", []) or [])
            if str(rec.get("path") or "") not in self._source_usage_map("otb4")
        ]
        self.mapping["unmatched_c3d_records"] = self.unmatched_c3d_records
        self.mapping["unmatched_otb4_records"] = self.unmatched_otb4_records

    def _mark_match_for_manual_review(self, match: Dict[str, Any], note: str) -> None:
        alignment = match.setdefault("alignment", {})
        alignment["manual_reassignment"] = True
        alignment["manual_reassignment_note"] = note
        match["review"] = {"auto_accept": False, "user_decision": None, "user_touched": False}
        _apply_review_defaults(match)

    def _detach_tsv_from_match(self, row: int, note: str) -> None:
        old_match = self.matches[row]
        rebuilt = _build_pair_match_from_records(old_match["otb4"], old_match["c3d"], match_id=int(old_match.get("match_id") or row + 1))
        _apply_otb4_repairs([rebuilt], [])
        self._mark_match_for_manual_review(rebuilt, note)
        rebuilt["alignment"]["tsv_match_status"] = "manual_reassigned"
        rebuilt["alignment"]["tsv_skip_reason"] = note
        self.matches[row] = rebuilt

    def _remove_match(self, row: int) -> Dict[str, Any]:
        return self.matches.pop(row)

    def _build_manual_reassignment_plan(
        self,
        *,
        c3d_record: Dict[str, Any],
        otb4_record: Dict[str, Any],
        tsv_record: Optional[Dict[str, Any]],
        exclude_rows: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        excluded = {int(row) for row in (exclude_rows or [])}
        c3d_path = str(c3d_record.get("path") or "")
        otb_path = str(otb4_record.get("path") or "")
        tsv_path = str((tsv_record or {}).get("path") or "")

        c3d_rows = set(self._source_usage_map("c3d", exclude_rows=excluded).get(c3d_path, []))
        otb_rows = set(self._source_usage_map("otb4", exclude_rows=excluded).get(otb_path, []))
        tsv_rows = set(self._source_usage_map("tsv", exclude_rows=excluded).get(tsv_path, [])) if tsv_path else set()
        remove_rows = sorted(c3d_rows | otb_rows)
        detach_rows = sorted(tsv_rows - set(remove_rows))
        effects: List[str] = []

        for row in remove_rows:
            reasons: List[str] = []
            if row in c3d_rows:
                reasons.append("selected C3D")
            if row in otb_rows:
                reasons.append("selected OTB4")
            existing = self.matches[row]
            freed_bits = [
                Path(str((existing.get("otb4") or {}).get("path") or "")).name,
                Path(str((existing.get("c3d") or {}).get("path") or "")).name,
            ]
            if existing.get("tsv"):
                freed_bits.append(Path(str((existing.get("tsv") or {}).get("path") or "")).name)
            effects.append(
                f"{' and '.join(reasons).capitalize()} already belongs to match {self._match_brief_label(row)}. "
                f"That match will be removed and these files become free: {', '.join(bit for bit in freed_bits if bit)}."
            )
        for row in detach_rows:
            existing = self.matches[row]
            effects.append(
                f"Selected TSV already belongs to match {self._match_brief_label(row)}. "
                f"That match will keep its OTB4/C3D pair but lose the TSV and return to review."
            )
        return {
            "remove_match_ids": [self._match_id_for_row(row) for row in remove_rows],
            "detach_tsv_match_ids": [self._match_id_for_row(row) for row in detach_rows],
            "effects": effects,
        }

    def _confirm_manual_reassignment(
        self,
        *,
        context_label: str,
        c3d_record: Dict[str, Any],
        otb4_record: Dict[str, Any],
        tsv_record: Optional[Dict[str, Any]],
        exclude_rows: Optional[Sequence[int]] = None,
        current_match: Optional[Dict[str, Any]] = None,
    ) -> bool:
        from PySide6 import QtWidgets

        plan = self._build_manual_reassignment_plan(
            c3d_record=c3d_record,
            otb4_record=otb4_record,
            tsv_record=tsv_record,
            exclude_rows=exclude_rows,
        )
        effects = list(plan["effects"])
        current = current_match or {}
        for key, label in (("c3d", "C3D"), ("otb4", "OTB4"), ("tsv", "TSV")):
            old_path = str(((current.get(key) or {}).get("path")) or "")
            new_path = str((((tsv_record if key == "tsv" else (c3d_record if key == "c3d" else otb4_record)) or {}).get("path")) or "")
            if old_path and old_path != new_path:
                effects.append(f"Current match will release {label} {Path(old_path).name}.")
        if not effects:
            return True
        text = (
            f"{context_label}\n\n"
            + "\n".join(f"- {line}" for line in effects)
            + "\n\nContinue?"
        )
        result = QtWidgets.QMessageBox.question(
            self.window,
            "Confirm Re-Match",
            text,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return result == QtWidgets.QMessageBox.StandardButton.Yes

    def _record_picker(
        self,
        *,
        title: str,
        prompt: str,
        records: Sequence[Dict[str, Any]],
        source_key: Optional[str] = None,
        current_record: Optional[Dict[str, Any]] = None,
        exclude_rows: Optional[Sequence[int]] = None,
        include_keep_current: bool = False,
        clear_label: Optional[str] = None,
    ) -> Any:
        from PySide6 import QtWidgets

        options: List[str] = []
        lookup: Dict[str, Any] = {}
        if include_keep_current:
            current_name = Path(str((current_record or {}).get("path") or "")).name or "current"
            keep_label = f"<Keep Current: {current_name}>"
            options.append(keep_label)
            lookup[keep_label] = _PICKER_KEEP_CURRENT
        if clear_label:
            options.append(clear_label)
            lookup[clear_label] = _PICKER_CLEAR
        ordered_records = list(records)
        if source_key:
            ordered_records.sort(
                key=lambda record: (
                    self._record_usage_text(
                        source_key,
                        record,
                        exclude_rows=exclude_rows,
                        current_record=current_record,
                    )[1],
                    Path(str(record.get("path") or "")).name.lower(),
                )
            )
        for record in ordered_records:
            path = Path(str(record.get("path") or ""))
            bits = [path.name]
            if source_key:
                usage_text, _order = self._record_usage_text(
                    source_key,
                    record,
                    exclude_rows=exclude_rows,
                    current_record=current_record,
                )
                bits.append(usage_text)
            if "sync_quality" in record:
                bits.append(f"sync={record.get('sync_quality')}")
            if "edge_count" in record:
                bits.append(f"edges={int(record.get('edge_count') or 0)}")
            label = " | ".join(bits)
            suffix = 2
            base_label = label
            while label in lookup:
                suffix += 1
                label = f"{base_label} [{suffix}]"
            options.append(label)
            lookup[label] = dict(record)
        if not options:
            return _PICKER_CANCEL
        choice, ok = QtWidgets.QInputDialog.getItem(self.window, title, prompt, options, 0, False)
        if not ok or not choice:
            return _PICKER_CANCEL
        return lookup.get(choice, _PICKER_CANCEL)

    def _resolve_record_choice(self, choice: Any, current_record: Optional[Dict[str, Any]]) -> Any:
        if choice is _PICKER_CANCEL:
            return _PICKER_CANCEL
        if choice is _PICKER_KEEP_CURRENT:
            return dict(current_record) if current_record else None
        if choice is _PICKER_CLEAR:
            return None
        return choice

    def _choose_match_source(
        self,
        *,
        title: str,
        prompt: str,
        records: Sequence[Dict[str, Any]],
        source_key: str,
        current_record: Optional[Dict[str, Any]] = None,
        exclude_rows: Optional[Sequence[int]] = None,
        include_keep_current: bool = False,
        clear_label: Optional[str] = None,
    ) -> Any:
        choice = self._record_picker(
            title=title,
            prompt=prompt,
            records=records,
            source_key=source_key,
            current_record=current_record,
            exclude_rows=exclude_rows,
            include_keep_current=include_keep_current,
            clear_label=clear_label,
        )
        return self._resolve_record_choice(choice, current_record)

    def _build_manual_match(
        self,
        *,
        c3d_record: Dict[str, Any],
        otb4_record: Dict[str, Any],
        tsv_record: Optional[Dict[str, Any]],
        match_id: int,
        note: str,
    ) -> Tuple[Dict[str, Any], bool]:
        manual_match = _build_pair_match_from_records(otb4_record, c3d_record, match_id=match_id)
        manual_match["manual_selection"] = True
        manual_match.setdefault("alignment", {})["manual_selection"] = True
        _apply_otb4_repairs([manual_match], [])
        tsv_ok = True
        if tsv_record is not None:
            tsv_ok = _apply_manual_tsv_selection(
                manual_match,
                tsv_record,
                filename_clock_offset_sec=float(self.mapping.get("filename_clock_offset_sec") or 0.0),
            )
        self._mark_match_for_manual_review(manual_match, note)
        return manual_match, bool(tsv_ok)

    def _select_match_sources(self, *, current_row: Optional[int]) -> Any:
        from PySide6 import QtWidgets

        if current_row is None:
            self._recompute_source_availability()
            c3d_record = self._choose_match_source(
                title="Select C3D",
                prompt="Choose an unmatched C3D file to assemble manually.",
                records=self.unmatched_c3d_records,
                source_key="c3d",
            )
            if c3d_record is _PICKER_CANCEL:
                if not self.unmatched_c3d_records:
                    QtWidgets.QMessageBox.information(
                        self.window,
                        "No Unmatched C3D",
                        "No unmatched C3D files are available for manual assembly.",
                    )
                return _PICKER_CANCEL
            otb4_record = self._choose_match_source(
                title="Select OTB4",
                prompt="Choose the OTB4 file that belongs with this C3D file. Unassociated files are listed first.",
                records=self.source_records.get("otb4", []) or [],
                source_key="otb4",
            )
            if otb4_record is _PICKER_CANCEL or otb4_record is None:
                return _PICKER_CANCEL
            tsv_record = self._choose_match_source(
                title="Select TSV",
                prompt="Choose the TSV file for this manual candidate, or leave TSV missing for now.",
                records=self.source_records.get("tsv", []) or [],
                source_key="tsv",
                clear_label="<Leave TSV Missing>",
            )
            if tsv_record is _PICKER_CANCEL:
                return _PICKER_CANCEL
            return {"c3d": c3d_record, "otb4": otb4_record, "tsv": tsv_record}

        current_match = self.matches[current_row]
        c3d_record = self._choose_match_source(
            title="Set C3D",
            prompt="Choose a C3D file for this match, or keep the current C3D.",
            records=self.source_records.get("c3d", []) or [],
            source_key="c3d",
            current_record=current_match.get("c3d"),
            exclude_rows=[current_row],
            include_keep_current=True,
        )
        if c3d_record is _PICKER_CANCEL or c3d_record is None:
            return _PICKER_CANCEL
        otb4_record = self._choose_match_source(
            title="Set OTB4",
            prompt="Choose an OTB4 file for this match, or keep the current OTB4.",
            records=self.source_records.get("otb4", []) or [],
            source_key="otb4",
            current_record=current_match.get("otb4"),
            exclude_rows=[current_row],
            include_keep_current=True,
        )
        if otb4_record is _PICKER_CANCEL or otb4_record is None:
            return _PICKER_CANCEL
        tsv_clear_label = "<Keep TSV Missing>" if not current_match.get("tsv") else "<Set TSV Missing>"
        tsv_record = self._choose_match_source(
            title="Set TSV",
            prompt="Choose a TSV file, keep the current TSV, or explicitly leave/set TSV missing.",
            records=self.source_records.get("tsv", []) or [],
            source_key="tsv",
            current_record=current_match.get("tsv"),
            exclude_rows=[current_row],
            include_keep_current=bool(current_match.get("tsv")),
            clear_label=tsv_clear_label,
        )
        if tsv_record is _PICKER_CANCEL:
            return _PICKER_CANCEL
        return {"c3d": c3d_record, "otb4": otb4_record, "tsv": tsv_record}

    def _apply_manual_selection(
        self,
        *,
        c3d_record: Dict[str, Any],
        otb4_record: Dict[str, Any],
        tsv_record: Optional[Dict[str, Any]],
        selected_row: Optional[int],
        note: str,
    ) -> int:
        from PySide6 import QtWidgets

        target_match_id = self._match_id_for_row(selected_row) if selected_row is not None else None
        exclude_rows = [selected_row] if selected_row is not None else []
        plan = self._build_manual_reassignment_plan(
            c3d_record=c3d_record,
            otb4_record=otb4_record,
            tsv_record=tsv_record,
            exclude_rows=exclude_rows,
        )

        for match_id in sorted(plan["remove_match_ids"], reverse=True):
            row = self._find_row_by_match_id(match_id)
            if row is not None:
                self._remove_match(row)
        for match_id in plan["detach_tsv_match_ids"]:
            row = self._find_row_by_match_id(match_id)
            if row is not None:
                note_text = f"TSV {Path(str((tsv_record or {}).get('path') or '')).name} was reassigned by a manual overwrite."
                self._detach_tsv_from_match(row, note_text)
        self._recompute_source_availability()

        if selected_row is None:
            match_id = max([int(match.get("match_id") or 0) for match in self.matches] + [0]) + 1
            manual_match, tsv_ok = self._build_manual_match(
                c3d_record=c3d_record,
                otb4_record=otb4_record,
                tsv_record=tsv_record,
                match_id=match_id,
                note=note,
            )
            self.matches.append(manual_match)
            selected_row = len(self.matches) - 1
        else:
            current_row = self._find_row_by_match_id(int(target_match_id or -1))
            if current_row is None:
                raise RuntimeError("Current match disappeared while applying the manual re-match.")
            manual_match, tsv_ok = self._build_manual_match(
                c3d_record=c3d_record,
                otb4_record=otb4_record,
                tsv_record=tsv_record,
                match_id=int(target_match_id),
                note=note,
            )
            self.matches[current_row] = manual_match
            selected_row = current_row

        self._recompute_source_availability()
        self.mapping["matches"] = self.matches
        if not tsv_ok:
            QtWidgets.QMessageBox.warning(
                self.window,
                "Manual TSV Failed",
                "The selected TSV could not be aligned automatically. The C3D/OTB4 pair was still updated for review.",
            )
        return int(selected_row)

    def add_manual_match(self) -> None:
        if self._busy:
            return
        selected = self._select_match_sources(current_row=None)
        if selected is _PICKER_CANCEL:
            return
        c3d_record = selected["c3d"]
        otb4_record = selected["otb4"]
        tsv_record = selected["tsv"]
        if not self._confirm_manual_reassignment(
            context_label=f"Manual assembly for {Path(str(c3d_record.get('path') or '')).name} will change source assignments.",
            c3d_record=c3d_record,
            otb4_record=otb4_record,
            tsv_record=tsv_record,
        ):
            return
        selected_row = self._apply_manual_selection(
            c3d_record=c3d_record,
            otb4_record=otb4_record,
            tsv_record=tsv_record,
            selected_row=None,
            note="Manual C3D/OTB4/TSV selection created this candidate.",
        )
        self._persist_refresh_and_maybe_finish(selected_row=selected_row)

    def edit_current_match(self) -> None:
        from PySide6 import QtWidgets

        if self._busy or self.current_match is None:
            return
        current_row = self._current_match_row()
        if current_row is None:
            QtWidgets.QMessageBox.warning(self.window, "Edit Match", "The selected match could not be located.")
            return
        current_match = self.matches[current_row]
        selected = self._select_match_sources(current_row=current_row)
        if selected is _PICKER_CANCEL:
            return
        c3d_record = selected["c3d"]
        otb4_record = selected["otb4"]
        tsv_record = selected["tsv"]
        old_c3d = str((current_match.get("c3d") or {}).get("path") or "")
        old_otb4 = str((current_match.get("otb4") or {}).get("path") or "")
        old_tsv = str((current_match.get("tsv") or {}).get("path") or "")
        new_c3d = str(c3d_record.get("path") or "")
        new_otb4 = str(otb4_record.get("path") or "")
        new_tsv = str((tsv_record or {}).get("path") or "")
        if old_c3d == new_c3d and old_otb4 == new_otb4 and old_tsv == new_tsv:
            QtWidgets.QMessageBox.information(self.window, "Edit Match", "No source files were changed.")
            return
        if not self._confirm_manual_reassignment(
            context_label=f"Re-matching {self._match_brief_label(current_row)} will update the current triplet.",
            c3d_record=c3d_record,
            otb4_record=otb4_record,
            tsv_record=tsv_record,
            exclude_rows=[current_row],
            current_match=current_match,
        ):
            return
        selected_row = self._apply_manual_selection(
            c3d_record=c3d_record,
            otb4_record=otb4_record,
            tsv_record=tsv_record,
            selected_row=current_row,
            note="Manual file reassignment updated this candidate.",
        )
        self._persist_refresh_and_maybe_finish(selected_row=selected_row)

    def export_current_mat(self) -> None:
        from PySide6 import QtWidgets

        if self._busy:
            return
        reviewed_count, total_count = _review_progress(self.mapping)
        if total_count > 0 and reviewed_count < total_count:
            QtWidgets.QMessageBox.information(
                self.window,
                "Review Not Complete",
                "Complete the review for all matches before exporting accepted MAT files.",
            )
            return
        accepted_matches = [
            match for match in self.matches
            if (match.get("review") or {}).get("final_accept") and match.get("tsv")
        ]
        if not accepted_matches:
            QtWidgets.QMessageBox.information(
                self.window,
                "No Accepted Matches",
                "No accepted triplets with TSV/OTB4/C3D are available for MAT export.",
            )
            return
        export_root = Path(self.mapping.get("export_root") or self.mapping_path.parent) / "matched"
        export_root.mkdir(parents=True, exist_ok=True)
        written: List[str] = []
        self._set_busy(True)
        try:
            self._set_progress(0, len(accepted_matches), "Exporting accepted MAT files %v/%m")
            use_parallel = self.parallel_export_chk.isChecked() and len(accepted_matches) > 1
            n_workers = min(self.worker_spin.value(), len(accepted_matches))
            if use_parallel:
                tasks = [(match, str(export_root)) for match in accepted_matches]
                future_map: Dict[concurrent.futures.Future[Dict[str, str]], Dict[str, Any]] = {}
                try:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        for task, match in zip(tasks, accepted_matches):
                            future_map[executor.submit(_export_match_mat_task, task)] = match
                        completed = 0
                        for future in concurrent.futures.as_completed(future_map):
                            match = future_map[future]
                            try:
                                outputs = future.result()
                            except Exception as exc:
                                self._set_progress(0, 1, "MAT export failed")
                                QtWidgets.QMessageBox.critical(
                                    self.window,
                                    "MAT Export Failed",
                                    f"Failed while exporting match {match['match_id']:03d}:\n{exc}",
                                )
                                return
                            match.setdefault("review_outputs", {}).update(outputs)
                            written.extend([outputs["mat_export"], outputs["pipe_mat_export"], outputs["pipe_json_export"]])
                            completed += 1
                            self._set_progress(completed, len(accepted_matches), "Exporting accepted MAT files %v/%m")
                except Exception as exc:
                    self._set_progress(0, 1, "MAT export failed")
                    QtWidgets.QMessageBox.critical(
                        self.window,
                        "MAT Export Failed",
                        f"Parallel MAT export failed before completion:\n{exc}",
                    )
                    return
            else:
                for idx, match in enumerate(accepted_matches, start=1):
                    try:
                        out_path = export_match_mat(match, export_root)
                        pipe_mat_path, pipe_json_path = export_match_pipe_bundle(match, export_root)
                    except Exception as exc:
                        self._set_progress(0, 1, "MAT export failed")
                        QtWidgets.QMessageBox.critical(
                            self.window,
                            "MAT Export Failed",
                            f"Failed while exporting match {match['match_id']:03d}:\n{exc}",
                        )
                        return
                    review_outputs = match.setdefault("review_outputs", {})
                    review_outputs["mat_export"] = str(out_path)
                    review_outputs["pipe_mat_export"] = str(pipe_mat_path)
                    review_outputs["pipe_json_export"] = str(pipe_json_path)
                    written.extend([str(out_path), str(pipe_mat_path), str(pipe_json_path)])
                    self._set_progress(idx, len(accepted_matches), "Exporting accepted MAT files %v/%m")
            self._save_reviews()
            self._set_progress(1, 1, "Ready")
            QtWidgets.QMessageBox.information(
                self.window,
                "MAT Exported",
                "Wrote per accepted triplet: .mat, _4pipe.mat, and _4pipe.json\n" + "\n".join(written),
            )
        finally:
            self._set_busy(False)
            self._update_review_controls()

    def _refresh_rows(self) -> None:
        self._set_busy(True)
        total = max(1, len(self.plot_rows))
        try:
            self._set_progress(0, total, "Loading selected match %v/%m")
            for idx, row in enumerate(self.plot_rows, start=1):
                row.set_match(self.current_match)
                self._set_progress(idx, total, "Loading selected match %v/%m")
            self._set_progress(1, 1, "Ready")
        finally:
            self._set_busy(False)
            self._update_review_controls()

    def add_plot_row(self) -> None:
        if self.current_match is None or self._busy:
            return
        row = PlotRowWidget(self, f"Plot row {len(self.plot_rows) - 1}")
        insert_at = self.detail_layout.count()
        self.plot_rows.append(row)
        self.detail_layout.insertWidget(insert_at, row.widget)
        row.set_match(self.current_match)
        self.add_series_to_row(row, replace=True)

    def add_series_to_row(self, row: PlotRowWidget, replace: bool = False) -> None:
        if self.current_match is None or self._busy:
            return
        from PySide6 import QtWidgets

        options: List[str] = []
        lookup: Dict[str, Dict[str, Any]] = {}
        for source in ("otb4", "c3d", "tsv"):
            rec = self.current_match.get(source)
            if not rec:
                continue
            for label, spec in _channel_options_for_record(rec):
                options.append(label)
                lookup[label] = spec
        if not options:
            QtWidgets.QMessageBox.information(self.window, "No channels", "No channels available to plot.")
            return
        choice, ok = QtWidgets.QInputDialog.getItem(
            self.window,
            "Select Channel",
            "Which channel should be plotted?",
            options,
            0,
            False,
        )
        if not ok or not choice:
            return
        if replace:
            row.series_specs = [lookup[choice]]
        else:
            row.series_specs.append(lookup[choice])
        row.match = self.current_match
        row.refresh()

    def _on_select(self, current, previous):
        from PySide6 import QtCore

        if self._busy or current is None:
            return
        row = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if row is None:
            return
        match = self.matches[int(row)]
        self.current_match = match
        self._apply_list_item_styles()
        if not self.plot_rows:
            self._ensure_default_rows()
        self._refresh_rows()
        self._update_header()

    def run(self) -> int:
        self._windowed_geometry = self.window.geometry()
        self.window.showFullScreen()
        return self.app.exec()

    def _close_event(self, event) -> None:
        from PySide6 import QtWidgets

        reviewed_count, total_count = _review_progress(self.mapping)
        if total_count > 0 and reviewed_count < total_count:
            answer = QtWidgets.QMessageBox.warning(
                self.window,
                "Quit Early?",
                f"Only {reviewed_count}/{total_count} matches have been reviewed. Quit anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._save_reviews()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Match OTB4, C3D, and TSV files by sync signal.")
    sub = p.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Scan a folder and generate mapping/log/matched copies.")
    scan.add_argument("folder", help="Folder containing the .otb4, .tsv, and .c3d files.")

    view = sub.add_parser("view", help="Open the PySide6 viewer for the generated mapping.")
    view.add_argument("folder", help="Folder containing matched_files.json and matched/.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "scan":
        folder = Path(args.folder)
        mapping_path, log_path, matched_dir = run_scan(folder)
        print(str(mapping_path))
        print(str(log_path))
        print(str(matched_dir))
        return 0
    if args.cmd == "view":
        folder = Path(args.folder)
        mapping_path = folder / "matched_files.json"
        matched_dir = folder / "matched"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing mapping file: {mapping_path}")
        viewer = ViewerWindow(mapping_path, matched_dir)
        return viewer.run()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
