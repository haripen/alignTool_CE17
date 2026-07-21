from __future__ import annotations

import os
import sys
import json
from pathlib import Path

from sync_alignment_tool import (
    ViewerWindow,
    run_scan,
    _iter_source_files,
    _load_otb4_track_data,
    _infer_pulse_duration_ms,
    _is_counter_track,
    _detect_buffer_events_abs,
    _fit_descending_sync_pattern,
    _discover_sync_channel_candidates,
    _resolve_sync_channel_roles,
    _available_global_pair_options,
    _apply_tracebio_paths_root_override,
    TRACEBIO_PATHS_ROOT_DEFAULT,
    _otb4_autonumber_plan,
    _apply_otb4_autonumber,
)


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


def _infer_plausible_sync_pulse_durations(scan_dir: Path) -> dict:
    allowed = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    candidates: set[float] = set()
    notes: list[str] = []
    otb4_paths = _iter_source_files(scan_dir, ".otb4", excluded_dir_names=("matched", "accepted", "__pycache__", ".git"))
    for path in otb4_paths[:3]:
        try:
            found_any = False
            for _offset, track, raw_arr in _load_otb4_track_data(path):
                if int(track.get("NumberOfChannels", 0) or 0) != 1:
                    continue
                device = str(track.get("Device", "")).strip()
                subtitle = str(track.get("SubTitle", "")).strip()
                raw = raw_arr[0]
                is_ramp = _is_counter_track(track, raw)
                is_syncstation_sync = device == "Syncstation" and (subtitle == "AUX 2" or (bool(track.get("IsControl")) and not subtitle and not is_ramp))
                if not is_syncstation_sync:
                    continue
                fs = float(track.get("SamplingFrequency") or 0.0)
                info = _infer_pulse_duration_ms(raw, fs)
                for value in info.get("candidate_durations_ms") or []:
                    candidates.add(float(value))
                notes.append(
                    f"{path.name} {subtitle or 'Control'}: sampled width {float(info.get('observed_width_ms_mode') or 0.0):.3f} ms at {fs:.0f} Hz"
                )
                found_any = True
            if found_any and 1.0 in candidates:
                break
        except Exception:
            continue
    plausible = [value for value in allowed if value in candidates]
    if not plausible:
        plausible = [1.0]
        notes.append("No reliable Syncstation pulse-width inference found; using the SyncMini default as the only suggested option.")
    recommended = 1.0 if 1.0 in plausible else plausible[0]
    return {
        "plausible_durations_ms": plausible,
        "recommended_duration_ms": recommended,
        "note": "\n".join(notes) if notes else "Syncstation pulse-width inference unavailable.",
    }


def _infer_plausible_sync_decrements(scan_dir: Path) -> dict:
    allowed_ms = [50.0, 100.0]
    score_rows: dict[float, list[tuple[bool, float, float]]] = {value: [] for value in allowed_ms}
    notes: list[str] = []
    otb4_paths = _iter_source_files(scan_dir, ".otb4", excluded_dir_names=("matched", "accepted", "__pycache__", ".git"))
    for path in otb4_paths[:3]:
        try:
            for _offset, track, raw_arr in _load_otb4_track_data(path):
                if int(track.get("NumberOfChannels", 0) or 0) != 1:
                    continue
                device = str(track.get("Device", "")).strip()
                subtitle = str(track.get("SubTitle", "")).strip()
                raw = raw_arr[0]
                is_ramp = _is_counter_track(track, raw)
                is_syncstation_sync = device == "Syncstation" and (subtitle == "AUX 2" or (bool(track.get("IsControl")) and not subtitle and not is_ramp))
                if not is_syncstation_sync:
                    continue
                fs = float(track.get("SamplingFrequency") or 0.0)
                if fs <= 0.0:
                    continue
                events = _detect_buffer_events_abs(raw, fs)
                event_times_sec = (events / fs).tolist()
                if len(event_times_sec) < 4:
                    continue
                per_track_bits = []
                for value_ms in allowed_ms:
                    fit = _fit_descending_sync_pattern(
                        event_times_sec,
                        default_step_sec=float(value_ms) / 1000.0,
                        candidate_step_secs=[float(value_ms) / 1000.0],
                    )
                    mae_ms = float(fit.get("gap_mae_ms") or float("inf"))
                    max_ms = float(fit.get("max_gap_error_ms") or float("inf"))
                    observed = bool(fit.get("pattern_observed"))
                    score_rows[float(value_ms)].append((observed, mae_ms, max_ms))
                    per_track_bits.append(f"{int(value_ms)} ms: mae={mae_ms:.1f}, max={max_ms:.1f}")
                notes.append(f"{path.name} {subtitle or 'Control'}: " + " | ".join(per_track_bits))
        except Exception:
            continue
    plausible: list[float] = []
    ranked: list[tuple[tuple[int, float, float, float], float]] = []
    for value_ms in allowed_ms:
        rows = score_rows.get(float(value_ms)) or []
        if not rows:
            continue
        observed_count = sum(1 for observed, _mae, _max in rows if observed)
        mean_mae = sum(mae for _observed, mae, _max in rows) / len(rows)
        mean_max = sum(max_ms for _observed, _mae, max_ms in rows) / len(rows)
        if observed_count > 0 or (mean_mae <= 25.0 and mean_max <= 60.0):
            plausible.append(float(value_ms))
        ranked.append(((0 if observed_count > 0 else 1, mean_mae, mean_max, abs(float(value_ms) - 100.0)), float(value_ms)))
    if not ranked:
        plausible = [100.0]
        recommended = 100.0
        notes.append("No reliable decrement inference found; using the SyncMini default decrement as the only suggested option.")
    else:
        ranked.sort(key=lambda item: item[0])
        recommended = ranked[0][1]
        if not plausible:
            plausible = [float(recommended)]
    return {
        "plausible_decrements_ms": plausible,
        "recommended_decrement_ms": float(recommended),
        "note": "\n".join(notes) if notes else "Sync decrement inference unavailable.",
    }


def _pick_item(parent, title: str, prompt: str, values: list[float], recommended: float) -> float | None:
    from PySide6 import QtWidgets

    labels: list[str] = []
    label_to_value: dict[str, float] = {}
    for value in values:
        label = f"{value:g} ms"
        if abs(float(value) - float(recommended)) < 1e-9:
            label += " (Recommended)"
        labels.append(label)
        label_to_value[label] = float(value)
    current_index = 0
    for idx, label in enumerate(labels):
        if "(Recommended)" in label:
            current_index = idx
            break
    choice, ok = QtWidgets.QInputDialog.getItem(parent, title, prompt, labels, current_index, False)
    if not ok or not choice:
        return None
    return float(label_to_value[choice])


_SKIP_CHANNEL_LABEL = "None -- skip (no dedicated sync; matcher falls back to its other methods)"


def _pick_from_choices(parent, title: str, prompt: str, labels: list[str], current_index: int) -> tuple[int, bool]:
    from PySide6 import QtWidgets

    if not labels:
        return -1, True
    choice, ok = QtWidgets.QInputDialog.getItem(parent, title, prompt, labels, max(0, current_index), False)
    if not ok or not choice:
        return -1, False
    try:
        return labels.index(choice), True
    except ValueError:
        return -1, False


def _format_pair_candidate_label(pair: dict) -> str:
    otb4 = pair.get("otb4") or {}
    c3d = pair.get("c3d") or {}
    otb4_label = f"{otb4.get('device', '?')} {otb4.get('subtitle', '')}".strip()
    bits = []
    width = c3d.get("pulse_width_ms")
    if width is not None:
        bits.append(f"{width:g} ms pulses")
    decrement = c3d.get("decrement_ms")
    if decrement is not None:
        bits.append(f"{decrement:g} ms decrement")
    conf = c3d.get("confidence")
    if conf is not None:
        bits.append(f"confidence {conf * 100:.0f}%")
    suffix = f" ({', '.join(bits)})" if bits else ""
    return f"OTB4 {otb4_label} <-> C3D {c3d.get('label', '?')}{suffix}"


def _format_bridge_candidate_label(pair: dict) -> str:
    base = _format_pair_candidate_label(pair)
    tsv = pair.get("tsv")
    if not tsv:
        return base + " | no matching TSV column found"
    decrement = tsv.get("decrement_ms")
    detail = f" ({decrement:g} ms decrement)" if isinstance(decrement, (int, float)) else ""
    return base + f" | TSV column found{detail}"


def _pick_sync_channel_map(scan_dir: Path, parent) -> dict | None:
    """Auto-scan (via _discover_sync_channel_candidates/_resolve_sync_channel_roles
    -- pulse-shape based, not name based) and confirm the two sync-channel
    roles the matcher needs: the OTB4<->C3D dedicated sync channel, and the
    TSV<->C3D/OTB4 bridge channel. Mirrors the existing pulse-duration/
    decrement picker pattern (_pick_item / QInputDialog.getItem) already
    used below, so overriding a device naming change no longer requires a
    code change -- just a different pick in this dialog.
    """
    if not any(_iter_source_files(scan_dir, ".otb4", excluded_dir_names=("matched", "accepted", "__pycache__", ".git"))):
        return {"otb4_c3d_sync": None, "tsv_bridge_sync": None}

    discovery = _discover_sync_channel_candidates(scan_dir, sample_limit=3)
    roles = _resolve_sync_channel_roles(discovery)
    paired = roles.get("all_paired_candidates") or []

    dedicated_labels = [_format_pair_candidate_label(p) for p in paired] + [_SKIP_CHANNEL_LABEL]
    recommended = roles.get("otb4_c3d_sync")
    dedicated_idx = paired.index(recommended) if recommended in paired else (len(dedicated_labels) - 1)
    if 0 <= dedicated_idx < len(paired):
        dedicated_labels[dedicated_idx] += " (Recommended)"
    idx, ok = _pick_from_choices(
        parent,
        "OTB4 / C3D Dedicated Sync Channel",
        "Which channel pair is the dedicated sync signal shared by OTB4 and C3D?\n\n"
        "Detected automatically from pulse-pattern shape (not by channel name) using a sample of "
        "your files. Pick a different pair if this looks wrong, or Skip if this session has no "
        "usable dedicated sync channel.",
        dedicated_labels,
        dedicated_idx,
    )
    if not ok:
        return None
    chosen_dedicated = paired[idx] if 0 <= idx < len(paired) else None

    result: dict = {
        "otb4_c3d_sync": (
            {
                "otb4_device": chosen_dedicated["otb4"]["device"],
                "otb4_subtitle": chosen_dedicated["otb4"]["subtitle"],
                "c3d_label": chosen_dedicated["c3d"]["label"],
            }
            if chosen_dedicated
            else None
        )
    }

    if not any(_iter_source_files(scan_dir, ".tsv", excluded_dir_names=("matched", "accepted", "__pycache__", ".git"))):
        result["tsv_bridge_sync"] = None
        return result

    bridge_labels = [_format_bridge_candidate_label(p) for p in paired] + [_SKIP_CHANNEL_LABEL]
    recommended_bridge = roles.get("tsv_bridge_sync")
    bridge_idx = len(bridge_labels) - 1
    if recommended_bridge is not None:
        for i, p in enumerate(paired):
            if p["c3d"]["label"] == recommended_bridge["c3d"]["label"]:
                bridge_idx = i
                break
    if 0 <= bridge_idx < len(paired):
        bridge_labels[bridge_idx] += " (Recommended)"
    idx, ok = _pick_from_choices(
        parent,
        "TSV Bridge Sync Channel",
        "Which channel bridges the TSV (biofeedback) recording to C3D/OTB4?\n\n"
        "This is separate from the OTB4/C3D dedicated sync above. If none is detected, or you'd "
        "rather rely on raw CoP cross-correlation (the long-standing default), choose Skip.",
        bridge_labels,
        bridge_idx,
    )
    if not ok:
        return None
    chosen_bridge = paired[idx] if 0 <= idx < len(paired) else None
    result["tsv_bridge_sync"] = (
        {
            "otb4_device": chosen_bridge["otb4"]["device"],
            "otb4_subtitle": chosen_bridge["otb4"]["subtitle"],
            "c3d_label": chosen_bridge["c3d"]["label"],
        }
        if chosen_bridge
        else None
    )
    return result


def _pick_preferred_pairs(scan_dir: Path, parent) -> list | None:
    """Let the user override which raw C3D/TSV channel pair is preferred for
    cross-correlation matching, or accept auto-detection (the default).
    Wires up `_available_global_pair_options`, which already existed for
    exactly this purpose but was never called from the startup dialog --
    `preferred_channel_pairs` was always hardcoded to `[]`.
    """
    options = [o for o in _available_global_pair_options(scan_dir) if o.get("kind") in ("auto", "raw")]
    if len(options) <= 1:
        return []
    labels = [opt["label"] + (" (Recommended)" if opt.get("kind") == "auto" else "") for opt in options]
    idx, ok = _pick_from_choices(
        parent,
        "Preferred Raw Channel Pair",
        "Which C3D/TSV raw channel pair should be preferred for cross-correlation matching?\n\n"
        "Auto detect (the default) tries the exact CoP Cx<->copx / Cy<->copy pairing automatically. "
        "Only override this if a specific pairing needs to be forced.",
        labels,
        0,
    )
    if not ok:
        return None
    chosen = options[idx]
    return [chosen] if chosen.get("kind") == "raw" else []


def _pick_tracebio_paths_root(parent) -> str | None:
    """Confirm the shared traceBio corridor/path-file folder once at startup
    instead of only discovering it's missing deep into MAT export or
    per-match review (where it previously showed up as a reactive per-file
    browse dialog). Silent when the default network location is reachable;
    only prompts when it isn't.
    """
    from PySide6 import QtWidgets

    default_root = Path(TRACEBIO_PATHS_ROOT_DEFAULT)
    if default_root.exists() and default_root.is_dir():
        return str(default_root)

    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    msg.setWindowTitle("traceBio Paths Folder Not Found")
    msg.setText(f"The shared traceBio path-file folder was not found at:\n{default_root}")
    msg.setInformativeText(
        "This folder holds the traceBio corridor/path TSV files referenced by biofeedback "
        "recordings' settings JSON (e.g. via a network share that may not be mapped right now). "
        "Browse to the correct location, or Skip to be asked again per-file later only if a "
        "specific path file turns out to be missing."
    )
    browse_btn = msg.addButton("Browse...", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    msg.addButton("Skip", QtWidgets.QMessageBox.ButtonRole.RejectRole)
    msg.exec()
    if msg.clickedButton() is not browse_btn:
        return None
    start_dir = default_root.parent if default_root.parent.exists() else Path.home()
    chosen = QtWidgets.QFileDialog.getExistingDirectory(
        parent,
        "Select traceBio Paths Folder",
        str(start_dir),
        QtWidgets.QFileDialog.Option.ShowDirsOnly,
    )
    return chosen or None


def _pick_matching_preferences(scan_dir: Path, parent) -> dict | None:
    tsv_present = any(_iter_source_files(scan_dir, ".tsv", excluded_dir_names=("matched", "accepted", "__pycache__", ".git")))
    tracebio_paths_root = _pick_tracebio_paths_root(parent) if tsv_present else None

    if not any(_iter_source_files(scan_dir, ".otb4", excluded_dir_names=("matched", "accepted", "__pycache__", ".git"))):
        return {
            "preferred_channel_pairs": [],
            "sync_channel_map": {"otb4_c3d_sync": None, "tsv_bridge_sync": None},
            "sync_pulse_duration_ms": None,
            "plausible_sync_pulse_durations_ms": [],
            "sync_decrement_ms": None,
            "plausible_sync_decrements_ms": [],
            "tracebio_paths_root": tracebio_paths_root,
        }

    sync_channel_map = _pick_sync_channel_map(scan_dir, parent)
    if sync_channel_map is None:
        return None

    pulse_meta = _infer_plausible_sync_pulse_durations(scan_dir)
    plausible = [float(v) for v in (pulse_meta.get("plausible_durations_ms") or [1.0])]
    recommended = float(pulse_meta.get("recommended_duration_ms") or plausible[0])
    selected_pulse_ms = _pick_item(
        parent,
        "Sync Pulse Duration",
        "Which SyncMini / Syncstation TTL duration was configured?\n\n"
        + "SyncMini default: 1 ms.\n"
        + str(pulse_meta.get("note") or "")
        + "\n\nOnly durations plausible with the sampled data are shown.",
        plausible,
        recommended,
    )
    if selected_pulse_ms is None:
        return None

    decrement_meta = _infer_plausible_sync_decrements(scan_dir)
    plausible_decrements = [float(v) for v in (decrement_meta.get("plausible_decrements_ms") or [100.0])]
    recommended_decrement = float(decrement_meta.get("recommended_decrement_ms") or plausible_decrements[0])
    selected_decrement_ms = _pick_item(
        parent,
        "Sync Decrement",
        "Which decrement between successive Sync events was configured?\n\n"
        + "SyncMini default: 100 ms decrement.\n"
        + str(decrement_meta.get("note") or "")
        + "\n\nOnly decrements plausible with the sampled data are shown.",
        plausible_decrements,
        recommended_decrement,
    )
    if selected_decrement_ms is None:
        return None

    preferred_pairs = _pick_preferred_pairs(scan_dir, parent)
    if preferred_pairs is None:
        return None

    return {
        "preferred_channel_pairs": preferred_pairs,
        "sync_channel_map": sync_channel_map,
        "sync_pulse_duration_ms": float(selected_pulse_ms),
        "plausible_sync_pulse_durations_ms": plausible,
        "sync_decrement_ms": float(selected_decrement_ms),
        "plausible_sync_decrements_ms": plausible_decrements,
        "tracebio_paths_root": tracebio_paths_root,
    }


def _max_source_mtime(scan_dir: Path) -> float:
    latest = 0.0
    excluded = {"matched", "accepted", "__pycache__", ".git"}
    for root, dirs, files in os.walk(scan_dir):
        dirs[:] = [d for d in dirs if d.lower() not in excluded]
        for name in files:
            if not name.lower().endswith((".otb4", ".c3d", ".tsv")):
                continue
            path = Path(root) / name
            try:
                latest = max(latest, float(path.stat().st_mtime))
            except Exception:
                continue
    return latest


def _reuse_existing_scan(scan_dir: Path, export_dir: Path, match_options: dict | None = None) -> tuple[Path, Path, Path] | None:
    mapping_path = export_dir / "matched_files.json"
    log_path = export_dir / "alignment.log"
    matched_dir = export_dir / "matched"
    if not mapping_path.exists():
        return None
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    source_root = Path(str(mapping.get("source_root") or "")).resolve()
    if source_root != scan_dir.resolve():
        return None
    existing_options = mapping.get("match_options") or {}
    wanted_pulse = float((match_options or {}).get("sync_pulse_duration_ms") or 0.0)
    existing_pulse = float(existing_options.get("sync_pulse_duration_ms") or 0.0)
    if abs(wanted_pulse - existing_pulse) > 1e-9:
        return None
    wanted_step = float((match_options or {}).get("sync_decrement_ms") or 0.0)
    existing_step = float(existing_options.get("sync_decrement_ms") or 0.0)
    if abs(wanted_step - existing_step) > 1e-9:
        return None
    try:
        mapping_mtime = float(mapping_path.stat().st_mtime)
    except Exception:
        return None
    if _max_source_mtime(scan_dir) > mapping_mtime:
        return None
    return mapping_path, log_path, matched_dir


def _pick_otb4_autonumber(scan_dir: Path, parent) -> None:
    """If any OTB4 files in scan_dir still use the old unnumbered
    "File__TIMESTAMP..." naming, offer to renumber them chronologically
    before scanning. Purely cosmetic -- the matcher never depends on
    filename numbers -- but makes the file list easier to cross-reference
    by eye against C3D "Setting_NN"/TSV "File_0NN" filenames. Silent (no
    dialog) if none are found.
    """
    from PySide6 import QtWidgets

    otb4_paths = _iter_source_files(scan_dir, ".otb4", excluded_dir_names=("matched", "accepted", "__pycache__", ".git"))
    plan = _otb4_autonumber_plan(otb4_paths)
    if not plan:
        return
    first_old, first_new = plan[0]
    preview = f"{first_old.name}\n  -> {first_new.name}"
    if len(plan) > 1:
        last_old, last_new = plan[-1]
        preview += f"\n...\n{last_old.name}\n  -> {last_new.name}"
    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
    msg.setWindowTitle("Renumber OTB4 Files")
    msg.setText(f"{len(plan)} OTB4 file(s) use the old unnumbered \"File__TIMESTAMP...\" naming.")
    msg.setInformativeText(
        "Renumber them chronologically by capture timestamp (File_01_..., File_02_..., ...)?\n\n"
        f"{preview}\n\n"
        "This only renames the files on disk -- it does not change how they are matched "
        "(matching never depends on filename numbers)."
    )
    yes_btn = msg.addButton("Renumber", QtWidgets.QMessageBox.ButtonRole.YesRole)
    msg.addButton("Leave As-Is", QtWidgets.QMessageBox.ButtonRole.NoRole)
    msg.exec()
    if msg.clickedButton() is not yes_btn:
        return
    results = _apply_otb4_autonumber(plan)
    QtWidgets.QMessageBox.information(
        parent,
        "OTB4 Files Renumbered",
        "\n".join(results) if results else "No files were renamed.",
    )


_DEFAULT_SCAN_ROOT = Path(r"U:\Daten_speziell\Vicon\Eclipse Datenbanken\Projekte\ASF-PPPD")


def _pick_directories() -> tuple[Path, Path, dict] | tuple[None, None, None]:
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    parent = QtWidgets.QWidget()
    parent.setWindowTitle("Match Files")

    start_dir = _DEFAULT_SCAN_ROOT if _DEFAULT_SCAN_ROOT.exists() else Path("C:/")
    scan_dir = QtWidgets.QFileDialog.getExistingDirectory(
        parent,
        "Select Directory To Scan",
        str(start_dir),
        QtWidgets.QFileDialog.Option.ShowDirsOnly,
    )
    if not scan_dir:
        return None, None, None
    _pick_otb4_autonumber(Path(scan_dir), parent)

    export_dir = QtWidgets.QFileDialog.getExistingDirectory(
        parent,
        "Select Export Directory",
        scan_dir,
        QtWidgets.QFileDialog.Option.ShowDirsOnly,
    )
    if not export_dir:
        return None, None, None
    match_options = _pick_matching_preferences(Path(scan_dir), parent)
    if match_options is None:
        return None, None, None
    return Path(scan_dir), Path(export_dir), match_options


def main() -> int:
    _configure_qt_runtime()
    scan_dir, export_dir, match_options = _pick_directories()
    if scan_dir is None or export_dir is None:
        print("Cancelled.")
        return 1
    # Applies regardless of whether the scan below is reused from cache or run
    # fresh -- a reused scan skips analyze_folder (where this would otherwise
    # also get applied), but traceBio export/review-time file resolution
    # still needs it in this process.
    _apply_tracebio_paths_root_override(match_options)

    reused = _reuse_existing_scan(scan_dir, export_dir, match_options=match_options)
    if reused is not None:
        mapping_path, log_path, matched_dir = reused
        print(f"Reusing existing scan from {mapping_path}")
    else:
        mapping_path, log_path, matched_dir = run_scan(scan_dir, export_dir=export_dir, match_options=match_options)
    viewer = ViewerWindow(mapping_path, matched_dir)
    rc = viewer.run()

    accepted_json = export_dir / "accepted_matches.json"
    print(f"Scan root: {scan_dir}")
    print(f"Export root: {export_dir}")
    print(f"Mapping JSON: {mapping_path}")
    print(f"Log: {log_path}")
    print(f"Matched files: {matched_dir} (only accepted files are written here after user review input)")
    print(f"Accepted JSON: {accepted_json}")
    print("Use matched_files.json for all reviewed candidates, accepted_matches.json for final accepted matches, and the UI Export accepted to .mat button to write short-named MAT files plus mergeInfo text files into matched/.")
    print("The review UI now also includes OTB4 peripheral probe sync diagnostics based on the descending-interval sync rule, plus extra non-sync probe detail rows with buffer/ramp plots.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
