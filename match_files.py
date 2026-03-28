from __future__ import annotations

import sys
from pathlib import Path

from sync_alignment_tool import ViewerWindow, run_scan


def _pick_directories() -> tuple[Path, Path] | tuple[None, None]:
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    parent = QtWidgets.QWidget()
    parent.setWindowTitle("Match Files")

    scan_dir = QtWidgets.QFileDialog.getExistingDirectory(
        parent,
        "Select Directory To Scan",
        str(Path.cwd()),
        QtWidgets.QFileDialog.Option.ShowDirsOnly,
    )
    if not scan_dir:
        return None, None

    export_dir = QtWidgets.QFileDialog.getExistingDirectory(
        parent,
        "Select Export Directory",
        scan_dir,
        QtWidgets.QFileDialog.Option.ShowDirsOnly,
    )
    if not export_dir:
        return None, None
    return Path(scan_dir), Path(export_dir)


def main() -> int:
    scan_dir, export_dir = _pick_directories()
    if scan_dir is None or export_dir is None:
        print("Cancelled.")
        return 1

    mapping_path, log_path, matched_dir = run_scan(scan_dir, export_dir=export_dir)
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
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
