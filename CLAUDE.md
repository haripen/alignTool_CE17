# CLAUDE.md

## Project overview

Sync Alignment Review — a PySide6 desktop tool that matches `.otb4` (OT Bioelettronica EMG), `.c3d` (Vicon motion capture), and `.tsv` (biofeedback) recordings by shared sync pulses and raw CoP agreement, then walks a reviewer through manually accepting or rejecting each proposed match before exporting aligned `.mat` files.

Full user-facing workflow, scan/output details, and review rules: see `README.md`. Full-resolution traceBio MAT export spec: `MAT_EXPORT_TRACEBIO_SPEC.md`.

## Environment

Conda only — no `requirements.txt`/`pyproject.toml`.

```powershell
conda env create -f sync_alignment_env.yml
conda activate sync-alignment
```

`sync_alignment_env.yml`: python 3.11, numpy, pandas, scipy, pyside6, pyqtgraph, ezc3d (conda-forge).

## Entry points / key files

- `match_files.py` — main entry point (`python match_files.py`).
- `sync_alignment_tool.py` — core matching, repair, review UI, and export logic (large, monolithic).
- `otb4_gap_repair.py` — standalone OTB4 gap-repair diagnostics, derived from the original MATLAB package logic.
- `analyze_otb4_buffers.py` — OTB4 buffer analysis utility.
- `vendor/hdsemg_shared/fileio/otb_4_file_io.py` — vendored OTB4 reader (local derivative of [`hdsemg-shared`](https://github.com/johanneskasser/hdsemg-shared)'s `file_io.py`). **Not** upstream-safe or assumed drop-in compatible with `hdsemg-pipe`/`hdsemg-shared` — treat changes here as local-only unless explicitly revalidated against those repos.
- `tools/migrate_sidecar_v2_reexport.py` — one-off batch migration script that upgrades traceBio sidecar JSON settings to `PerformedTransformVersion 2` and re-exports/validates `.mat` files against a project root of `accepted_matches.json` files. Reuses internal helpers from `sync_alignment_tool.py`.

## Key alignment rules

- Default OTB4/C3D alignment uses dedicated sync pulses, with OTB4 gap repair from SyncStation/device ramp zones; cumulative repair combinations are allowed when they improve OTB4/C3D sync.
- Late-start C3D special case (`late_c3d_raw_bridge`): only applied when TSV/C3D raw correlation is very strong and raw lag is strongly negative. Skips early OTB4 sync edges, not C3D edges.
- TSV/C3D raw matching uses exact same-signal CoP pairs: `Cx <-> copx`, `Cy <-> copy`.
- Sync channels expected: OTB4 `Syncstation AUX 2 [V]`, C3D `Voltage.2_Sync`, TSV `2_Sync [Volt] [sync]`. TSV dedicated sync is diagnostic only, not an auto-reject reason.

## MAT export rules

- Base `.mat`: aligned-only export, one shared time axis, common matched time window only, OTB4 gaps inserted before export.
- `_4pipe.mat` (per accepted triplet, `hdsemg-pipe`-compatible but not yet fully tested against it): EMG channels de-meaned, TSV/C3D aligned tracks added at the OTB4/EMG sample rate, OTB4 gap insertion applied per device label group. Companion `_4pipe.json` carries mean-subtraction metadata.

## Known constraints

- The repo has project-specific assumptions baked in about labels and signal names.
- The vendored OTB4 reader is local-only until validated elsewhere.
- Review UI and exports are designed around accepted triplets; pair-only review entries are still possible.
- Real capture data (Vicon/OTB4/biofeedback sessions) lives on the network share (e.g. `U:\...\ASF-PPPD\...`), never inside this repo. Repo-local `tmp_*` scratch/export folders should not be committed — see `.gitignore`.

## Sanity checks before relying on a change

1. `python -m py_compile sync_alignment_tool.py match_files.py otb4_gap_repair.py`
2. Run `python match_files.py` against a known test folder and confirm review/export still behaves as expected (e.g. late-start C3D cases route through `late_c3d_raw_bridge`, generally-repaired OTB4/C3D cases stay in the general repair path, `_4pipe.mat` still contains device-repaired OTB4 channels and de-meaned EMG).
3. If touching OTB4 reading logic, re-check compatibility separately against `hdsemg-pipe` before considering it for upstreaming.
