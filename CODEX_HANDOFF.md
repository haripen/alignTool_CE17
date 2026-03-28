# Codex Handoff

## Current State

- Main entry points:
  - `match_files.py`
  - `sync_alignment_tool.py`
- Portable OTB4 access is provided through:
  - `vendor/hdsemg_shared/fileio/otb_4_file_io.py`
- The vendored OTB4 reader is a local derivative for this repo.
  It is not yet considered upstream-safe for `hdsemg-pipe` or the original `hdsemg-shared` repositories.

## Key Alignment Rules

- Default OTB4/C3D alignment:
  - use dedicated sync pulses
  - repair OTB4 from SyncStation / device ramp-derived gap zones
  - allow cumulative repair combinations when they improve OTB4/C3D sync
- Late-start C3D special case:
  - only considered when TSV/C3D raw correlation is extremely strong and raw lag is strongly negative
  - implemented as `late_c3d_raw_bridge`
  - skips early OTB4 sync edges, not C3D edges
- TSV/C3D raw matching:
  - exact same-signal CoP matching
  - `Cx <-> copx`
  - `Cy <-> copy`

## MAT Export Rules

- Base `.mat`:
  - aligned-only export
  - one shared time axis
  - common matched time window only
  - OTB4 gaps inserted before export
- `_4pipe.mat`:
  - per accepted triplet
  - EMG channels are de-meaned
  - TSV and C3D aligned tracks are added at the OTB4/EMG sample rate
  - OTB4 gap insertion is applied per device label group

## Known Important Files

- `sync_alignment_tool.py`
  Main matching, repair, review UI, export logic.
- `otb4_gap_repair.py`
  Standalone repair diagnostics derived from the MATLAB package logic.
- `README.md`
  User-facing workflow.

## Known Constraints

- The repo still contains project-specific assumptions about labels and signal names.
- The vendored OTB4 reader should be treated as local-only until validated elsewhere.
- The review UI and exports are designed around accepted triplets, with pair-only review entries still possible.

## Suggested First Checks In A Later Session

1. Re-run `python -m py_compile sync_alignment_tool.py match_files.py`.
2. Run `python match_files.py` on a known CE17 folder.
3. Verify:
   - `File_02` is handled as `late_c3d_raw_bridge`
   - repaired OTB4/C3D cases like `File_13` stay in the general repair path
   - `_4pipe.mat` still contains device-repaired OTB4 channels and de-meaned EMG
4. If touching OTB4 reading logic, re-check compatibility separately against `hdsemg-pipe` before upstreaming anything.
