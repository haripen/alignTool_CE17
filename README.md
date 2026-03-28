# Sync Alignment Review

PySide6 workflow for matching `.otb4`, `.c3d`, and `.tsv` recordings by shared sync and raw CoP agreement, then reviewing each proposed match manually.

## Repo Status

- The repo is intended to be self-contained for matching, review, and MAT export.
- A vendored `otb_4_file_io.py` is included under `vendor/hdsemg_shared/fileio/`.
- That vendored OTB4 reader is a local derivative for this project. It is not treated as upstream-compatible with the `hdsemg-pipe` workflow or with the original `hdsemg-shared` package.
- For now, changes to the vendored OTB4 reader should be treated as local-only unless they are explicitly revalidated against the other repositories.

## What It Does

- Scans a selected source directory recursively.
- Ignores `matched` and `accepted` subfolders during scanning.
- Extracts sync from:
  - OTB4: `Syncstation AUX 2 [V]`
  - C3D: `Voltage.2_Sync`
  - TSV: `2_Sync [Volt] [sync]`
- Verifies raw agreement using exact same-signal CoP pairs:
  - TSV `.../CoP/Cx [Meter] [raw]` <-> C3D `copx`
  - TSV `.../CoP/Cy [Meter] [raw]` <-> C3D `copy`
- Exports time-shift metadata for later inner-merge alignment of all channels.
- Opens a review UI where every proposed match must be confirmed as accepted or rejected.
- Defers copying files into `matched/` until the user has entered review input.
- Supports an `Export accepted to .mat` button that writes one aligned MAT file per accepted triplet from the original files.
- Adds a `Parallel export` checkbox and worker selector for MAT export.
- Shows progress bars while loading a selected match and while exporting MAT files.
- Shows the common matched time window in the plots as a very light green band with dashed light-green boundaries.

## Install

Use the provided conda environment file:

```powershell
conda env create -f sync_alignment_env.yml
conda activate sync-alignment
```

## Run

From the repository folder:

```powershell
python match_files.py
```

Workflow:

1. Select the directory to scan.
2. Select the export directory.
3. The tool scans for `.otb4`, `.c3d`, and `.tsv` files recursively.
4. The review UI opens automatically.
5. For each proposed triplet, confirm `Accept -> Next` or `Reject -> Next`.
6. Use `Export accepted to .mat` to write one aligned MAT file per accepted triplet into `matched/`.
   By default `Parallel export` is enabled and `n_workers=6`.
   MAT export is enabled only after the review is complete.
7. When all matches are reviewed, the tool shows a completion message.

If you try to close the UI before reviewing all matches, the tool warns you.

## Review Rules

- Automatic `accept` is suggested when the raw C3D/TSV match is excellent and the OTB4/C3D dedicated sync is still strong, even if the TSV dedicated sync is weak.
- User review can overwrite the automatic decision.
- The JSON stores both the automatic decision and the user-reviewed final decision.
- Review colors:
  - light green: auto-accept
  - dark green: user-accept
  - light red: auto-reject
  - dark red: user-reject
- Missing TSV dedicated sync is not by itself an auto-reject reason.

## Outputs

The selected export directory receives:

- `matched_files.json`
  - All reviewed candidate triplets
  - Sync/raw quality metrics
  - User review state
  - `inner_merge` alignment metadata for later channel-level alignment
- `accepted_matches.json`
  - Only final accepted matches
- `alignment.log`
  - Detailed extraction, matching, and diagnostic log
- `matched/`
  - Copied files for only final accepted triplets
  - Created only after the full review is complete
- `<C3DName>_YYYYMMDD_HHMMSS.mat`
  - Aligned per-match MAT export created inside `matched/` by the `Export accepted to .mat` button
- `<C3DName>_YYYYMMDD_HHMMSSmergeInfo.txt`
  - Short text summary written next to each MAT file
- `<C3DName>_YYYYMMDD_HHMMSS_4pipe.mat`
  - Additional hdsemg-pipe-compatible MAT export
- `<C3DName>_YYYYMMDD_HHMMSS_4pipe.json`
  - Mean-subtraction metadata for the `_4pipe.mat` file

The main JSON stores both the original filenames and the planned renamed filenames for `matched/`.

## Inner-Merge Alignment Metadata

Each match includes:

- `plot_time_shifts_sec`
- `inner_merge.formula`
- `inner_merge.source_time_ranges`
- `inner_merge.aligned_sync_edges_sec`
- `inner_merge.inner_merge_start_sec`
- `inner_merge.inner_merge_end_sec`
- `inner_merge.inner_merge_duration_sec`

Use:

```text
aligned_time_sec = source_time_sec + shift_sec
```

to align any channel from OTB4, C3D, or TSV onto the common time axis.

## Notes

- The repo includes a vendored copy of `otb_4_file_io.py` under `vendor/` so the review tool can follow OTB4 metadata locally.
- The vendored `otb_4_file_io.py` is a local derivative. It should not currently be assumed to be drop-in compatible with `hdsemg-pipe`, `hdsemg-shared`, or other downstream tools unless revalidated there.
- Review and export logic now assumes:
  - generalized OTB4/C3D sync repair is the default path
  - late-start C3D is a special case, detected from strong TSV/C3D raw agreement plus a large negative raw lag
  - per-device OTB4 gap zones are transferred into both the aligned base `.mat` and `_4pipe.mat`
