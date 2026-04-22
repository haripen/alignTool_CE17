# MAT Export Spec For Full-Resolution traceBio Paths

This note defines what the aligned MAT export in this repository must write so that the exported MAT files contain:

- the correct **non-smoothed performed path in percent**
- the correct **desired**, **desired upper**, and **desired lower** paths
- all four traces at **full sample resolution**
- all four traces on a **correctly aligned common time axis**
- the desired-path signals with the **true path edges**, not low-rate fitting artifacts

This is an implementation spec for `sync_alignment_tool.py`.

## Problem To Fix

Current export behavior is not sufficient for this use case:

- `export_match_mat(...)` currently writes:
  - `c3d_analog`
  - `c3d_point`
  - `c3d_cop`
  - `tsv`
- the exported `tsv` struct is only the original TSV-rate data from `_build_tsv_mat_struct(...)`
- the helper `_synth_tsv_performed_values(...)` is a fitted model and must **not** be used as the source of truth for exported performed-path data
- the current common aligned bundle uses `_resample_to_target_time(...)`, which is appropriate for continuous signals, but **not** sufficient for the desired path traces if exact path edges must be preserved
- the `_4pipe.mat` export must also be updated, because its current path channels still reflect the old interpolation-based behavior

## Required New Output

Two export targets must be updated:

1. the existing `_4pipe.mat`
2. the existing base aligned `.mat`

### `_4pipe.mat` requirement

The `_4pipe.mat` export must be updated so that the current path versions are written at full `2000 Hz` resolution.

This includes:

- `desired`
- `desired_upper`
- `desired_lower`
- the correctly reconstructed `performed` path

Important:

- the `performed` path in `_4pipe.mat` must no longer be the old interpolated or fitted version
- it must be the authoritative reconstructed path derived from the high-rate source, as defined below
- the desired-path channels in `_4pipe.mat` must also be full-rate and edge-exact

### Base `.mat` requirement

The base aligned `.mat` must gain an additional top-level dict, for example:

- `tsv_reconstructed`

It must contain at minimum:

- `time`
- `frames`
- `sample_rate`
- `performed_percent_unsmoothed`
- `desired`
- `desired_upper`
- `desired_lower`
- `raw_m`
- `offset_m`
- `zero_offset_m`
- `record_min_m`
- `record_max_m`
- `selected_signal`
- `selected_path`
- `corridor_half_width`
- `relative_mode`
- `smoothing_method`
- `smoothing_frames`
- `label_map`

The four main exported traces

- `performed_percent_unsmoothed`
- `desired`
- `desired_upper`
- `desired_lower`

must all be written on the same full-rate aligned time base.

In other words:

- `_4pipe.mat` must carry the corrected full-rate path channels for downstream processing
- the base `.mat` must expose the same information explicitly in a dedicated top-level reconstruction dict such as `tsv_reconstructed`

## Canonical Time Base

Use the aligned C3D analog time base as the canonical full-resolution export time base.

In this repository, that means:

- start from the C3D analog native time
- shift it by `alignment.plot_time_shifts_sec["c3d"]`
- crop it to the inner merge window

This is already consistent with:

- `_build_c3d_mat_struct(...)`
- `alignment.inner_merge`

So the full-rate export time base should be:

```text
t_full_aligned = c3d_analog_time_native + plot_time_shifts_sec["c3d"]
```

cropped to:

```text
[inner_merge_start_sec, inner_merge_end_sec]
```

Do not downsample to TSV rate.

## Performed Path: Source Of Truth

The full-rate non-smoothed performed path must be reconstructed from the high-rate C3D analog force/moment channels, not from TSV interpolation and not from a fitted `performed` model.

### Raw signal identity

Use the selected signal from the companion traceBio settings JSON:

- `gaitway3D_total_force/CoP/Cy [Meter]`
- `gaitway3D_total_force/CoP/Cx [Meter]`

Map them as:

- `Cy`: `raw_m = Moment_Mx1 / Force_Fz1 / 1000.0`
- `Cx`: `raw_m = Moment_My1 / Force_Fz1 / 1000.0`

This matches the validated matched exports.

### Zero correction

Use the TSV raw/offset pair as the source of truth for the run-specific zero level:

```text
zero_offset_m = median(tsv_raw - tsv_offset)
offset_m = raw_m - zero_offset_m
```

This is deterministic and tied to the actual saved run.

### Percent conversion

For the requested **true non-smoothed path in percent**, use the unsmoothed record range from the settings JSON:

- `RecordMin`
- `RecordMax`

Compute:

```text
performed_percent_unsmoothed =
    clamp(100 * (offset_m - RecordMin) / (RecordMax - RecordMin), 0, 100)
```

Important:

- do **not** smooth `offset_m` before writing `performed_percent_unsmoothed`
- do **not** use `_fit_tsv_performed_model(...)`
- do **not** use `_synth_tsv_performed_values(...)`

Those fitted helpers are acceptable for preview fallback behavior, but not for authoritative MAT export.

## Desired Path: Source Of Truth

The full-rate desired-path signals must not be reconstructed from low-rate TSV columns by interpolation.

Specifically, do **not**:

- upsample `tsv.desired`
- upsample `tsv.desired_upper`
- upsample `tsv.desired_lower`

Those TSV columns are only low-rate samples of the UI state and do not preserve exact edge timing well enough.

### Required source

Use the companion traceBio settings JSON plus the original selected path file:

- settings JSON field: `SelectedPath`
- settings JSON field: `CorridorHalfWidth`
- settings JSON field: `RelativeMode`

The exporter must be able to locate and load the original path TSV corresponding to `SelectedPath`.

If that file is not yet copied into `matched/`, add support for one of these:

1. copy the selected path file into `matched/` during export, or
2. serialize the exact path vertices into the MAT file before generating the full-rate traces

Without the original path definition, exact desired-path export is not possible.

## Desired Path Evaluation Rules

The desired path must be evaluated from the exact path vertices, not from a fitted curve.

### Desired center path

For each full-rate aligned sample time:

1. convert aligned export time to path-relative time
2. evaluate the path directly from the path vertices using the same path timing as traceBio

Use the exact piecewise path definition:

- preserve flat sections
- preserve ramps
- preserve corners
- preserve start and end edges

No smoothing.

### Desired upper/lower path

Use the same corridor-generation logic as traceBio, not a new approximation.

If the traceBio path uses offset-path construction, the Python exporter must reproduce that behavior. The source-of-truth implementation is in the WPF app:

- `MainWindow.xaml.cs`
  - path sampling
  - offset-path sampling
- `SignalPlotView.cs`
  - corridor rendering assumptions

Port the actual path/corridor logic instead of inventing a simplified replacement.

In particular:

- use the exact corridor half-width from settings
- preserve exact segment boundaries
- preserve exact end behavior at the path start and path end
- do not replace the upper/lower paths with a generic spline or with a simple low-rate interpolation of TSV samples

## Time Alignment For Desired Paths

Desired-path time must align to the same aligned export time base as the performed signal.

Use:

```text
path_relative_time = t_full_aligned - t_path_start_aligned
```

where `t_path_start_aligned` must correspond to the actual run start used by traceBio for that exported TSV.

That start time must be derived from the matched run metadata, not guessed from the path shape.

The exported desired signals must therefore be generated directly on the full-rate aligned grid:

- `desired(t_full_aligned)`
- `desired_upper(t_full_aligned)`
- `desired_lower(t_full_aligned)`

## Exact Edge Requirement

This requirement is explicit:

- the three desired-path outputs must contain the **exact path information and edges**
- not merely a fit through sampled TSV values

Therefore:

- path vertices or segment definitions must be treated as exact
- edge times must come from the underlying path definition and run timing
- evaluation must be piecewise exact

Interpolation is only acceptable inside a segment where the original path definition itself is linear.

Interpolation is **not** acceptable as a substitute for:

- recovering path vertices from low-rate TSV samples
- recreating upper/lower corridor edges from low-rate sampled `desired_upper/lower`

## Concrete Code Changes Needed

The current implementation points to update are:

- `export_match_mat(...)`
- `_build_c3d_mat_struct(...)`
- `_build_tsv_mat_struct(...)`
- `_build_common_aligned_mat_struct(...)`

Add a new export builder, for example:

- `_build_tracebio_fullrate_struct(match, start_sec, end_sec)`

That builder should:

1. load the companion traceBio settings JSON
2. load or embed the exact selected path definition
3. construct the aligned full-rate C3D time base
4. derive `raw_m` from C3D analog channels
5. derive `zero_offset_m` from TSV raw-offset
6. compute `offset_m`
7. compute `performed_percent_unsmoothed`
8. evaluate exact `desired`, `desired_upper`, `desired_lower` on the same time grid
9. write all of that into the base MAT under a new top-level dict such as `tsv_reconstructed`
10. feed the same full-rate reconstructed channels into the `_4pipe.mat` export instead of the current interpolated or fitted path values

Concretely:

- `export_match_mat(...)` must attach `tsv_reconstructed`
- `export_match_pipe_bundle(...)` must include the same authoritative `2000 Hz` path channels in `_4pipe.mat`
- there must be one shared reconstruction implementation, not two independent path exporters

## Validation Requirements

After implementation, validate at least:

1. `performed_percent_unsmoothed` sampled back to TSV times agrees with the unsmoothed percent transform of TSV `offset` using JSON `RecordMin/RecordMax`
2. `desired`, `desired_upper`, and `desired_lower` sampled at TSV times agree with the original low-rate TSV columns where TSV sampling lands away from exact edges
3. around path corners and start/end edges, the full-rate export shows the exact segment transitions instead of rounded interpolation artifacts
4. `Cy` and `Cx` runs both validate correctly
5. `_4pipe.mat` and the base `.mat` agree on the exported full-rate path channels

## Non-Negotiable Rules

- No fitted `performed` export
- No TSV-rate interpolation for desired-path export
- No smoothing in `performed_percent_unsmoothed`
- Full-rate aligned output must use the C3D analog time grid
- Desired-path export must come from the exact path definition and exact run timing
- `_4pipe.mat` must be updated to carry the corrected full-rate path channels
- the base `.mat` must expose the same reconstructed channels in a dedicated top-level dict such as `tsv_reconstructed`
