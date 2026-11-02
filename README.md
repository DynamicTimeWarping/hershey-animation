# Dynamic time warping of vector lettering

The exact text `dynamic time warping` is rendered as Hershey `futural` and `timesi` (Times Italic) vector strokes with Hershey-Fonts 2.1.0.

## Run

Install `uv` and ffmpeg, then run from the repository root:

```bash
uv sync --locked
uv run make_animation.py
```

`uv` installs the Python 3.14 interpreter selected by `.python-version` if needed and manages the project environment and dependencies using `pyproject.toml` and `uv.lock`. ffmpeg must be available on your `PATH`.

Generated files are saved in `outputs/` next to the script. The directory is created automatically.

This uses the DTW suite's Python implementation, **dtw-python**, imported as `from dtw import dtw`. It is the Python port of the R `dtw` package, not the unrelated PyPI package named `dtw`.

## Method

Each series has two channels: x and y. Fonts are normalized to the same text width (14 units), with the same vertical scaling and baseline. This is a whole-string affine normalization; no letters are matched beforehand. Each original stroke segment is sampled at approximately uniform spatial spacing, using maximum spacings 0.075 for A and 0.065 for B. These define discrete synthetic pen times, not measured handwriting times. Original vertex and stroke order is preserved. Pen-up moves are instantaneous jumps: no extra points are inserted in the air, and no connecting ink line is drawn. Stroke IDs control rendering only, not matching.

The complete sequences are aligned once using:

```python
alignment = dtw(A, B, dist_method='euclidean',
                step_pattern='symmetric2', keep_internals=True)
i, j = alignment.index1, alignment.index2
```

The local cost is c(i,j) = ||A[i] - B[j]||_2. The symmetric2 recurrence is:

D(i,j) = min(D(i-1,j)+c(i,j), D(i,j-1)+c(i,j), D(i-1,j-1)+2*c(i,j)).

Alignment is global, starts and ends at sequence endpoints, and has no band constraint. Horizontal and vertical steps repeat a sample from one series while advancing the other. The normalized distance is the weighted cumulative cost divided by N+M.

## Animation

Every DTW matched pair appears in one video frame (30 pairs per second), with pauses at start and end. The two bright pen tips and gold link show the current pair; faint recent links show the local correspondence. Vertical separation of the text rows is only for display and is excluded from the cost. The lower-left panel shows the **local**, not cumulative, distance matrix and the selected path. The lower-right panel shows the y coordinates on the matched-pair clock; alignment still uses both x and y. The video reveals a precomputed optimum, not an online solution or a visualization of filling the dynamic-programming table.

Different fonts have different stroke counts and stroke orders. Coordinate-only DTW can therefore produce imperfect matches; it is not character recognition and does not guarantee semantic stroke correspondence.

## Files

- `outputs/dtw_text_alignment.mp4`: full animation, one frame per match
- `outputs/preview.png`: still preview
- `make_animation.py`: reproducible source
- `outputs/alignment.npz`: coordinates, stroke IDs, both alignment index arrays, and local cost matrix
- `outputs/metrics.json`: sample counts and normalized distance

## References

- DTW suite: https://dynamictimewarping.github.io/
- API: https://dynamictimewarping.github.io/py-api/html/api/dtw.dtw.html
- Hershey font loader and built-in data: https://github.com/apshu/HersheyFonts

Font data and third-party packages retain their original licenses. Install these dependencies separately; they are not bundled.
