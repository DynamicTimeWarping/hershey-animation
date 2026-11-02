"""Render growing-query open-end DTW: uv run make_open_end_animation.py.

Samples the same trajectories as make_animation.py and computes
an independent alignment for each frame. Requires ffmpeg on PATH.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from dtw import dtw

from animation_common import (
    OUT, FG, MUTED, BLUE, ORANGE, GOLD,
    trajectories, segments, configure_style, video_writer,
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A, sa, B, sb = trajectories()
    seg_a, ends_a = segments(A, sa, 2)
    seg_b, ends_b = segments(B, sb, 0)
    configure_style(edge_color=MUTED)
    fig = plt.figure(figsize=(14.4, 9), dpi=100)
    fig.text(.05, .95, 'OPEN-END DYNAMIC TIME WARPING', fontsize=24, weight='bold')
    fig.text(.05, .91, 'Growing A prefix → full B reference · a fresh alignment every frame',
             fontsize=13, color=MUTED)
    ax = fig.add_axes([.05, .52, .9, .33])
    ax.set(xlim=(-.2, 14.2), ylim=(-.4, 3.5), aspect='equal')
    ax.axis('off')
    for seg, color in [(seg_a, BLUE), (seg_b, ORANGE)]:
        ax.add_collection(LineCollection(seg, colors=color, alpha=.18, linewidths=1.4))
    ink_a = LineCollection([], colors=BLUE, linewidths=2.1)
    ink_b = LineCollection([], colors=ORANGE, linewidths=2.1)
    links = LineCollection([], colors=GOLD, linewidths=.6, alpha=.3)
    for collection in (ink_a, ink_b, links):
        ax.add_collection(collection)
    tip_a, = ax.plot([], [], 'o', color=BLUE, ms=8)
    tip_b, = ax.plot([], [], 'o', color=ORANGE, ms=8)
    ax.text(0, 3.08, 'A / QUERY PREFIX', color=BLUE, weight='bold')
    ax.text(0, 1.06, 'B / BEST MATCHING REFERENCE PREFIX', color=ORANGE, weight='bold')
    status = fig.text(.05, .48, '', color=GOLD, family='monospace', fontsize=11)

    path_ax = fig.add_axes([.07, .12, .38, .28])
    path_ax.set(xlim=(0, len(A)-1), ylim=(0, len(B)-1),
                xlabel='A sample index i', ylabel='B sample index j',
                title='CURRENT COMPLETE WARPING PATH')
    path, = path_ax.plot([], [], color=GOLD, lw=1.5)
    endpoint, = path_ax.plot([], [], 'o', color=FG, ms=5)
    frontier = path_ax.axvline(0, color=BLUE, alpha=.5)
    history_ax = fig.add_axes([.56, .12, .39, .28])
    history_ax.set(xlim=(1, len(A)), ylim=(0, len(B)-1),
                   xlabel='Observed A samples', ylabel='Selected B endpoint j',
                   title='ENDPOINT ESTIMATE AS QUERY GROWS')
    history, = history_ax.plot([], [], color=ORANGE, lw=1.5)
    fig.text(.05, .035,
             'Euclidean (x, y) cost · symmetric2 · fixed start, free reference end · earlier matches may change',
             color=MUTED, fontsize=10)

    records = []
    writer = video_writer()
    with writer.saving(fig, str(OUT / 'dtw_open_end.mp4'), 100):
        for n in range(1, len(A)+1):
            # KEY STEP: align all observed A samples to a leading portion of B.
            # open_end=True is Python's spelling of R's open.end=TRUE.
            # symmetric2 chooses j minimizing D(n-1,j)/(n+j+1).
            # Recompute and backtrack the whole path; this does not freeze prior
            # matches, and the selected reference endpoint can move backwards.
            alignment = dtw(A[:n], B, dist_method='euclidean',
                            step_pattern='symmetric2', open_end=True)
            ii, jj = alignment.index1, alignment.index2
            j = int(jj[-1])
            assert ii[0] == jj[0] == 0 and ii[-1] == n-1
            assert j == alignment.jmin and 0 <= j < len(B)
            records.append({'query_samples': n, 'reference_endpoint': j,
                            'matched_pairs': len(ii),
                            'normalized_distance': float(alignment.normalizedDistance)})
            ink_a.set_segments(seg_a[ends_a < n])
            ink_b.set_segments(seg_b[ends_b <= j])
            # Show correspondences over the entire current path, sparsely enough
            # to retain legibility, rather than advancing along a single path.
            chosen = np.unique(np.r_[np.arange(0, len(ii), 12), len(ii)-1])
            links.set_segments(np.stack([A[ii[chosen]] + [0, 2], B[jj[chosen]]], axis=1))
            tip_a.set_data([A[n-1, 0]], [A[n-1, 1]+2])
            tip_b.set_data([B[j, 0]], [B[j, 1]])
            path.set_data(ii, jj)
            endpoint.set_data([n-1], [j])
            frontier.set_xdata([n-1, n-1])
            history.set_data(np.arange(1, n+1), [r['reference_endpoint'] for r in records])
            status.set_text(f'A: {n:3d}/{len(A)} samples    B endpoint: {j:4d}/{len(B)-1}'
                            f'    normalized distance: {alignment.normalizedDistance:.5f}')
            writer.grab_frame()
            if n == len(A)//2:
                fig.savefig(OUT / 'dtw_open_end_preview.png', dpi=100)
            if n % 50 == 0:
                print(f'Rendered {n}/{len(A)} alignments', flush=True)
    (OUT / 'dtw_open_end_metrics.json').write_text(json.dumps(records, indent=2) + '\n')
    plt.close(fig)
    print(f'Done: {len(records)} frames, one alignment per frame', flush=True)


if __name__ == '__main__':
    main()
