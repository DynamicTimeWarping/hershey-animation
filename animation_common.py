"""Shared sampling, stroke geometry, style, and encoding for both DTW demos."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from HersheyFonts import HersheyFonts

OUT = Path(__file__).resolve().parent / 'outputs'
TEXT = 'dynamic time warping'
BLUE, ORANGE, GOLD = '#54c6ff', '#ffad75', '#f3e29b'
BG, FG, MUTED = '#0c1422', '#e9f0f8', '#93a4ba'

def trajectory(font, spacing):
    """Return ordered (x, y) samples and their stroke IDs for one font."""
    f = HersheyFonts(); f.load_default_font(font); f.normalize_rendering(1)
    strokes = [np.asarray(s, float) for s in f.strokes_for_text(TEXT)]
    allpts = np.vstack(strokes)
    # Common text width and common vertical scale; no per-letter alignment.
    lo, hi = allpts[:,0].min(), allpts[:,0].max()
    strokes = [np.column_stack(((s[:,0]-lo)/(hi-lo)*14, s[:,1]-.25)) for s in strokes]
    samples, ids = [], []
    for sid, s in enumerate(strokes):
        samples.append(s[0]); ids.append(sid)
        for a,b in zip(s[:-1],s[1:]):
            # Subdivide each original segment so no sample gap exceeds spacing.
            # Vertex order is retained; samples define a synthetic pen clock.
            n = max(1,int(np.ceil(np.linalg.norm(b-a)/spacing)))
            for t in np.linspace(0,1,n+1)[1:]:
                samples.append(a+t*(b-a)); ids.append(sid)
    # Pen-up is a jump, not an interpolated ink stroke. Stroke id is only for drawing.
    return np.array(samples), np.array(ids)

def trajectories():
    """Sample the same query and reference independently for either demo."""
    A, sa = trajectory('futural', .075)
    B, sb = trajectory('timesi', .065)
    return A, sa, B, sb


def segments(P, ids, shift):
    """Build drawable segments, excluding jumps between distinct pen strokes."""
    seg=np.stack([P[:-1],P[1:]],axis=1); seg[:,:,1]+=shift
    return seg[ids[:-1]==ids[1:]], np.flatnonzero(ids[:-1]==ids[1:])+1


def configure_style(edge_color='#334359'):
    """Use a consistent dark palette and headless rendering."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'text.color': FG,
        'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'font.size': 10, 'axes.facecolor': BG, 'figure.facecolor': BG,
        'axes.edgecolor': edge_color,
    })


def video_writer(fps=30):
    """Encode an H.264 MP4 compatible with common video players."""
    return FFMpegWriter(fps=fps, codec='libx264', bitrate=2200,
                        extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
