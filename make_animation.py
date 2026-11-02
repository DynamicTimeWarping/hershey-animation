"""Reproduce with: uv sync --locked; uv run make_animation.py.
Requires ffmpeg on PATH. No character labels are used in DTW.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter
from HersheyFonts import HersheyFonts
from dtw import dtw

OUT = Path(__file__).resolve().parent / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
TEXT = 'dynamic time warping'
BLUE, ORANGE, GOLD = '#54c6ff', '#ffad75', '#f3e29b'
BG, FG, MUTED = '#0c1422', '#e9f0f8', '#93a4ba'

def trajectory(font, spacing):
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
            n = max(1,int(np.ceil(np.linalg.norm(b-a)/spacing)))
            for t in np.linspace(0,1,n+1)[1:]:
                samples.append(a+t*(b-a)); ids.append(sid)
    # Pen-up is a jump, not an interpolated ink stroke. Stroke id is only for drawing.
    return np.array(samples), np.array(ids)

A, sa = trajectory('futural', .075)
B, sb = trajectory('timesi', .065)
alignment = dtw(A, B, dist_method='euclidean', step_pattern='symmetric2', keep_internals=True)
ii, jj = alignment.index1, alignment.index2
assert (ii[0],jj[0]) == (0,0) and (ii[-1],jj[-1]) == (len(A)-1,len(B)-1)
assert np.all(np.isin(np.diff(ii),[0,1])) and np.all(np.isin(np.diff(jj),[0,1]))
np.savez_compressed(OUT/'alignment.npz', A=A, B=B, stroke_A=sa, stroke_B=sb, index_A=ii,index_B=jj, local_cost=alignment.localCostMatrix)
metrics={'font_A':'futural','font_B':'timesi','samples_A':len(A),'samples_B':len(B),'matched_pairs':len(ii),'normalized_distance':float(alignment.normalizedDistance)}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(metrics, flush=True)
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':FG,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'font.size':10,'axes.facecolor':BG,'figure.facecolor':BG,'axes.edgecolor':'#334359'})
fig=plt.figure(figsize=(14.4,9),dpi=100)
fig.text(.05,.95,'DYNAMIC TIME WARPING',fontsize=25,weight='bold')
fig.text(.05,.913,'The same words. Two vector fonts. One monotone correspondence.',fontsize=13,color=MUTED)
ax=fig.add_axes([.05,.51,.90,.35]); ax.set_xlim(-.2,14.2); ax.set_ylim(-.4,3.5); ax.set_aspect('equal'); ax.axis('off')
shiftA, shiftB = 2.0,0.0

def segments(P, ids, shift):
    seg=np.stack([P[:-1],P[1:]],axis=1); seg[:,:,1]+=shift
    return seg[ids[:-1]==ids[1:]], np.flatnonzero(ids[:-1]==ids[1:])+1
segA, endsA=segments(A,sa,shiftA); segB,endsB=segments(B,sb,shiftB)
for seg,c in [(segA,BLUE),(segB,ORANGE)]: ax.add_collection(LineCollection(seg,colors=c,alpha=.22,linewidths=1.4))
inkA=LineCollection([],colors=BLUE,linewidths=2.1); inkB=LineCollection([],colors=ORANGE,linewidths=2.1)
ax.add_collection(inkA);ax.add_collection(inkB)
ax.text(0,3.08,'A  /  HERSHEY FUTURAL',color=BLUE,fontsize=10,weight='bold')
ax.text(0,1.06,'B  /  HERSHEY TIMES ITALIC',color=ORANGE,fontsize=10,weight='bold')
links=LineCollection([],colors=GOLD,linewidths=.7,alpha=.18);ax.add_collection(links)
link,=ax.plot([],[],color=GOLD,lw=1.6,alpha=.9)
pa,=ax.plot([],[],'o',ms=8,color=BLUE,mec=FG,mew=1)
pb,=ax.plot([],[],'o',ms=8,color=ORANGE,mec=FG,mew=1)
status=fig.text(.05,.485,'',fontsize=12,color=GOLD,family='monospace')
# Matrix is local Euclidean cost, not accumulated cost.
cax=fig.add_axes([.07,.115,.38,.29])
cax.imshow(alignment.localCostMatrix.T,origin='lower',aspect='auto',cmap='magma_r',vmin=0,vmax=4,extent=[0,len(A)-1,0,len(B)-1],interpolation='nearest')
cax.plot(ii,jj,color='white',alpha=.15,lw=1)
trail,=cax.plot([],[],color=GOLD,lw=1.8)
cp,=cax.plot([],[],'o',color=FG,ms=5)
cax.set_title('LOCAL DISTANCE + OPTIMAL WARPING PATH',loc='left',color=FG,fontsize=11,pad=13)
cax.set_xlabel('A sample index  i');cax.set_ylabel('B sample index  j')
cax.text(.02,.95,'light = closer coordinates',transform=cax.transAxes,color=FG,fontsize=9,va='top')
tax=fig.add_axes([.55,.115,.40,.29])
k=np.arange(len(ii))
tax.plot(k,A[ii,1],color=BLUE,alpha=.8,lw=.8,label='A: y(i)')
tax.plot(k,B[jj,1],color=ORANGE,alpha=.8,lw=.8,label='B: y(j)')
cursor=tax.axvline(0,color=GOLD,lw=1.5)
ta,=tax.plot([],[],'o',color=BLUE,ms=5);tb,=tax.plot([],[],'o',color=ORANGE,ms=5)
tax.set_xlim(0,len(ii)-1);tax.set_ylim(-.3,.95)
tax.set_title('VERTICAL COORDINATES ON THE WARPED CLOCK',loc='left',color=FG,fontsize=11,pad=13)
tax.set_xlabel('Matched-pair index  k');tax.set_ylabel('y position')
tax.legend(loc='upper right',frameon=False,labelcolor=FG,ncol=2,fontsize=9)
tax.spines[['top','right']].set_visible(False)
fig.text(.05,.036,'DTW uses (x, y) Euclidean distance • symmetric2 steps • pen-up moves are jumps • display rows are offset only',fontsize=10,color=MUTED)

def update(k):
    i,j=int(ii[k]),int(jj[k]); a=A[i]+[0,shiftA];b=B[j]+[0,shiftB]
    inkA.set_segments(segA[endsA<=i]);inkB.set_segments(segB[endsB<=j])
    pa.set_data([a[0]],[a[1]]);pb.set_data([b[0]],[b[1]])
    link.set_data([a[0],b[0]],[a[1],b[1]])
    recent=np.arange(max(0,k-70),k+1,7)
    links.set_segments(np.stack([A[ii[recent]]+[0,shiftA],B[jj[recent]]+[0,shiftB]],axis=1))
    trail.set_data(ii[:k+1],jj[:k+1]);cp.set_data([i],[j])
    cursor.set_xdata([k,k]);ta.set_data([k],[A[i,1]]);tb.set_data([k],[B[j,1]])
    action='START' if k==0 else ('B advances / A holds' if ii[k]==ii[k-1] else 'A advances / B holds' if jj[k]==jj[k-1] else 'Both advance')
    status.set_text(f'PAIR {k+1:4d}/{len(ii)}     A[{i:4d}] ↔ B[{j:4d}]     {action}')

fps=30
writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=2200,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
with writer.saving(fig,str(OUT/'dtw_text_alignment.mp4'),100):
    update(0)
    for _ in range(20):writer.grab_frame()
    for k in range(len(ii)):
        update(k);writer.grab_frame()
        if k%250==0: print('frame',k,flush=True)
    for _ in range(60):writer.grab_frame()
update(len(ii)//2);fig.savefig(OUT/'preview.png',dpi=100)
plt.close(fig)
print('Done',flush=True)
