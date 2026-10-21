import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os
os.makedirs('figures',exist_ok=True)
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans'})

# Fig1: bitsweep
fig,ax1=plt.subplots(figsize=(6,3.5))
bits=[1,2,3,4,6,8]
h=[0.818,0.950,0.987,0.997,0.9998,1.0]
l=[0.599,0.872,0.967,0.992,0.9994,1.0]
a=[18.75,37.5,62.5,100,100,100]
ax1.plot(bits,h,'o-',label='Hidden cosine',color='#1f77b4',lw=2)
ax1.plot(bits,l,'s--',label='Logits cosine',color='#ff7f0e',lw=2)
ax1.set_xlabel('Quantization bits'); ax1.set_ylabel('Cosine similarity')
ax1.set_ylim(0.4,1.05); ax1.grid(alpha=0.3); ax1.legend(loc='lower right')
ax2=ax1.twinx(); ax2.plot(bits,a,'^:',color='#2ca02c',alpha=0.6)
ax2.set_ylabel('Argmax agreement (%)',color='#2ca02c')
ax2.tick_params(axis='y',colors='#2ca02c'); ax2.set_ylim(0,110)
plt.tight_layout(); plt.savefig('figures/fig1_bitsweep.pdf'); plt.close()

# Fig2: layers
fig,ax=plt.subplots(figsize=(6,3.5))
layers=['Layer 0','Layer 1','Layer 2','Layer 3']
r=[0.146,0.169,0.380,0.078]; idn=[0.053,0.150,0.130,0.057]; q=[0.017,0.016,0.016,0.014]
x=np.arange(4); w=0.25
ax.bar(x-w,r,w,label='Random',color='#d62728',alpha=0.8)
ax.bar(x,idn,w,label='Identity',color='#ff7f0e',alpha=0.8)
ax.bar(x+w,q,w,label='4-bit quant',color='#2ca02c',alpha=0.8)
ax.axhline(0.0145,color='black',linestyle='--',label='Baseline',lw=1)
ax.set_xticks(x); ax.set_xticklabels(layers); ax.set_ylabel('Test loss')
ax.legend(); ax.grid(alpha=0.3,axis='y')
plt.tight_layout(); plt.savefig('figures/fig2_layers.pdf'); plt.close()

# Fig3: discovery
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3.2))
D=[8,16,32,64,128,256,512,1024]; Da=[31.2,35.8,44.8,47.9,50.4,51.8,51.3,50.8]
ax1.plot(range(len(D)),Da,'o-',color='#1f77b4',lw=2)
ax1.set_xticks(range(len(D))); ax1.set_xticklabels(D)
ax1.set_xlabel('RF Dimension D'); ax1.set_ylabel('Validation accuracy (%)')
ax1.set_title('(a) Width growth'); ax1.axvline(5,color='red',linestyle='--',alpha=0.5)
ax1.grid(alpha=0.3)
L=[1,2,3,4]; La=[51.8,50.8,50.2,50.2]
ax2.plot(L,La,'s-',color='#ff7f0e',lw=2)
ax2.set_xlabel('Number of layers L'); ax2.set_ylabel('Validation accuracy (%)')
ax2.set_title('(b) Depth growth'); ax2.axvline(1,color='red',linestyle='--',alpha=0.5)
ax2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('figures/fig3_discovery.pdf'); plt.close()

# Fig4: RF vs MLP
fig,ax=plt.subplots(figsize=(6,3.5))
models=['Baseline\n(most-freq)','MLP\nh=256','MLP\nh=512','RF\nD=256','RF\nD=512','RF\nD=1024']
accs=[25.7,36.4,38.3,41.1,42.5,43.2]; times=[0,28.3,53.3,0.52,1.14,2.72]
colors=['#888']+['#ff7f0e']*2+['#2ca02c']*3
bars=ax.bar(models,accs,color=colors,alpha=0.8)
for bar,t in zip(bars,times):
    y=bar.get_height(); lbl=f'{t:.1f}s' if t>0 else '—'
    ax.text(bar.get_x()+bar.get_width()/2,y+0.5,lbl,ha='center',fontsize=8)
ax.set_ylabel('Test accuracy (%)'); ax.set_ylim(0,50); ax.grid(alpha=0.3,axis='y')
plt.tight_layout(); plt.savefig('figures/fig4_rf_vs_mlp.pdf'); plt.close()

# Fig5: SINDy
fig,ax=plt.subplots(figsize=(6,3))
ds=['Poly','Periodic','Mixed','Language']
mlp=[0.00478,0.33134,0.00385,2.03]; fml=[0.0,0.0,0.00521,0.077]
x=np.arange(4); w=0.35
ax.bar(x-w/2,mlp,w,label='MLP',color='#ff7f0e',alpha=0.8)
ax.bar(x+w/2,fml,w,label='Formula (SINDy)',color='#2ca02c',alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(ds,fontsize=8)
ax.set_ylabel('Test loss'); ax.set_yscale('symlog',linthresh=0.01)
ax.legend(); ax.grid(alpha=0.3,axis='y')
plt.tight_layout(); plt.savefig('figures/fig5_sindy.pdf'); plt.close()
print("✅ 5 figures saved")
