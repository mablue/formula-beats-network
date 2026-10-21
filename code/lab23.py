import numpy as np, time, math, sys, random
np.random.seed(42); random.seed(42)

def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-24s [%s] %3d%% %s'%(tag[:24],bar,int(pct*100),msg)); sys.stdout.flush()

# ============================================================
# [1/4] کورپس بزرگ‌تر
# ============================================================
print("="*60); print(" lab23: تست قطعی روی زبان (numpy)"); print("="*60)
nouns=['cat','dog','bird','fish','tree','house','car','book','man','woman',
       'child','friend','teacher','student','doctor','artist','river','mountain',
       'city','village','garden','library']
verbs=['sat','ate','saw','sang','ran','read','looked','found','lost','held',
       'wrote','drew','opened','closed','pushed','pulled','bought','sold']
adjs=['big','small','red','blue','old','new','happy','sad','bright','dark',
      'soft','hard','warm','cold','fast','slow']
places=['park','house','tree','sky','road','river','city','garden','library',
        'school','market','station']
preps=['in','on','at','near','under','over','behind','beside']

templates=[
    "the {n} {v} the {n}",
    "the {a} {n} {v}",
    "a {a} {n} {v} the {n}",
    "the {n} {v} {p} the {p2}",
    "the {a} {n} {v} the {a} {n}",
    "the {n} and the {n} {v}",
    "the {n} {v} the {n} and the {n}",
    "the {a} {a} {n} {v}",
    "the {n} {v} {p} the {a} {n}",
    "the {n} {v} the {n} {p} the {p2}",
]
def gen(rng):
    return rng.choice(templates).format(
        n=rng.choice(nouns),v=rng.choice(verbs),a=rng.choice(adjs),
        p=rng.choice(preps),p2=rng.choice(places)).split()

rng=random.Random(42)
tr_s=[gen(rng) for _ in range(2000)]
te_s=[gen(rng) for _ in range(500)]
aw=sorted(set(w for s in tr_s+te_s for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
print("\n[1/4] کورپس")
print("  vocab=%d | train=%d جمله | test=%d جمله"%(V,len(tr_s),len(te_s)))

# زوج (context=3, target)
CTX=3
def make_xy(sents):
    X,Y=[],[]
    for s in sents:
        ids=[w2i[w] for w in s]
        for i in range(CTX,len(ids)):
            X.append(ids[i-CTX:i]); Y.append(ids[i])
    return np.array(X), np.array(Y)
Xtr,Ytr=make_xy(tr_s); Xte,Yte=make_xy(te_s)
print("  نمونه آموزش=%d | تست=%d"%(len(Xtr),len(Xte)))

# ============================================================
# [2/4] Embedding از PPMI+SVD
# ============================================================
print("\n[2/4] ساخت embedding")
cooc=np.zeros((V,V),dtype=np.float32)
for s in tr_s:
    ids=[w2i[w] for w in s]
    for i in range(len(ids)):
        for j in range(max(0,i-3),min(len(ids),i+4)):
            if i!=j: cooc[ids[i],ids[j]]+=1.0
total=cooc.sum()
P=(cooc+1e-3)/(total+1e-3*V*V)
pi=P.sum(axis=1,keepdims=True); pj=P.sum(axis=0,keepdims=True)
pmi=np.log(P/(pi*pj+1e-9)+1e-9)
ppmi=np.maximum(pmi,0)
U,S,Vt=np.linalg.svd(ppmi,full_matrices=False)
D_EMB=8
emb=(U[:,:D_EMB]*np.sqrt(S[:D_EMB])).astype(np.float32)
# نرمالیزه
emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-9)
print("  embedding: %d بعد از PPMI+SVD"%(D_EMB))

# ورودی هر مدل: concat embedding ۳ کلمه‌ی context
def ctx_emb(X): return np.concatenate([emb[X[:,0]],emb[X[:,1]],emb[X[:,2]]],axis=1)
Ftr=ctx_emb(Xtr); Fte=ctx_emb(Xte)  # (n, CTX*d_emb)
Etr=emb[Ytr]; Ete=emb[Yte]  # هدف: embedding کلمه بعدی

# baseline
from collections import Counter
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)

def acc_from_emb(pred, targets):
    """پیش‌بینی کلمه از نزدیک‌ترین embedding"""
    # pred: (n, d_emb) | targets: (n,) کلمه‌ها
    dists=np.linalg.norm(pred[:,None,:]-emb[None,:,:],axis=2)  # (n, V)
    return float(np.mean(dists.argmin(axis=1)==targets))

# ============================================================
# [3/4] آموزش ۳ مدل
# ============================================================
print("\n[3/4] آموزش مدل‌ها")

# ---------- 3a: MLP ----------
print("\n  [3a] MLP")
di=Ftr.shape[1]; dh=32; do=D_EMB
rng_np=np.random.RandomState(42)
W1=rng_np.randn(di,dh).astype(np.float32)*np.sqrt(2.0/di)
b1=np.zeros(dh,dtype=np.float32)
W2=rng_np.randn(dh,do).astype(np.float32)*np.sqrt(2.0/dh)
b2=np.zeros(do,dtype=np.float32)
vW1=np.zeros_like(W1); vb1=np.zeros_like(b1); vW2=np.zeros_like(W2); vb2=np.zeros_like(b2)
t0=time.time()
for ep in range(100):
    z=Ftr@W1+b1; h=np.maximum(z,0); p=h@W2+b2
    diff=(p-Etr)*2/len(Ftr)
    dW2=h.T@diff; db2=diff.sum(0)
    dh_=diff@W2.T; dz=dh_*(z>0)
    dW1=Ftr.T@dz; db1=dz.sum(0)
    vW1=0.9*vW1-0.05*dW1; W1+=vW1
    vb1=0.9*vb1-0.05*db1; b1+=vb1
    vW2=0.9*vW2-0.05*dW2; W2+=vW2
    vb2=0.9*vb2-0.05*db2; b2+=vb2
    if (ep+1)%25==0 or ep==0:
        ph=np.maximum(Fte@W1+b1,0)@W2+b2
        a=acc_from_emb(ph,Yte); l=float(np.mean((ph-Ete)**2))
        render("MLP",ep+1,100,"loss=%.4f acc=%.1f%%"%(l,100*a))
sys.stdout.write('\n'); sys.stdout.flush()
mlp_pred=np.maximum(Fte@W1+b1,0)@W2+b2
mlp_acc=acc_from_emb(mlp_pred,Yte); mlp_loss=float(np.mean((mlp_pred-Ete)**2))
mlp_params=di*dh+dh+dh*do+do
mlp_time=time.time()-t0
print("  → acc=%.1f%% loss=%.4f params=%d time=%.1fs"%(mlp_acc*100,mlp_loss,mlp_params,mlp_time))

# ---------- 3b: Fuzzy TSK ----------
print("\n  [3b] Fuzzy TSK")
K=48; di=Ftr.shape[1]; do=D_EMB
rng_f=np.random.RandomState(43)
# مراکز: نمونه‌های تصادفی
idx=rng_f.choice(len(Ftr),K,replace=False)
C=Ftr[idx].copy()
s_width=np.ones(K,dtype=np.float32)*2.0
B=rng_f.randn(K,do).astype(np.float32)*0.01
t0=time.time()
for ep in range(100):
    # forward: μ_k = exp(-||f - c_k||²/2s²), out = Σμ_k·b_k / Σμ_k
    diff=Ftr[:,None,:]-C[None,:,:]                # (n,K,di)
    d2=(diff*diff).sum(axis=2)                    # (n,K)
    mu=np.exp(-d2/(2*s_width[None,:]**2+1e-9))    # (n,K)
    Z=mu.sum(axis=1,keepdims=True)+1e-9
    pred=(mu[:,:,None]*B[None,:,:]).sum(axis=1)/Z # (n,do)
    # grad only on B (keep C,s fixed)
    err=(pred-Etr)*2/len(Ftr)                     # (n,do)
    dB=(mu[:,:,None]*err[:,None,:]).sum(axis=0)/Z.sum()
    B-=0.1*dB
    if (ep+1)%25==0 or ep==0:
        diff2=Fte[:,None,:]-C[None,:,:]; d2t=(diff2*diff2).sum(2)
        mut=np.exp(-d2t/(2*s_width[None,:]**2+1e-9)); Zt=mut.sum(1,keepdims=True)+1e-9
        pt=(mut[:,:,None]*B[None,:,:]).sum(1)/Zt
        a=acc_from_emb(pt,Yte); l=float(np.mean((pt-Ete)**2))
        render("Fuzzy K=48",ep+1,100,"loss=%.4f acc=%.1f%%"%(l,100*a))
sys.stdout.write('\n'); sys.stdout.flush()
diff2=Fte[:,None,:]-C[None,:,:]; d2t=(diff2*diff2).sum(2)
mut=np.exp(-d2t/(2*s_width[None,:]**2+1e-9)); Zt=mut.sum(1,keepdims=True)+1e-9
fz_pred=(mut[:,:,None]*B[None,:,:]).sum(1)/Zt
fz_acc=acc_from_emb(fz_pred,Yte); fz_loss=float(np.mean((fz_pred-Ete)**2))
fz_params=K*do+K
fz_time=time.time()-t0
print("  → acc=%.1f%% loss=%.4f params=%d time=%.1fs"%(fz_acc*100,fz_loss,fz_params,fz_time))

# ---------- 3c: Random Features (ridge) ----------
print("\n  [3c] Random Features (ridge، بدون training)")
D_RF=256
rng_rf=np.random.RandomState(44)
W_rf=rng_rf.randn(di,D_RF).astype(np.float32)*np.sqrt(2.0/di)
b_rf=rng_rf.uniform(0,2*np.pi,D_RF).astype(np.float32)
t0=time.time()
Phi_tr=np.cos(Ftr@W_rf+b_rf)
Phi_te=np.cos(Fte@W_rf+b_rf)
lam=0.5
A=Phi_tr.T@Phi_tr+lam*np.eye(D_RF)
Th=np.linalg.solve(A,Phi_tr.T@Etr)
rf_pred=Phi_te@Th
rf_acc=acc_from_emb(rf_pred,Yte); rf_loss=float(np.mean((rf_pred-Ete)**2))
rf_params=D_RF*do+di*D_RF
rf_time=time.time()-t0
print("  → acc=%.1f%% loss=%.4f params=%d time=%.1fs"%(rf_acc*100,rf_loss,rf_params,rf_time))

# ============================================================
# [4/4] نتیجه
# ============================================================
print("\n"+"="*60); print(" [4/4] مقایسه نهایی"); print("="*60)
print("  %-20s %-10s %-10s %-10s %-8s"%("model","test_acc","test_loss","params","time"))
print("  %-20s %-10.1f %-10s %-10s %-8s"%("Most-freq",base_acc*100,"-","-","-"))
for name,acc,loss,par,t in [("MLP",mlp_acc,mlp_loss,mlp_params,mlp_time),
                             ("Fuzzy TSK",fz_acc,fz_loss,fz_params,fz_time),
                             ("RF (ridge)",rf_acc,rf_loss,rf_params,rf_time)]:
    print("  %-20s %-10.1f %-10.4f %-10d %-8.1fs"%(name,acc*100,loss,par,t))

print("\n"+"="*60); print(" نتیجه‌گیری"); print("="*60)
best=max([("MLP",mlp_acc,mlp_params),("Fuzzy TSK",fz_acc,fz_params),("RF",rf_acc,rf_params)],
         key=lambda x:x[1])
print("  بهترین: %s (acc=%.1f%%, params=%d)"%(best[0],best[1]*100,best[2]))
print()
gains={
    "MLP": (mlp_acc-base_acc)*100,
    "Fuzzy": (fz_acc-base_acc)*100,
    "RF": (rf_acc-base_acc)*100
}
for k,v in gains.items():
    print("  %-8s vs baseline: %+.1f%%"%(k,v))
print()
if mlp_acc>base_acc+0.08 or fz_acc>base_acc+0.08:
    winner="MLP" if mlp_acc>fz_acc else "Fuzzy"
    print("  ✓✓✓ %s یاد گرفت!"%winner)
    if winner=="Fuzzy":
        print("       فرمول فازی روی زبان برنده شد → حرف تو تأیید شد")
    else:
        print("       MLP برنده شد → برای این داده، شبکه معمولی کافیه")
elif rf_acc>base_acc+0.05:
    print("  ✓ RF یاد گرفت بدون آموزش تکراری → فرمول کرنل کار می‌کنه")
else:
    print("  ✗ هیچ مدلی یاد نگرفت → داده هنوز کافی/ساختارمند نیست")
