import numpy as np, time, sys, random
np.random.seed(42); random.seed(42)
def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-26s [%s] %3d%% %s'%(tag[:26],bar,int(pct*100),msg)); sys.stdout.flush()

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
templates=["the {n} {v} the {n}","the {a} {n} {v}","a {a} {n} {v} the {n}",
    "the {n} {v} {p} the {p2}","the {a} {n} {v} the {a} {n}",
    "the {n} and the {n} {v}","the {n} {v} the {n} and the {n}",
    "the {a} {a} {n} {v}","the {n} {v} {p} the {a} {n}",
    "the {n} {v} the {n} {p} the {p2}"]
def gen(rng):
    return rng.choice(templates).format(n=rng.choice(nouns),v=rng.choice(verbs),
        a=rng.choice(adjs),p=rng.choice(preps),p2=rng.choice(places)).split()
rng=random.Random(42)
tr_s=[gen(rng) for _ in range(2000)]; te_s=[gen(rng) for _ in range(500)]
aw=sorted(set(w for s in tr_s+te_s for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
CTX=3
def mxy(ss):
    X,Y=[],[]
    for s in ss:
        ids=[w2i[w] for w in s]
        for i in range(CTX,len(ids)): X.append(ids[i-CTX:i]); Y.append(ids[i])
    return np.array(X),np.array(Y)
Xtr,Ytr=mxy(tr_s); Xte,Yte=mxy(te_s)
n_val=len(Xtr)//5
perm=np.random.RandomState(0).permutation(len(Xtr))
val_idx=perm[:n_val]; tr_idx=perm[n_val:]

cooc=np.zeros((V,V),dtype=np.float32)
for s in tr_s:
    ids=[w2i[w] for w in s]
    for i in range(len(ids)):
        for j in range(max(0,i-3),min(len(ids),i+4)):
            if i!=j: cooc[ids[i],ids[j]]+=1.0
P=(cooc+1e-3)/(cooc.sum()+1e-3*V*V)
ppmi=np.maximum(np.log(P/(P.sum(1,keepdims=True)*P.sum(0,keepdims=True)+1e-9)+1e-9),0)
U,S,_=np.linalg.svd(ppmi,full_matrices=False)
D_EMB=8
emb=(U[:,:D_EMB]*np.sqrt(S[:D_EMB])).astype(np.float32)
emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-9)
e1_tr=emb[Xtr[:,0]]; e2_tr=emb[Xtr[:,1]]; e3_tr=emb[Xtr[:,2]]
e1_te=emb[Xte[:,0]]; e2_te=emb[Xte[:,1]]; e3_te=emb[Xte[:,2]]
Etr=emb[Ytr]; Ete=emb[Yte]
Etr_tr=Etr[tr_idx]; Etr_val=Etr[val_idx]
Ytr_val=Ytr[val_idx]
from collections import Counter
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)

def acc_on(pred,Y):
    d=np.linalg.norm(pred[:,None,:]-emb[None,:,:],axis=2)
    return float(np.mean(d.argmin(1)==Y))

def ridge_eval(Ptr, Pval, Pte, lam=0.5):
    d=Ptr.shape[1]
    A=Ptr.T@Ptr+lam*np.eye(d)
    Th=np.linalg.solve(A,Ptr.T@Etr_tr)
    v=acc_on(Pval@Th,Ytr_val)
    t=acc_on(Pte@Th,Yte)
    l=float(np.mean((Pte@Th-Ete)**2))
    return v,t,l,d

# ---------- فرمول A: پایه (flat cos) ----------
def formula_A(e1,e2,e3,D=256,seed=100):
    r=np.random.RandomState(seed)
    x=np.concatenate([e1,e2,e3],axis=1)
    W=r.randn(x.shape[1],D).astype(np.float32)*np.sqrt(2.0/x.shape[1])
    b=r.uniform(0,2*np.pi,D).astype(np.float32)
    return np.cos(x@W+b)

# ---------- فرمول B: موقعیت‌دار (هر کلمه فرمول خودش) ----------
def formula_B(e1,e2,e3,D_per=64,seed=200):
    r=np.random.RandomState(seed)
    feats=[]
    for e in [e1,e2,e3]:
        W=r.randn(e.shape[1],D_per).astype(np.float32)*np.sqrt(2.0/e.shape[1])
        b=r.uniform(0,2*np.pi,D_per).astype(np.float32)
        feats.append(np.cos(e@W+b))
    return np.concatenate(feats,axis=1)  # 3*D_per

# ---------- فرمول C: تعاملی (ضرب بین کلمات) ----------
def formula_C(e1,e2,e3,D_per=64,seed=300):
    r=np.random.RandomState(seed)
    f1=np.cos(e1@r.randn(e1.shape[1],D_per).astype(np.float32)*np.sqrt(2.0/e1.shape[1])+r.uniform(0,6.28,D_per).astype(np.float32))
    f2=np.cos(e2@r.randn(e2.shape[1],D_per).astype(np.float32)*np.sqrt(2.0/e2.shape[1])+r.uniform(0,6.28,D_per).astype(np.float32))
    f3=np.cos(e3@r.randn(e3.shape[1],D_per).astype(np.float32)*np.sqrt(2.0/e3.shape[1])+r.uniform(0,6.28,D_per).astype(np.float32))
    # تعامل + پایه
    return np.concatenate([f1*f2, f2*f3, f1*f3, f1, f2, f3],axis=1)

# ---------- فرمول D: اتنشن‌مانند (سه‌تایی) ----------
def formula_D(e1,e2,e3,D_per=64,seed=400):
    r=np.random.RandomState(seed)
    # Q=e1, K=e2, V=e3 ساده
    Wq=r.randn(e1.shape[1],D_per).astype(np.float32)*0.3
    Wk=r.randn(e2.shape[1],D_per).astype(np.float32)*0.3
    Wv=r.randn(e3.shape[1],D_per).astype(np.float32)*0.3
    Q=e1@Wq; K=e2@Wk; V_=e3@Wv
    score=np.sum(Q*K,axis=1,keepdims=True)  # dot product
    # بدون softmax (چون فقط یک query)
    return np.concatenate([Q*V_, score*V_, Q, V_],axis=1)

def test_formula(name, fn, D_extra=None):
    Ptr=fn(e1_tr[tr_idx],e2_tr[tr_idx],e3_tr[tr_idx])
    Pval=fn(e1_tr[val_idx],e2_tr[val_idx],e3_tr[val_idx])
    Pte=fn(e1_te,e2_te,e3_te)
    v,t,l,d=ridge_eval(Ptr,Pval,Pte)
    par=sum(x.shape[1] for x in [Ptr])  # ویژگی تولیدی
    # param count = feature_dim
    return v,t,l,d

print("="*60); print(" lab29: فرمول ساختاریافته vs تخت"); print("="*60)
print("baseline most-freq=%.1f%%\n"%(base_acc*100))

results={}
# A: flat cos
print("[A] فرمول تخت: cos([e1,e2,e3]W+b)")
for D in [256, 512]:
    v,t,l,d=test_formula("A_%d"%D, lambda a,b,c,D=D: formula_A(a,b,c,D=D))
    results["A_D%d"%D]=(v,t,l,d)
    print("  D=%-4d val=%.1f%% test=%.1f%% loss=%.4f feat_dim=%d"%(D,v*100,t*100,l,d))

# B: position-aware
print("\n[B] فرمول موقعیت‌دار: cos(e1W1)·cos(e2W2)·cos(e3W3)")
for Dp in [32, 64, 128]:
    v,t,l,d=test_formula("B_%d"%Dp, lambda a,b,c,Dp=Dp: formula_B(a,b,c,Dp))
    results["B_D%d"%Dp]=(v,t,l,d)
    print("  D_per=%-4d val=%.1f%% test=%.1f%% loss=%.4f feat_dim=%d"%(Dp,v*100,t*100,l,d))

# C: interaction
print("\n[C] فرمول تعاملی: f1·f2 + f2·f3 + f1·f3 + f1 + f2 + f3")
for Dp in [32, 64, 128]:
    v,t,l,d=test_formula("C_%d"%Dp, lambda a,b,c,Dp=Dp: formula_C(a,b,c,Dp))
    results["C_D%d"%Dp]=(v,t,l,d)
    print("  D_per=%-4d val=%.1f%% test=%.1f%% loss=%.4f feat_dim=%d"%(Dp,v*100,t*100,l,d))

# D: attention-like
print("\n[D] فرمول اتنشن‌مانند: Q·V + score·V (بدون softmax)")
for Dp in [32, 64, 128]:
    v,t,l,d=test_formula("D_%d"%Dp, lambda a,b,c,Dp=Dp: formula_D(a,b,c,Dp))
    results["D_D%d"%Dp]=(v,t,l,d)
    print("  D_per=%-4d val=%.1f%% test=%.1f%% loss=%.4f feat_dim=%d"%(Dp,v*100,t*100,l,d))

print("\n"+"="*60); print(" خلاصه"); print("="*60)
print("  %-10s %-10s %-10s %-10s"%("formula","test_acc","feat_dim","vs_base"))
best_key=None; best_acc=0
for k,(v,t,l,d) in results.items():
    print("  %-10s %-10.1f %-10d %+.1f%%"%(k,t*100,d,(t-base_acc)*100))
    if t>best_acc: best_acc=t; best_key=k
print()
print("  بهترین: %s  (test=%.1f%%)"%(best_key,best_acc*100))
if best_acc>0.60:
    print("  ✓✓✓ ساختار سقف رو شکست! زبان فرمول ساختاریافته داره")
    print("       حرف تو اثبات شد: باید ساختار کشف شه، نه اندازه")
elif best_acc>0.55:
    print("  ✓ ساختار کمی کمک کرد (از ~۵۳٪ به %.0f%%)"%(best_acc*100))
    print("       ولی نه یک پیروزی بزرگ")
else:
    print("  ✗ هیچ ساختاری سقف ۵۳٪ رو نشکست")
    print("       زبان با این کلاس فرمول‌ها بیان نمی‌شه")

# آنالیز تفاوت
A_best=max([results[k][1] for k in results if k.startswith('A_')])
C_best=max([results[k][1] for k in results if k.startswith('C_')])
D_best=max([results[k][1] for k in results if k.startswith('D_')])
print("\n"+"="*60); print(" تحلیل ساختار"); print("="*60)
print("  تخت (A):          %.1f%%"%(A_best*100))
print("  تعاملی (C):       %.1f%%  (%+.1f%%)"%((C_best)*100,(C_best-A_best)*100))
print("  اتنشن‌مانند (D):  %.1f%%  (%+.1f%%)"%((D_best)*100,(D_best-A_best)*100))
if D_best>A_best+0.03:
    print("  ✓ اتنشن‌مانند > تخت → ساختار query-key کمک می‌کنه")
if C_best>A_best+0.03:
    print("  ✓ تعاملی > تخت → ضرب بین کلمات مهمه")
