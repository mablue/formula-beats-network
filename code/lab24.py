import numpy as np, time, math, sys, random
np.random.seed(42); random.seed(42)
def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-26s [%s] %3d%% %s'%(tag[:26],bar,int(pct*100),msg)); sys.stdout.flush()

# داده (همون lab23)
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
    "the {n} {v} the {n}","the {a} {n} {v}","a {a} {n} {v} the {n}",
    "the {n} {v} {p} the {p2}","the {a} {n} {v} the {a} {n}",
    "the {n} and the {n} {v}","the {n} {v} the {n} and the {n}",
    "the {a} {a} {n} {v}","the {n} {v} {p} the {a} {n}",
    "the {n} {v} the {n} {p} the {p2}",
]
def gen(rng):
    return rng.choice(templates).format(
        n=rng.choice(nouns),v=rng.choice(verbs),a=rng.choice(adjs),
        p=rng.choice(preps),p2=rng.choice(places)).split()
rng=random.Random(42)
tr_s=[gen(rng) for _ in range(2000)]; te_s=[gen(rng) for _ in range(500)]
aw=sorted(set(w for s in tr_s+te_s for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
CTX=3
def make_xy(sents):
    X,Y=[],[]
    for s in sents:
        ids=[w2i[w] for w in s]
        for i in range(CTX,len(ids)): X.append(ids[i-CTX:i]); Y.append(ids[i])
    return np.array(X),np.array(Y)
Xtr,Ytr=make_xy(tr_s); Xte,Yte=make_xy(te_s)

# embedding
cooc=np.zeros((V,V),dtype=np.float32)
for s in tr_s:
    ids=[w2i[w] for w in s]
    for i in range(len(ids)):
        for j in range(max(0,i-3),min(len(ids),i+4)):
            if i!=j: cooc[ids[i],ids[j]]+=1.0
P=(cooc+1e-3)/(cooc.sum()+1e-3*V*V)
pi=P.sum(1,keepdims=True); pj=P.sum(0,keepdims=True)
ppmi=np.maximum(np.log(P/(pi*pj+1e-9)+1e-9),0)
U,S,_=np.linalg.svd(ppmi,full_matrices=False)
D_EMB=8
emb=(U[:,:D_EMB]*np.sqrt(S[:D_EMB])).astype(np.float32)
emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-9)

Ftr=np.concatenate([emb[Xtr[:,0]],emb[Xtr[:,1]],emb[Xtr[:,2]]],axis=1)
Fte=np.concatenate([emb[Xte[:,0]],emb[Xte[:,1]],emb[Xte[:,2]]],axis=1)
Etr=emb[Ytr]; Ete=emb[Yte]
from collections import Counter
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)

def acc_from_emb(pred):
    dists=np.linalg.norm(pred[:,None,:]-emb[None,:,:],axis=2)
    return float(np.mean(dists.argmin(1)==Yte))

def build_features(F, layers, D_RF, seed, kernel='cos'):
    """F: (n, d_in) → لایه‌های RF"""
    rng=np.random.RandomState(seed)
    cur=F.copy()
    for i,din in enumerate([F.shape[1]]+layers[:-1]):
        dout=layers[i]
        W=rng.randn(din,dout).astype(np.float32)*np.sqrt(2.0/din)
        b=rng.uniform(0,2*np.pi,dout).astype(np.float32)
        if kernel=='cos': cur=np.cos(cur@W+b)
        elif kernel=='sin': cur=np.sin(cur@W+b)
        elif kernel=='elu': cur=np.maximum(cur@W+b,0)+1
        elif kernel=='tanh': cur=np.tanh(cur@W+b)
        elif kernel=='rbf': cur=np.exp(-((cur@W+b)**2))
    return cur

def run_rf(layers, D_RF=256, lam=0.5, kernel='cos', seed=100):
    Phi_tr=build_features(Ftr, layers, D_RF, seed, kernel)
    Phi_te=build_features(Fte, layers, D_RF, seed, kernel)
    d=Phi_tr.shape[1]
    A=Phi_tr.T@Phi_tr+lam*np.eye(d)
    Th=np.linalg.solve(A,Phi_tr.T@Etr)
    pred=Phi_te@Th
    acc=acc_from_emb(pred)
    loss=float(np.mean((pred-Ete)**2))
    params=sum(Ftr.shape[1] if i==0 else layers[i-1] for i in range(len(layers))) * 0
    total_params=0
    din=Ftr.shape[1]
    for L in layers:
        total_params+=din*L
        din=L
    total_params+=din*D_EMB
    return acc,loss,total_params,d

print("="*60); print(" lab24: آیا stacking RF کمک می‌کنه؟"); print("="*60)
print("baseline most-freq=%.1f%%"%(base_acc*100))
print("RF-1layer (D=256): 53.0%% (از lab23)\n")

# تست‌های stacking
print("[1/3] Stacking: همان بودجه‌ی محاسبه (D_RF=256 هر لایه)")
configs=[
    ([128,128],256,'cos'),
    ([128,128,128],256,'cos'),
    ([64,64,64,64],256,'cos'),
    ([128,64],256,'cos'),
    ([256,64],256,'cos'),
]
best_stacked=(None,0)
for ls,D_RF,k in configs:
    t0=time.time()
    acc,loss,par,d=run_rf(ls,D_RF,lam=0.5,kernel=k,seed=100)
    print("  layers=%-20s acc=%.1f%% loss=%.4f params=%d time=%.1fs"%(
        str(ls),acc*100,loss,par,time.time()-t0))
    if acc>best_stacked[1]: best_stacked=(ls,acc)
    render("stacking",configs.index((ls,D_RF,k))+1,len(configs),"")
sys.stdout.write('\n'); sys.stdout.flush()

# تست single layer با ابعاد مختلف
print("\n[2/3] Single layer با ابعاد مختلف")
best_single=(None,0)
for D in [64,128,256,512,1024]:
    acc,loss,par,d=run_rf([D],lam=0.5,kernel='cos',seed=100)
    print("  D_RF=%-6d acc=%.1f%% loss=%.4f params=%d"%(D,acc*100,loss,par))
    if acc>best_single[1]: best_single=(D,acc)

# تست کرنل‌های مختلف
print("\n[3/3] کرنل‌های مختلف (D=512، single layer)")
best_kernel=(None,0)
for k in ['cos','sin','elu','tanh','rbf']:
    acc,loss,par,d=run_rf([512],lam=0.5,kernel=k,seed=100)
    print("  %-6s: acc=%.1f%% loss=%.4f"%(k,acc*100,loss))
    if acc>best_kernel[1]: best_kernel=(k,acc)

print("\n"+"="*60); print(" خلاصه"); print("="*60)
print("  baseline:              %.1f%%"%(base_acc*100))
print("  RF-1layer (D=256):     53.0%%")
print("  بهترین stacking:       %.1f%% layers=%s"%(best_stacked[1]*100,best_stacked[0]))
print("  بهترین single layer:   %.1f%% (D=%s)"%(best_single[1]*100,best_single[0]))
print("  بهترین کرنل:           %s → %.1f%%"%(best_kernel[0],best_kernel[1]*100))
print()
if best_stacked[1]>0.55:
    print("  ✓✓✓ stacking کمک کرد → فرمول روی فرمول جواب می‌ده")
    print("       حرف تو تأیید: چند انتگرال + یک رگرسیون = کل مدل")
elif best_single[1]>0.55:
    print("  ✓ یک لایه RF با D بزرگ کافیه — stacking لازم نیست")
else:
    print("  ~ سقف RF در همین حده")
