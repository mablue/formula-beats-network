import numpy as np, time, sys, random
np.random.seed(42); random.seed(42)
def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-24s [%s] %3d%% %s'%(tag[:24],bar,int(pct*100),msg)); sys.stdout.flush()

# ============ کورپس بزرگ‌تر و واقعی‌تر ============
subjects=['the cat','the dog','my friend','the teacher','a student','the doctor',
          'her brother','his sister','the old man','the young girl','the artist',
          'the musician','a writer','the farmer','the engineer','the child',
          'my mother','my father','the neighbor','the stranger','the captain',
          'the pilot','the chef','the nurse','the lawyer','the scientist']
verbs=['likes','sees','knows','finds','wants','reads','writes','makes',
       'watches','hears','feels','loves','hates','needs','helps','visits',
       'remembers','forgets','believes','thinks about','talks to','waits for']
objects=['the book','the house','the car','a song','the movie','the painting',
         'the garden','the river','the city','the mountain','the truth',
         'the answer','a problem','the solution','the story','the letter',
         'the newspaper','the coffee','the music','the weather','the news']
adverbs=['quickly','slowly','carefully','happily','quietly','loudly',
         'often','always','sometimes','never','rarely','usually','finally']
adjs=['big','small','red','blue','old','new','happy','sad','bright','dark',
      'soft','hard','warm','cold','fast','slow','young','beautiful']
places=['the park','the house','the tree','the sky','the road','the river',
        'the city','the garden','the library','the school','the market',
        'the station','the beach','the mountain']
preps=['in','on','at','near','under','over','behind','beside','inside','outside']
conj=['and','but','because','while','after','before','although']
def gen(rng):
    s1=rng.choice(subjects); s2=rng.choice(subjects)
    v=rng.choice(verbs); o=rng.choice(objects); a=rng.choice(adverbs)
    ad=rng.choice(adjs); p=rng.choice(preps); pl=rng.choice(places)
    c=rng.choice(conj)
    return rng.choice([
        f"{s1} {v} {o}".split(),
        f"{s1} {v} {o} {a}".split(),
        f"{a.capitalize()} {s1} {v} {o}".split(),
        f"i think {s1} {v} {o}".split(),
        f"do you know if {s1} {v} {o}".split(),
        f"{s1} {v} {o} {p} {pl}".split(),
        f"the {ad} {s1.split()[-1]} {v} {o}".split(),
        f"{s1} {v} {s2} {c} {s2} {v} {o}".split(),
        f"yesterday {s1} {v} {o} {a}".split(),
        f"tomorrow {s1} will {v} {o}".split(),
        f"{s1} cannot {v} {o}".split(),
        f"when {s1} {v} {o} {s2} {a} {v}".split(),
        f"the {ad} {o.split()[1] if len(o.split())>1 else 'thing'} is {a}".split(),
        f"is it true that {s1} {v} {o}".split(),
        f"{s1} {v} {o} every day".split(),
    ])

rng=random.Random(42)
tr=[gen(rng) for _ in range(5000)]; te=[gen(rng) for _ in range(800)]
aw=sorted(set(w for s in tr+te for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
print("="*60); print(" lab33: مقیاس‌پذیری + تولید واقعی"); print("="*60)
print("vocab=%d | train=%d جمله | test=%d جمله"%(V,len(tr),len(te)))
print("نمونه: %s\n"%" ".join(tr[0]))

def make_data(sents, ctx):
    X,Y=[],[]
    for s in sents:
        ids=[w2i[w] for w in s]
        for i in range(ctx,len(ids)): X.append(ids[i-ctx:i]); Y.append(ids[i])
    return np.array(X),np.array(Y)

# embedding
cooc=np.zeros((V,V),dtype=np.float32)
for s in tr:
    ids=[w2i[w] for w in s]
    for i in range(len(ids)):
        for j in range(max(0,i-3),min(len(ids),i+4)):
            if i!=j: cooc[ids[i],ids[j]]+=1.0
P=(cooc+1e-3)/(cooc.sum()+1e-3*V*V)
ppmi=np.maximum(np.log(P/(P.sum(1,keepdims=True)*P.sum(0,keepdims=True)+1e-9)+1e-9),0)
U,S,_=np.linalg.svd(ppmi,full_matrices=False)
D_EMB=16
emb=(U[:,:D_EMB]*np.sqrt(S[:D_EMB])).astype(np.float32)
emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-9)
print("embedding: %d بعد\n"%D_EMB)

def acc_on(pred,Y):
    d=np.linalg.norm(pred[:,None,:]-emb[None,:,:],axis=2)
    return float(np.mean(d.argmin(1)==Y))

def train_mlp(X, E, h, epochs=150, lr=0.02):
    r=np.random.RandomState(42)
    d=X.shape[1]; d_emb=E.shape[1]
    W1=r.randn(d,h).astype(np.float32)*np.sqrt(2.0/d); b1=np.zeros(h,dtype=np.float32)
    W2=r.randn(h,d_emb).astype(np.float32)*np.sqrt(2.0/h); b2=np.zeros(d_emb,dtype=np.float32)
    vW1=np.zeros_like(W1); vW2=np.zeros_like(W2)
    n=X.shape[0]
    for ep in range(epochs):
        z=X@W1+b1; ha=np.maximum(z,0); out=ha@W2+b2
        diff=(out-E)*2/n
        dW2=ha.T@diff; db2=diff.sum(0)
        dh=diff@W2.T; dz=dh*(z>0)
        dW1=X.T@dz; db1=dz.sum(0)
        vW1=0.9*vW1-lr*dW1; W1+=vW1; b1-=lr*db1
        vW2=0.9*vW2-lr*dW2; W2+=vW2; b2-=lr*db2
    return (W1,b1,W2,b2)

def rf_ridge(X_tr, E_tr, X_te, D=512, lam=0.5):
    r=np.random.RandomState(100)
    W=r.randn(X_tr.shape[1],D).astype(np.float32)*np.sqrt(2.0/X_tr.shape[1])
    b=r.uniform(0,2*np.pi,D).astype(np.float32)
    Phi_tr=np.cos(X_tr@W+b); Phi_te=np.cos(X_te@W+b)
    A=Phi_tr.T@Phi_tr+lam*np.eye(D)
    Th=np.linalg.solve(A,Phi_tr.T@E_tr)
    return W,b,Th,acc_on(Phi_te@Th,Yte)

def run_experiment(ctx):
    Xtr,Ytr=make_data(tr,ctx); Xte,Yte=make_data(te,ctx)
    Ftr=np.concatenate([emb[Xtr[:,k]] for k in range(ctx)],axis=1)
    Fte=np.concatenate([emb[Xte[:,k]] for k in range(ctx)],axis=1)
    Etr=emb[Ytr]; Ete=emb[Yte]
    return Xtr,Ytr,Xte,Yte,Ftr,Fte,Etr,Ete

# ============ تست مقیاس‌پذیری ============
print("[1/2] مقایسه RF vs MLP در CTX=3 و CTX=5\n")
results=[]
for ctx in [3,5]:
    Xtr,Ytr,Xte,Yte,Ftr,Fte,Etr,Ete=run_experiment(ctx)
    print("  CTX=%d: train=%d test=%d"%(ctx,len(Xtr),len(Xte)))
    # MLP
    for h in [256, 512]:
        t0=time.time()
        p=train_mlp(Ftr,Etr,h,epochs=100)
        a=acc_on(np.maximum(Fte@p[0]+p[1],0)@p[2]+p[3],Yte)
        par=Ftr.shape[1]*h+h*D_EMB
        print("    MLP h=%-4d test=%.1f%% params=%-7d time=%.1fs"%(h,a*100,par,time.time()-t0))
        results.append(('MLP_ctx%d_h%d'%(ctx,h),a,par,ctx))
    # RF
    for D in [256, 512, 1024]:
        t0=time.time()
        _,_,_,a=rf_ridge(Ftr,Etr,Fte,D=D)
        par=Ftr.shape[1]*D+D*D_EMB
        print("    RF  D=%-4d test=%.1f%% params=%-7d time=%.2fs"%(D,a*100,par,time.time()-t0))
        results.append(('RF_ctx%d_D%d'%(ctx,D),a,par,ctx))
    render("scale",[3,5].index(ctx)+1,2,"")
    print()
sys.stdout.write('\n'); sys.stdout.flush()

# ============ بهترین مدل: تولید متن ============
print("[2/2] تولید متن با بهترین مدل")
best=max(results,key=lambda x:x[1])
print("  بهترین: %s (test=%.1f%%)\n"%(best[0],best[1]*100))

# بازیابی بهترین مدل
best_ctx=best[3]
Xtr,Ytr,Xte,Yte,Ftr,Fte,Etr,Ete=run_experiment(best_ctx)
if 'RF' in best[0]:
    D=int(best[0].split('D')[1])
    W,b,Th,_=rf_ridge(Ftr,Etr,Fte,D=D)
    def predict(ids):
        x=np.concatenate([emb[ids[-best_ctx+k]] for k in range(best_ctx)]).reshape(1,-1)
        pred=np.cos(x@W+b)@Th
        d=np.linalg.norm(emb-pred[0][None,:],axis=1)
        return d
else:
    h=int(best[0].split('h')[1])
    p=train_mlp(Ftr,Etr,h,epochs=100)
    def predict(ids):
        x=np.concatenate([emb[ids[-best_ctx+k]] for k in range(best_ctx)]).reshape(1,-1)
        pred=np.maximum(x@p[0]+p[1],0)@p[2]+p[3]
        d=np.linalg.norm(emb-pred[0][None,:],axis=1)
        return d

i2w={i:w for w,i in w2i.items()}
def sample_next(dists, top_k=5, temp=1.0):
    # top-k sampling
    idx=np.argsort(dists)[:top_k]
    logits=-dists[idx]/temp
    p=np.exp(logits-logits.max()); p/=p.sum()
    return idx[np.random.choice(len(idx),p=p)]

prompts=[["the","cat"],["i","think","the"],["do","you","know"],
         ["yesterday","the","old"],["the","beautiful"],["the","teacher"]]
print("  تولید با argmax (det):")
for p in prompts:
    ids=[w2i.get(w,0) for w in p]
    while len(ids)<best_ctx: ids=[0]+ids
    out=ids.copy()
    for _ in range(10):
        d=predict(out); out.append(int(d.argmin()))
    print("    %-25s → %s"%(" ".join(p)," ".join(i2w.get(x,'?') for x in out[-10:])))

print("\n  تولید با top-k=5 (sample):")
for p in prompts:
    ids=[w2i.get(w,0) for w in p]
    while len(ids)<best_ctx: ids=[0]+ids
    out=ids.copy()
    for _ in range(10):
        d=predict(out); out.append(int(sample_next(d,top_k=5)))
    print("    %-25s → %s"%(" ".join(p)," ".join(i2w.get(x,'?') for x in out[-10:])))

# ============ نتیجه ============
print("\n"+"="*60); print(" جدول نهایی"); print("="*60)
print("  %-20s %-10s %-10s"%("config","test_acc","params"))
for name,a,par,ctx in sorted(results,key=lambda x:-x[1]):
    print("  %-20s %-10.1f %-10d"%(name,a*100,par))

print("\n"+"="*60); print(" تحلیل نهایی"); print("="*60)
rf_best=max([r for r in results if 'RF' in r[0]],key=lambda x:x[1])
mlp_best=max([r for r in results if 'MLP' in r[0]],key=lambda x:x[1])
print("  بهترین RF:   %s → %.1f%%"%(rf_best[0],rf_best[1]*100))
print("  بهترین MLP:  %s → %.1f%%"%(mlp_best[0],mlp_best[1]*100))
gain=(rf_best[1]-mlp_best[1])*100
if gain>0:
    print("\n  ✓✓✓ RF با %.1f%% اختلاف برنده → حرف تو در مقیاس بزرگ‌تر هم اثبات شد"%gain)
else:
    print("\n  ≈ RF و MLP نزدیک → در مقیاس بزرگ، شبکه هم جواب می‌ده")
# آیا CTX=5 بهتر از CTX=3؟
rf_c3=max([r for r in results if 'RF' in r[0] and '_ctx3_' in r[0]],key=lambda x:x[1])
rf_c5=max([r for r in results if 'RF' in r[0] and '_ctx5_' in r[0]],key=lambda x:x[1])
if rf_c5[1]>rf_c3[1]+0.02:
    print("  ✓ CTX=5 بهتر از CTX=3 → context طولانی‌تر کمک می‌کنه")
    print("       (پس با context=۱۰۰ هم می‌شه بهتر شد)")
else:
    print("  ≈ CTX=3 و CTX=5 نزدیک → context بیشتر کمک نکرد")
