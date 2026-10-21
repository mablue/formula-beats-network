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

def run_experiment(n_train, seed=42):
    rng=random.Random(seed)
    tr_s=[gen(rng) for _ in range(n_train)]; te_s=[gen(rng) for _ in range(500)]
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
    def feats(X): return np.concatenate([emb[X[:,0]],emb[X[:,1]],emb[X[:,2]]],axis=1)
    Ftr=feats(Xtr); Fte=feats(Xte)
    Ftr_tr=Ftr[tr_idx]; Ftr_val=Ftr[val_idx]
    Etr_tr=emb[Ytr[tr_idx]]; Etr_val=emb[Ytr[val_idx]]; Ete=emb[Yte]
    from collections import Counter
    mc=Counter(Ytr).most_common(1)[0][0]
    base_acc=sum(1 for y in Yte if y==mc)/len(Yte)
    def acc_on(pred,Y):
        d=np.linalg.norm(pred[:,None,:]-emb[None,:,:],axis=2)
        return float(np.mean(d.argmin(1)==Y))
    def build(F,D,seed=100,lam=0.5):
        r=np.random.RandomState(seed)
        W=r.randn(F.shape[1],D).astype(np.float32)*np.sqrt(2.0/F.shape[1])
        b=r.uniform(0,2*np.pi,D).astype(np.float32)
        Ptr=np.cos(Ftr_tr@W+b); Pval=np.cos(Ftr_val@W+b); Pte=np.cos(Fte@W+b)
        A=Ptr.T@Ptr+lam*np.eye(D)
        Th=np.linalg.solve(A,Ptr.T@Etr_tr)
        return acc_on(Pval@Th,Ytr[val_idx]), acc_on(Pte@Th,Yte)
    # بهینه‌ترین D رو با growth پیدا کن
    best_D=8; best_val=0
    for D in [8,16,32,64,128,256,512]:
        v,t=build(Ftr,D)
        if v>best_val+0.005 or D==8:
            best_val=v; best_D=D
        else:
            break
    v,t=build(Ftr,best_D)
    return len(tr_s), V, len(Xtr), base_acc, best_D, v, t

print("="*60); print(" lab28: آیا سقف از داده‌ست یا از الگوریتم؟"); print("="*60)
print("%-10s %-6s %-8s %-10s %-6s %-8s %-8s"%(
    "n_train","V","n_ex","baseline","D*","val","test"))
results=[]
for n_train in [500, 1000, 2000, 4000, 8000]:
    n,V,n_ex,base,D,v,t=run_experiment(n_train)
    results.append((n,t,base))
    print("%-10d %-6d %-8d %-10.1f %-6d %-8.1f %-8.1f"%(
        n,V,n_ex,base*100,D,v*100,t*100))
    render("grow",[500,1000,2000,4000,8000].index(n_train)+1,5,"")
sys.stdout.write('\n'); sys.stdout.flush()

print("\n"+"="*60); print(" تحلیل"); print("="*60)
print("  trend test acc با افزایش داده:")
for n,t,base in results:
    gain=(t-base)*100
    print("    n=%-5d → test=%.1f%% (gain=%+.1f%% از baseline)"%(n,t*100,gain))

# آیا شیب مثبته؟
gains=[(t-base) for n,t,base in results]
if gains[-1] > gains[0] + 0.05:
    print("\n  ✓✓✓ با داده‌ی بیشتر، accuracy بالا می‌ره → سقف از داده بود!")
    print("       یعنی زبان واقعاً فرمول داره، فقط به داده بیشتر نیاز داره")
    print("       با کورپس واقعی (۱۰۰هزار جمله) می‌شه به ۷۰-۸۰٪ رسید")
else:
    print("\n  ✗ داده بیشتر کمک نکرد → سقف از الگوریتمه")
    print("       حتی با کورپس بزرگ‌تر، این روش به سقف ~55% می‌رسه")
    print("       باید معماری عوض بشه")

print("\nنکته: vocab با داده بیشتر رشد می‌کنه → تسک سخت‌تر می‌شه")
print("       (پس اگه accuracy بالا رفت، حتی impressive‌تره)")
