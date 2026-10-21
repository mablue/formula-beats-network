import random, math, sys, time
from collections import Counter

# چک سریع numpy
HAS_NP = False
try:
    import numpy as np
    _ = np.ones((2,2)) @ np.ones((2,2))
    HAS_NP = True
    print("numpy: ✓ کار می‌کند")
except: print("numpy: ✗ خراب (pure python استفاده می‌شود)")

nouns=['cat','dog','bird','fish','tree','house','car','book','man','woman']
verbs=['sat','ate','saw','sang','ran','read','looked','jumped']
adjs=['big','small','red','blue','old','new','happy']
places=['park','house','tree','sky','road']
def gen(rng):
    return rng.choice([
        "the {n} {v} the {n}","the {a} {n} {v}",
        "a {a} {n} {v} the {n}","the {n} {v} in the {p}",
    ]).format(n=rng.choice(nouns),v=rng.choice(verbs),a=rng.choice(adjs),p=rng.choice(places)).split()
rng=random.Random(42)
tr_s=[gen(rng) for _ in range(160)]; te_s=[gen(rng) for _ in range(40)]
aw=sorted(set(w for s in tr_s+te_s for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
def mxy(ss,ctx=2):
    X,Y=[],[]
    for s in ss:
        ids=[w2i[w] for w in s]
        for i in range(ctx,len(ids)): X.append(ids[i-ctx:i]); Y.append(ids[i])
    return X,Y
Xtr,Ytr=mxy(tr_s); Xte,Yte=mxy(te_s)
def oh(c):
    v=[0.0]*(V*len(c))
    for k,ci in enumerate(c): v[k*V+ci]=1.0
    return v
DI=V*2; DO=V   # ← اصلاح: do=V نه len(set)
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)

class RF:
    def __init__(s,di,D,seed):
        r=random.Random(seed)
        s.W=[[r.gauss(0,math.sqrt(2.0/di)) for _ in range(di)] for _ in range(D)]
        s.b=[r.uniform(0,2*math.pi) for _ in range(D)]
        s.D=D; s.di=di
    def feats(s,x):
        if HAS_NP:
            W=np.array(s.W); b=np.array(s.b); xa=np.array(x)
            return list(np.cos(W@xa+b))
        return [math.cos(sum(s.W[d][i]*x[i] for i in range(s.di))+s.b[d]) for d in range(s.D)]

def ridge_solve(Phi,Y,do,lam=1e-2):
    """حل ridge با numpy اگر موجود، وگرنه pure"""
    n=len(Phi); D=len(Phi[0])
    if HAS_NP:
        P=np.array(Phi); Y_arr=np.zeros((n,do))
        for k,y in enumerate(Y): Y_arr[k,y]=1.0
        A=P.T@P+lam*np.eye(D)
        Th=np.linalg.solve(A, P.T@Y_arr)
        return Th
    A=[[sum(Phi[k][i]*Phi[k][j] for k in range(n))+(lam if i==j else 0) for j in range(D)] for i in range(D)]
    Th=[[0.0]*do for _ in range(D)]
    for cls in range(do):
        b=[sum(Phi[k][i]*(1.0 if Y[k]==cls else 0.0) for k in range(n)) for i in range(D)]
        A2=[A[i][:]+[b[i]] for i in range(D)]
        for i in range(D):
            piv=max(range(i,D),key=lambda r:abs(A2[r][i]))
            A2[i],A2[piv]=A2[piv],A2[i]
            if abs(A2[i][i])<1e-12: continue
            for r in range(D):
                if r==i: continue
                f=A2[r][i]/A2[i][i]
                for c in range(i,D+1): A2[r][c]-=f*A2[i][c]
        for i in range(D): Th[i][cls]=A2[i][D]/A2[i][i] if abs(A2[i][i])>1e-12 else 0.0
    return Th

def run_stacked(Xtr,Ytr,Xte,Yte,layers,lam=1e-2):
    curXtr=[oh(x) for x in Xtr]; curXte=[oh(x) for x in Xte]
    for D in layers:
        rf=RF(len(curXtr[0]),D,seed=D)
        curXtr=[rf.feats(x) for x in curXtr]
        curXte=[rf.feats(x) for x in curXte]
    D=len(curXtr[0])
    Th=ridge_solve(curXtr,Ytr,DO,lam)
    c=0; L=0.0
    for x,y in zip(curXte,Yte):
        if HAS_NP:
            o=list(np.array(x)@Th)
        else:
            o=[sum(x[i]*Th[i][j] for i in range(D)) for j in range(DO)]
        m=max(o); e=[math.exp(v-m) for v in o]; ss=sum(e); p=[ei/ss for ei in e]
        if p.index(max(p))==y: c+=1
        L+=-math.log(p[y]+1e-9)
    return c/len(Xte),L/len(Xte)

print("="*60); print(" lab18b: RF چندلایه (باگ fix شده)"); print("="*60)
print("Vocab=%d Train=%d Test=%d | Most-freq=%.1f%%\n"%(V,len(Xtr),len(Xte),base_acc*100))

layersets=[[64,64],[128,128],[64,128,64],[128,64],[256,256]]
results={}
t0=time.time()
for i,ls in enumerate(layersets):
    print("[%d/%d] layers=%s"%(i+1,len(layersets),ls))
    a,l=run_stacked(Xtr,Ytr,Xte,Yte,ls)
    results[tuple(ls)]=(a,l)
    print("      acc=%.1f%% loss=%.3f"%(a*100,l))
print("\nزمان کل: %.1f ثانیه"%(time.time()-t0))

print("\n"+"="*60); print(" مقایسه نهایی"); print("="*60)
print("  Most-freq:     %.1f%%"%(base_acc*100))
print("  MLP (fixed):   35.0%%  (از lab18 قبلی)")
print("  RF-1layer:     33.3%%  (D=256)")
best_ls=max(results,key=lambda k:results[k][0])
print("  RF-stacked:    %.1f%%  layers=%s"%(results[best_ls][0]*100,list(best_ls)))
print()
mlp_acc=0.35; rf1_acc=0.333; best_a=results[best_ls][0]
if best_a>mlp_acc+0.02:
    print("  ✓✓✓ RF چندلایه از MLP بهتره → فرمول روی فرمول جواب داد")
elif best_a>rf1_acc+0.02:
    print("  ✓ RF چندلایه از RF تک‌لایه بهتره → عمق در فرمول کمک می‌کنه")
elif best_a>mlp_acc-0.02:
    print("  ≈ همه نزدیک (35-33%) → تسک سقفش همینه")
else:
    print("  ✗ RF-stacked ضعیف‌تر")
