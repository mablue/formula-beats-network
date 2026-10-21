import random, math, sys
from collections import Counter
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
DI=V*2
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)
def render(tag,ep,total,loss,acc,w=15):
    pct=ep/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    line='  %-24s [%s] %3d%% loss=%.3f acc=%.0f%%'%(tag[:24],bar,int(pct*100),loss,acc*100)
    sys.stdout.write('\r'+line.ljust(80)); sys.stdout.flush()

class MLP:
    def __init__(s,di,dh,do,seed=42):
        r=random.Random(seed)
        s.W1=[[r.gauss(0,1.0/math.sqrt(di)) for _ in range(dh)] for _ in range(di)]
        s.b1=[0.0]*dh
        s.W2=[[r.gauss(0,1.0/math.sqrt(dh)) for _ in range(do)] for _ in range(dh)]
        s.b2=[0.0]*do
        s.vW1=[[0.0]*dh for _ in range(di)]; s.vW2=[[0.0]*do for _ in range(dh)]
    def fwd(s,x):
        z=[sum(x[i]*s.W1[i][j] for i in range(len(x)))+s.b1[j] for j in range(len(s.b1))]
        h=[max(0.0,v) for v in z]
        o=[sum(h[i]*s.W2[i][j] for i in range(len(h)))+s.b2[j] for j in range(len(s.b2))]
        return o,z,h
    def step(s,x,y,z,h,lr,mom=0.9,wd=0.0):
        o,z2,h2=s.fwd(x); m=max(o); e=[math.exp(v-m) for v in o]; ss=sum(e); p=[ei/ss for ei in e]
        dy=p[:]; dy[y]-=1.0
        dW2=[[h[i]*dy[j] for j in range(len(dy))] for i in range(len(h))]
        dh=[sum(s.W2[i][j]*dy[j] for j in range(len(dy))) for i in range(len(h))]
        dz=[dh[i]*(1.0 if z[i]>0 else 0.0) for i in range(len(z))]
        dW1=[[x[i]*dz[j] for j in range(len(dz))] for i in range(len(x))]
        for i in range(len(s.W1)):
            for j in range(len(s.W1[0])):
                g=dW1[i][j]+wd*s.W1[i][j]
                s.vW1[i][j]=mom*s.vW1[i][j]-lr*g; s.W1[i][j]+=s.vW1[i][j]
        for j in range(len(s.b1)): s.b1[j]-=lr*dz[j]
        for i in range(len(s.W2)):
            for j in range(len(s.W2[0])):
                g=dW2[i][j]+wd*s.W2[i][j]
                s.vW2[i][j]=mom*s.vW2[i][j]-lr*g; s.W2[i][j]+=s.vW2[i][j]
        for j in range(len(s.b2)): s.b2[j]-=lr*dy[j]

def ev_mlp(m,X,Y):
    c=0; L=0.0
    for xi,y in zip(X,Y):
        x=oh(xi); o=m.fwd(x)[0]; mm=max(o); e=[math.exp(v-mm) for v in o]; ss=sum(e); p=[ei/ss for ei in e]
        if p.index(max(p))==y: c+=1
        L+=-math.log(p[y]+1e-9)
    return c/len(X),L/len(X)

def train_mlp(X,Y,dh,epochs,lr,tag,wd=1e-4):
    m=MLP(DI,dh,V)
    r=random.Random(1)
    for ep in range(epochs):
        idx=list(range(len(X))); r.shuffle(idx)
        for ii in idx:
            x=oh(X[ii]); _,z,h=m.fwd(x); m.step(x,Y[ii],z,h,lr,wd=wd)
        if (ep+1)%30==0 or ep==0:
            a,l=ev_mlp(m,Xte,Yte); render(tag,ep+1,epochs,l,a)
    sys.stdout.write('\n'); sys.stdout.flush()
    return m

class RF:
    def __init__(s,di,D,seed):
        r=random.Random(seed)
        s.W=[[r.gauss(0,math.sqrt(2.0/di)) for _ in range(di)] for _ in range(D)]
        s.b=[r.uniform(0,2*math.pi) for _ in range(D)]
        s.D=D; s.di=di
    def feats(s,x):
        return [math.cos(sum(s.W[d][i]*x[i] for i in range(s.di))+s.b[d]) for d in range(s.D)]
    def solve(s,Phi_raw,Y,do,lam=1e-2):
        Phi=[s.feats(x) for x in Phi_raw]
        n=len(Phi); D=s.D
        A=[[sum(Phi[k][i]*Phi[k][j] for k in range(n))+(lam if i==j else 0) for j in range(D)] for i in range(D)]
        s.Th=[[0.0]*do for _ in range(D)]
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
            for i in range(D): s.Th[i][cls]=A2[i][D]/A2[i][i] if abs(A2[i][i])>1e-12 else 0.0
    def prob(s,x):
        f=s.feats(x)
        o=[sum(f[i]*s.Th[i][j] for i in range(s.D)) for j in range(len(s.Th[0]))]
        m=max(o); e=[math.exp(v-m) for v in o]; ss=sum(e)
        return [ei/ss for ei in e]

def ev_rf(m,X,Y):
    c=0; L=0.0
    for xi,y in zip(X,Y):
        p=m.prob(oh(xi))
        if p.index(max(p))==y: c+=1
        L+=-math.log(p[y]+1e-9)
    return c/len(X),L/len(X)

def run_stacked_rf(Xtr,Ytr,Xte,Yte,layers,lam=1e-2):
    curXtr=[oh(x) for x in Xtr]; curXte=[oh(x) for x in Xte]
    for D in layers:
        rf=RF(len(curXtr[0]),D,seed=D)
        curXtr=[rf.feats(x) for x in curXtr]; curXte=[rf.feats(x) for x in curXte]
    D=len(curXtr[0]); do=len(set(Ytr))
    n=len(curXtr)
    A=[[sum(curXtr[k][i]*curXtr[k][j] for k in range(n))+(lam if i==j else 0) for j in range(D)] for i in range(D)]
    Th=[[0.0]*do for _ in range(D)]
    for cls in range(do):
        b=[sum(curXtr[k][i]*(1.0 if Ytr[k]==cls else 0.0) for k in range(n)) for i in range(D)]
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
    c=0; L=0.0
    for x,y in zip(curXte,Yte):
        o=[sum(x[i]*Th[i][j] for i in range(D)) for j in range(do)]
        m=max(o); e=[math.exp(v-m) for v in o]; ss=sum(e); p=[ei/ss for ei in e]
        if p.index(max(p))==y: c+=1
        L+=-math.log(p[y]+1e-9)
    return c/len(Xte),L/len(Xte)

print("="*60); print(" lab18-fix: MLP + RF چندلایه"); print("="*60)
print("Vocab=%d DI=%d Train=%d Test=%d | Most-freq=%.1f%%\n"%(V,DI,len(Xtr),len(Xte),base_acc*100))

print("[1/3] MLP lr=0.01 wd=1e-3")
m=train_mlp(Xtr,Ytr,32,200,0.01,"MLP lr=0.01 wd",wd=1e-3)
ma,ml=ev_mlp(m,Xte,Yte)
print("  → acc=%.1f%% loss=%.3f"%(ma*100,ml))

print("\n[2/3] RF تک‌لایه")
accs={}
for D in [64,128,256]:
    rf=RF(DI,D,seed=42)
    rf.solve([oh(x) for x in Xtr],Ytr,V)
    a,l=ev_rf(rf,Xte,Yte); accs[D]=(a,l)
    print("  D=%-4d acc=%.1f%% loss=%.3f"%(D,a*100,l))

print("\n[3/3] RF چندلایه")
layersets=[[64,64],[128,128],[64,128,64],[128,64]]
stacked_res={}
for ls in layersets:
    a,l=run_stacked_rf(Xtr,Ytr,Xte,Yte,ls)
    stacked_res[tuple(ls)]=(a,l)
    print("  layers=%-15s acc=%.1f%% loss=%.3f"%(str(ls),a*100,l))

print("\n"+"="*60); print(" خلاصه"); print("="*60)
print("  Most-freq:        %.1f%%"%(base_acc*100))
print("  MLP (fixed):      %.1f%%  loss=%.3f"%(ma*100,ml))
print("  RF-1layer(D=256): %.1f%%  loss=%.3f"%(accs[256][0]*100,accs[256][1]))
best_ls=max(stacked_res,key=lambda k:stacked_res[k][0])
best_a,best_l=stacked_res[best_ls]
print("  RF-stacked best:  %.1f%%  loss=%.3f  layers=%s"%(best_a*100,best_l,list(best_ls)))
print()
if best_a>ma+0.02:
    print("  ✓✓✓ RF چندلایه از MLP بهتره → فرمول‌های چیده‌شده جواب می‌دن")
elif best_a>ma-0.02:
    print("  ✓ RF چندلایه ≈ MLP → هر دو نزدیک سقف")
else:
    print("  ✗ MLP بهتره")
