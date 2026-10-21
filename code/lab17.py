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
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)
def render(tag,ep,total,loss,acc,w=15):
    pct=ep/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    line='  %-22s [%s] %3d%% loss=%.3f acc=%.0f%%'%(tag[:22],bar,int(pct*100),loss,acc*100)
    sys.stdout.write('\r'+line.ljust(80)); sys.stdout.flush()

class MLP:
    def __init__(s,di,dh,do,seed=42):
        r=random.Random(seed)
        s.W1=[[r.gauss(0,1/math.sqrt(di)) for _ in range(dh)] for _ in range(di)]
        s.b1=[0.0]*dh; s.W2=[[r.gauss(0,1/math.sqrt(dh)) for _ in range(do)] for _ in range(dh)]; s.b2=[0.0]*do
        s.vW1=[[0.0]*dh for _ in range(di)]; s.vW2=[[0.0]*do for _ in range(dh)]
    def fwd(s,x):
        z=[sum(x[i]*s.W1[i][j] for i in range(len(x)))+s.b1[j] for j in range(len(s.b1))]
        h=[v if v>0 else 0.0 for v in z]
        o=[sum(h[i]*s.W2[i][j] for i in range(len(h)))+s.b2[j] for j in range(len(s.b2))]
        return o,z,h
    def step(s,x,y,z,h,lr,mom=0.9):
        o,z2,h2=s.fwd(x); m=max(o); e=[math.exp(v-m) for v in o]; ss=sum(e); p=[ei/ss for ei in e]
        dy=p[:]; dy[y]-=1.0
        dW2=[[h[i]*dy[j] for j in range(len(dy))] for i in range(len(h))]
        dh=[sum(s.W2[i][j]*dy[j] for j in range(len(dy))) for i in range(len(h))]
        dz=[dh[i]*(1.0 if z[i]>0 else 0.0) for i in range(len(z))]
        dW1=[[x[i]*dz[j] for j in range(len(dz))] for i in range(len(x))]
        for i in range(len(s.W1)):
            for j in range(len(s.W1[0])):
                s.vW1[i][j]=mom*s.vW1[i][j]-lr*dW1[i][j]; s.W1[i][j]+=s.vW1[i][j]
        for j in range(len(s.b1)): s.b1[j]-=lr*dz[j]
        for i in range(len(s.W2)):
            for j in range(len(s.W2[0])):
                s.vW2[i][j]=mom*s.vW2[i][j]-lr*dW2[i][j]; s.W2[i][j]+=s.vW2[i][j]
        for j in range(len(s.b2)): s.b2[j]-=lr*dy[j]

# Random Features: phi(x) = cos(Wx + b) با W,b تصادفی و ثابت
class RF:
    def __init__(s,di,D,seed=42):
        r=random.Random(seed)
        s.W=[[r.gauss(0,1/math.sqrt(di)) for _ in range(di)] for _ in range(D)]
        s.b=[r.uniform(0,2*math.pi) for _ in range(D)]
        s.D=D
    def feats(s,x):
        return [math.cos(sum(s.W[d][i]*x[i] for i in range(len(x)))+s.b[d]) for d in range(s.D)]
    def solve(s,X,Y,do):
        n=len(X); D=s.D
        Phi=[s.feats(x) for x in X]
        # ridge: (Phi^T Phi + λI)^-1 Phi^T Y
        lam=1e-2
        A=[[sum(Phi[k][i]*Phi[k][j] for k in range(n))+(lam if i==j else 0) for j in range(D)] for i in range(D)]
        s.Th=[[0.0]*do for _ in range(D)]
        for cls in range(do):
            b=[sum(Phi[k][i]*(1.0 if Y[k]==cls else 0.0) for k in range(n)) for i in range(D)]
            # گاوس-جوردن
            A2=[row[:]+[b[i]] for i,row in enumerate(A)]
            for i in range(D):
                piv=max(range(i,D),key=lambda r:abs(A2[r][i]))
                A2[i],A2[piv]=A2[piv],A2[i]
                if abs(A2[i][i])<1e-12: continue
                for r in range(D):
                    if r==i: continue
                    f=A2[r][i]/A2[i][i]
                    for c in range(i,D+1): A2[r][c]-=f*A2[i][c]
            for i in range(D):
                s.Th[i][cls]=A2[i][D]/A2[i][i] if abs(A2[i][i])>1e-12 else 0.0
    def fwd(s,x):
        f=s.feats(x)
        return [sum(f[i]*s.Th[i][j] for i in range(s.D)) for j in range(len(s.Th[0]))]
    def prob(s,x):
        o=s.fwd(x); m=max(o); e=[math.exp(v-m) for v in o]; ss=sum(e)
        return [ei/ss for ei in e]

def ev(m,X,Y,kind='mlp'):
    c=0; L=0.0
    for xi,y in zip(X,Y):
        x=oh(xi)
        if kind=='mlp':
            o=m.fwd(x)[0]; mm=max(o); e=[math.exp(v-mm) for v in o]; ss=sum(e); p=[ei/ss for ei in e]
        else: p=m.prob(x)
        if p.index(max(p))==y: c+=1
        L+=-math.log(p[y]+1e-9)
    return c/len(X),L/len(X)

print("="*60); print(" lab17: Random Features — کرنل به‌عنوان فرمول زبان"); print("="*60)
print("Vocab=%d Train=%d Test=%d | Most-freq=%.1f%%\n"%(V,len(Xtr),len(Xte),base_acc*100))
di=V*2; do=V

print("[1/2] MLP (h=32, lr=0.05, mom) — baseline")
mlp=MLP(di,32,do)
for ep in range(300):
    for xi,y in zip(Xtr,Ytr):
        x=oh(xi); _,z,h=mlp.fwd(x); mlp.step(x,y,z,h,0.05)
    if (ep+1)%50==0 or ep==0:
        a,l=ev(mlp,Xte,Yte); render("MLP",ep+1,300,l,a)
sys.stdout.write('\n'); sys.stdout.flush()
ma,ml=ev(mlp,Xte,Yte)

for D in [64, 128, 256]:
    print("\n[RF D=%d] Random Features + ridge (بدون آموزش تکراری)"%D)
    rf=RF(di,D,seed=42)
    rf.solve([oh(x) for x in Xtr],Ytr,do)
    a,l=ev(rf,Xte,Yte,kind='rf')
    print("  D=%-4d: acc=%.1f%% loss=%.3f params=%d"%(D,a*100,l,D*do))

# بهترین RF
rf_best=RF(di,256,seed=42); rf_best.solve([oh(x) for x in Xtr],Ytr,do)
fa,fl=ev(rf_best,Xte,Yte,kind='rf')

print("\n"+"="*60); print(" نتیجه"); print("="*60)
print("  Most-freq:  %.1f%%"%(base_acc*100))
print("  MLP:        %.1f%%  loss=%.3f"%(ma*100,ml))
print("  RF(D=256):  %.1f%%  loss=%.3f"%(fa*100,fl))
print()
if fa>=ma-0.03:
    print("  ✓✓✓ RF ≈ MLP → فرمول خطی-کرنل روی زبان کافیه!")
    print("      یعنی: زبان هم با یک فرمول خطی در فضای رندوم بالا حل می‌شه")
elif fa>=ma-0.10:
    print("  ~ RF نزدیکه ولی MLP بهتره")
else:
    print("  ✗ RF ضعیف‌تره → فرمول خطی برای زبان کافی نیست")
    print("      باید فرمول غیرخطی پیدا کنیم (گراف، تانسور، ...)")
