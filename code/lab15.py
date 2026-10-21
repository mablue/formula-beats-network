import random, math, sys
from collections import Counter

nouns = ['cat','dog','bird','fish','tree','house','car','book','man','woman']
verbs = ['sat','ate','saw','sang','ran','read','looked','jumped']
adjs  = ['big','small','red','blue','old','new','happy']
places= ['park','house','tree','sky','road']

def gen(rng):
    t = rng.choice([
        "the {n} {v} the {n}",
        "the {a} {n} {v}",
        "a {a} {n} {v} the {n}",
        "the {n} {v} in the {p}",
    ])
    return t.format(n=rng.choice(nouns),v=rng.choice(verbs),a=rng.choice(adjs),p=rng.choice(places)).split()

rng = random.Random(42)
tr_s = [gen(rng) for _ in range(160)]
te_s = [gen(rng) for _ in range(40)]

all_words = sorted(set(w for s in tr_s+te_s for w in s))
word2id = {w:i for i,w in enumerate(all_words)}
V = len(all_words)

def make_xy(sents, ctx=2):
    X, Y = [], []
    for s in sents:
        ids=[word2id[w] for w in s]
        for i in range(ctx,len(ids)):
            X.append(ids[i-ctx:i]); Y.append(ids[i])
    return X, Y

X_tr,Y_tr = make_xy(tr_s); X_te,Y_te = make_xy(te_s)

def onehot(ctx):
    v=[0.0]*(V*len(ctx))
    for c,cid in enumerate(ctx): v[c*V+cid]=1.0
    return v

most_common = Counter(Y_tr).most_common(1)[0][0]
base_acc = sum(1 for y in Y_te if y==most_common)/len(Y_te)

def render(tag,ep,total,loss,acc,w=15):
    pct=ep/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    line='  %-22s [%s] %3d%%  loss=%.3f acc=%.0f%%'%(tag[:22],bar,int(pct*100),loss,acc*100)
    line=line.ljust(80)
    sys.stdout.write('\r'+line); sys.stdout.flush()

class MLP:
    def __init__(self,di,dh,do,seed=42):
        r=random.Random(seed)
        self.W1=[[r.gauss(0,1/math.sqrt(di)) for _ in range(dh)] for _ in range(di)]
        self.b1=[0.0]*dh
        self.W2=[[r.gauss(0,1/math.sqrt(dh)) for _ in range(do)] for _ in range(dh)]
        self.b2=[0.0]*do
    def fwd(self,x):
        z=[sum(x[i]*self.W1[i][j] for i in range(len(x)))+self.b1[j] for j in range(len(self.b1))]
        h=[v if v>0 else 0.0 for v in z]
        o=[sum(h[i]*self.W2[i][j] for i in range(len(h)))+self.b2[j] for j in range(len(self.b2))]
        return o,z,h
    def step(self,x,y,z,h,lr):
        o,z2,h2=self.fwd(x)
        m=max(o); e=[math.exp(v-m) for v in o]; s=sum(e); p=[ei/s for ei in e]
        dy=p[:]; dy[y]-=1.0
        dW2=[[h[i]*dy[j] for j in range(len(dy))] for i in range(len(h))]
        dh=[sum(self.W2[i][j]*dy[j] for j in range(len(dy))) for i in range(len(h))]
        dz=[dh[i]*(1.0 if z[i]>0 else 0.0) for i in range(len(z))]
        dW1=[[x[i]*dz[j] for j in range(len(dz))] for i in range(len(x))]
        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])): self.W1[i][j]-=lr*dW1[i][j]
        for j in range(len(self.b1)): self.b1[j]-=lr*dz[j]
        for i in range(len(self.W2)):
            for j in range(len(self.W2[0])): self.W2[i][j]-=lr*dW2[i][j]
        for j in range(len(self.b2)): self.b2[j]-=lr*dy[j]

class Fuzzy:
    def __init__(self,di,do,K,X_init,seed=42):
        r=random.Random(seed)
        self.c=[onehot(X_init[i%len(X_init)]) for i in range(K)]
        for k in range(K):
            for i in range(len(self.c[k])): self.c[k][i]+=r.gauss(0,0.05)
        self.s=[2.0]*K
        self.b=[[r.gauss(0,0.1) for _ in range(do)] for _ in range(K)]
    def fwd(self,x):
        mus=[]
        for k in range(len(self.c)):
            d2=sum((x[i]-self.c[k][i])**2 for i in range(len(x)))
            mus.append(math.exp(-d2/(2*self.s[k]**2+1e-9)))
        Z=sum(mus)+1e-9
        o=[sum(mus[k]*self.b[k][j] for k in range(len(mus)))/Z for j in range(len(self.b[0]))]
        return o,mus,Z
    def step(self,x,y,lr):
        o,mus,Z=self.fwd(x)
        m=max(o); e=[math.exp(v-m) for v in o]; s=sum(e); p=[ei/s for ei in e]
        dy=p[:]; dy[y]-=1.0
        for k in range(len(self.c)):
            for j in range(len(self.b[0])): self.b[k][j]-=lr*dy[j]*mus[k]/Z
            dm=sum(dy[j]*(self.b[k][j]-o[j]) for j in range(len(dy)))/Z
            d2=sum((x[i]-self.c[k][i])**2 for i in range(len(x)))
            for i in range(len(self.c[k])):
                dmu=mus[k]*(x[i]-self.c[k][i])/(self.s[k]**2+1e-9)
                self.c[k][i]-=lr*dm*dmu
            dmu_s=mus[k]*d2/(self.s[k]**3+1e-9)
            self.s[k]=max(0.3,self.s[k]-lr*dm*dmu_s)

def eval_m(m,X,Y):
    c=0; L=0.0
    for xi,y in zip(X,Y):
        x=onehot(xi); o=m.fwd(x)[0]
        mm=max(o); e=[math.exp(v-mm) for v in o]; s=sum(e); p=[ei/s for ei in e]
        if p.index(max(p))==y: c+=1
        L+=-math.log(p[y]+1e-9)
    return c/len(X), L/len(X)

print("="*60); print(" lab15: زبان با داده‌ی کافی"); print("="*60)
print("Vocab=%d | Train=%d | Test=%d" % (V,len(X_tr),len(X_te)))
print("Random baseline: %.1f%% | Most-frequent: %.1f%%\n" % (100/V, base_acc*100))

di=V*2; dh=32; do=V
mlp_params=di*dh+dh+dh*do+do
K=32
fz_params=K*(di+1+do)

print("[1/2] MLP (h=%d, params≈%d)" % (dh,mlp_params))
mlp=MLP(di,dh,do)
for ep in range(200):
    for xi,y in zip(X_tr,Y_tr):
        x=onehot(xi); _,z,h=mlp.fwd(x); mlp.step(x,y,z,h,0.3)
    if (ep+1)%25==0 or ep==0:
        a,l=eval_m(mlp,X_te,Y_te); render("MLP",ep+1,200,l,a)
sys.stdout.write('\n'); sys.stdout.flush()
mlp_acc,mlp_loss=eval_m(mlp,X_te,Y_te)

print("\n[2/2] Fuzzy TSK (K=%d, params≈%d)" % (K,fz_params))
fz=Fuzzy(di,do,K,X_tr)
for ep in range(200):
    for xi,y in zip(X_tr,Y_tr):
        x=onehot(xi); fz.step(x,y,0.3)
    if (ep+1)%25==0 or ep==0:
        a,l=eval_m(fz,X_te,Y_te); render("Fuzzy K=%d"%K,ep+1,200,l,a)
sys.stdout.write('\n'); sys.stdout.flush()
fz_acc,fz_loss=eval_m(fz,X_te,Y_te)

print("\n"+"="*60); print(" نتیجه"); print("="*60)
print("  Random:       %.1f%%" % (100/V))
print("  Most-freq:    %.1f%%" % (base_acc*100))
print("  MLP:          %.1f%%  (loss=%.3f, params=%d)" % (mlp_acc*100,mlp_loss,mlp_params))
print("  Fuzzy:        %.1f%%  (loss=%.3f, params=%d)" % (fz_acc*100,fz_loss,fz_params))
print()
if fz_acc >= mlp_acc - 0.05 and fz_acc > base_acc + 0.10:
    print("  ✓✓✓ فازی روی زبان جواب داد")
elif fz_acc > base_acc + 0.10:
    print("  ~ فازی یاد گرفت ولی از MLP عقبه")
elif mlp_acc > base_acc + 0.20:
    print("  ✗ MLP یاد گرفت، فازی نه → محدودیت فازی")
else:
    print("  ✗✗ هیچ‌کدام یاد نگرفتن → تسک/داده مشکل داره")
