import random, math, sys, itertools

def poly_feats(x, deg):
    fs = [1.0]; d = len(x)
    for k in range(1, deg+1):
        for c in itertools.combinations_with_replacement(range(d), k):
            f = 1.0
            for i in c: f *= x[i]
            fs.append(f)
    return fs

def four_feats(x, nf):
    fs = [1.0]
    for f in range(1, nf+1):
        for i in range(len(x)):
            fs.append(math.sin(f*math.pi*x[i]))
            fs.append(math.cos(f*math.pi*x[i]))
    return fs

def solve_lstsq(Phi, Y):
    n, p = len(Phi), len(Phi[0])
    A = [[sum(Phi[k][i]*Phi[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    b = [sum(Phi[k][i]*Y[k] for k in range(n)) for i in range(p)]
    for i in range(p):
        piv = max(range(i,p), key=lambda r: abs(A[r][i]))
        A[i],A[piv]=A[piv],A[i]; b[i],b[piv]=b[piv],b[i]
        if abs(A[i][i])<1e-10: continue
        for r in range(i+1,p):
            f=A[r][i]/A[i][i]
            for c in range(i,p): A[r][c]-=f*A[i][c]
            b[r]-=f*b[i]
    th=[0.0]*p
    for i in range(p-1,-1,-1):
        if abs(A[i][i])<1e-10: continue
        s=b[i]
        for j in range(i+1,p): s-=A[i][j]*th[j]
        th[i]=s/A[i][i]
    return th

class MLP:
    def __init__(self, di, dh, seed=42):
        r=random.Random(seed)
        self.W1=[[r.gauss(0,1/math.sqrt(di)) for _ in range(dh)] for _ in range(di)]
        self.b1=[0.0]*dh
        self.w2=[r.gauss(0,1/math.sqrt(dh)) for _ in range(dh)]
        self.b2=0.0
        self.dh=dh
    def fwd(self,x):
        z=[sum(x[i]*self.W1[i][j] for i in range(len(x)))+self.b1[j] for j in range(self.dh)]
        h=[v if v>0 else 0.0 for v in z]
        o=sum(h[j]*self.w2[j] for j in range(self.dh))+self.b2
        return o,z,h
    def step(self,x,y,lr):
        o,z,h=self.fwd(x)
        dO=2*(o-y)
        dw2=[h[j]*dO for j in range(self.dh)]
        self.b2-=lr*dO
        dh=[self.w2[j]*dO for j in range(self.dh)]
        dz=[dh[j]*(1.0 if z[j]>0 else 0.0) for j in range(self.dh)]
        dW1=[[x[i]*dz[j] for j in range(self.dh)] for i in range(len(x))]
        for i in range(len(self.W1)):
            for j in range(self.dh): self.W1[i][j]-=lr*dW1[i][j]
        for j in range(self.dh): self.b1[j]-=lr*dz[j]
        for j in range(self.dh): self.w2[j]-=lr*dw2[j]

def train_mlp(X,Y,dh=24,epochs=150,lr=0.05):
    m=MLP(len(X[0]),dh)
    for ep in range(epochs):
        for x,y in zip(X,Y): m.step(x,y,lr)
        if (ep+1)%50==0:
            sys.stdout.write('\r  آموزش MLP... %3d/%d'%(ep+1,epochs))
            sys.stdout.flush()
    sys.stdout.write('\r'+' '*40+'\r'); sys.stdout.flush()
    return m

def eval_mlp(m,X,Y):
    return sum((m.fwd(x)[0]-y)**2 for x,y in zip(X,Y))/len(X)

def eval_lin(X,Y,th,fn):
    return sum((sum(t*f for t,f in zip(th,fn(x)))-y)**2 for x,y in zip(X,Y))/len(X)

def gen_poly(rng,n):
    X,Y=[],[]
    for _ in range(n):
        x=[rng.uniform(-1,1) for _ in range(3)]
        Y.append(2*x[0]+3*x[1]*x[2]-x[0]**2+0.5*x[0]*x[1]*x[2]); X.append(x)
    return X,Y

def gen_periodic(rng,n):
    X,Y=[],[]
    for _ in range(n):
        x=[rng.uniform(-1,1) for _ in range(3)]
        Y.append(math.sin(math.pi*x[0])+0.5*math.cos(2*math.pi*x[1])+0.3*math.sin(3*math.pi*x[2])); X.append(x)
    return X,Y

def gen_mixed(rng,n):
    X,Y=[],[]
    for _ in range(n):
        x=[rng.uniform(-1,1) for _ in range(3)]
        Y.append(x[0]*x[1]+math.sin(math.pi*x[2])); X.append(x)
    return X,Y

print("="*60); print(" lab16: Formula-first vs Network-first"); print("="*60)

summary = []
for idx,(name,gen) in enumerate([
    ("Poly deg3 (data=x²,x·y,...)", gen_poly),
    ("Fourier (data=sin,cos)", gen_periodic),
    ("Mixed (data=x·y+sin)", gen_mixed),
]):
    print("\n[%d/3] %s" % (idx+1,name))
    rng=random.Random(42)
    Xtr,Ytr=gen(rng,150); Xte,Yte=gen(rng,40)
    res=[]
    m=train_mlp(Xtr,Ytr)
    res.append(('MLP(h=24)', eval_mlp(m,Xte,Yte), 3*24+24+24+1))
    pth=solve_lstsq([poly_feats(x,3) for x in Xtr],Ytr)
    res.append(('Poly(deg3)', eval_lin(Xte,Yte,pth,lambda x: poly_feats(x,3)), len(pth)))
    fth=solve_lstsq([four_feats(x,3) for x in Xtr],Ytr)
    res.append(('Fourier(3)', eval_lin(Xte,Yte,fth,lambda x: four_feats(x,3)), len(fth)))
    for nm,l,p in res:
        print("  %-15s loss=%.5f  params=%d" % (nm,l,p))
    best=min(res,key=lambda t:t[1])
    print("  → برنده: %s" % best[0])
    summary.append((name,best,res))

print("\n"+"="*60); print(" خلاصه"); print("="*60)
for name,best,res in summary:
    mlp_l=[r for r in res if 'MLP' in r[0]][0][1]
    gain = (mlp_l-best[1])/mlp_l*100 if mlp_l>0 else 0
    print("  %-30s → %-12s (%.0f%% بهتر از MLP)" % (name[:30],best[0],gain))
print("\nنتیجه:")
print("  • روی داده‌ی هم‌ساختار: فرمول مناسب می‌بره، با پارامتر کمتر")
print("  • روی داده‌ی مختلط: هیچ فرمول تنها کافی نیست → MLP")
print("  • زبان از جنس «مختلط» است — فرمول شناخته‌شده ندارد")
