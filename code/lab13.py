import random, math, json, sys
def make_data(rng,n,D):
    X=[]; Y=[]
    for _ in range(n):
        x=[rng.uniform(-1,1) for _ in range(D)]
        y1=math.sin(x[0]*x[1])
        y2=x[2]*x[2]+x[3]
        for k in range(4,D):
            if k%2==0: y1+=0.3*math.cos(x[k]+x[k-1])
            else: y2+=0.2*x[k]*x[k-1]
        X.append(x); Y.append([y1,y2])
    return X,Y
def mse_pt(y,t): return sum((y[j]-t[j])**2 for j in range(len(y)))
def init_rule(rng,d,out):
    return {'c':[rng.uniform(-0.8,0.8) for _ in range(d)],
            's':rng.uniform(0.5,1.5),
            'a':[[rng.gauss(0,0.3) for _ in range(d)] for _ in range(out)],
            'b':[rng.gauss(0,0.1) for _ in range(out)]}
def mu(x,r):
    d2=sum((xi-ci)**2 for xi,ci in zip(x,r['c']))
    return math.exp(-d2/(2*r['s']*r['s']+1e-9))
def forward(x,rules):
    fires=[(mu(x,r),
            [r['b'][j]+sum(r['a'][j][k]*x[k] for k in range(len(x))) for j in range(len(r['b']))])
           for r in rules]
    Z=sum(m for m,_ in fires)+1e-9
    out_dim=len(rules[0]['b'])
    y=[sum(m*f[j] for m,f in fires)/Z for j in range(out_dim)]
    return y,fires,Z
def update(x,Y,rules,lr,train_cs):
    y,fires,Z=forward(x,rules)
    out_dim=len(rules[0]['b'])
    dy=[2*(y[j]-Y[j]) for j in range(out_dim)]
    for i,r in enumerate(rules):
        m_i,f_i=fires[i]
        df=[dy[j]*m_i/Z for j in range(out_dim)]
        dm=sum(dy[j]*(f_i[j]-y[j]) for j in range(out_dim))/Z
        for j in range(out_dim):
            r['b'][j]-=lr*df[j]
            for k in range(len(x)): r['a'][j][k]-=lr*df[j]*x[k]
        if train_cs:
            d2=sum((x[k]-r['c'][k])**2 for k in range(len(x)))
            for k in range(len(x)):
                dmu=mu(x,r)*(x[k]-r['c'][k])/(r['s']*r['s']+1e-9)
                r['c'][k]-=lr*dm*dmu
            dmu_s=mu(x,r)*d2/(r['s']**3+1e-9)
            r['s']=max(0.1,r['s']-lr*dm*dmu_s)
    return mse_pt(y,Y)
def render(tag,ep,total,loss,width=15):
    pct=ep/total; n=int(pct*width); bar='#'*n+'-'*(width-n)
    sys.stdout.write('\r  %-30s [%s] %3d%%  loss=%.4f'%(tag[:30],bar,int(pct*100),loss))
    sys.stdout.flush()
def eval_loss(rules,X_te,Y_te):
    return sum(mse_pt(forward(x,rules)[0],y) for x,y in zip(X_te,Y_te))/len(X_te)
def train_tsk(X_tr,Y_tr,X_te,Y_te,D,K,epochs=150,lr=0.05,train_cs=True,seed=42,tag=''):
    rng=random.Random(seed)
    rules=[init_rule(rng,D,2) for _ in range(K)]
    for ep in range(epochs):
        for x,y in zip(X_tr,Y_tr): update(x,y,rules,lr,train_cs)
        if (ep+1)%25==0 or ep==0:
            render(tag,ep+1,epochs,eval_loss(rules,X_te,Y_te))
    sys.stdout.write('\n'); sys.stdout.flush()
    loss=eval_loss(rules,X_te,Y_te)
    n_params=K*(D+1+2*D+2) if train_cs else K*(2*D+2)
    return loss,n_params
random.seed(1); rng=random.Random(42)
print("="*60); print(" lab13: تست مقیاس‌پذیری میدان فازی"); print("="*60)
print("Baseline MLP (4D, 80 samples): loss=0.0226, params=2144")
print("هر تست: 150 epoch\n")
R={}
configs=[
 ('A: K=8,  4D, 400s', 4,  8, 400, True),
 ('B: K=16, 4D, 400s', 4, 16, 400, True),
 ('C: K=16, 8D, 400s', 8, 16, 400, True),
 ('D: K=16, 4D, c/s-frozen', 4, 16, 400, False),
]
results=[]
for i,(name,D,K,ntrain,tcs) in enumerate(configs):
    print("[%d/%d] %s"%(i+1,len(configs),name))
    rng=random.Random(99)
    X_tr,Y_tr=make_data(rng,ntrain,D)
    X_te,Y_te=make_data(rng,100,D)
    loss,nparams=train_tsk(X_tr,Y_tr,X_te,Y_te,D,K,epochs=150,train_cs=tcs,tag=name[:30])
    results.append((name,loss,nparams,loss/0.0226))
    R[name]={'loss':loss,'params':nparams,'ratio':loss/0.0226,'D':D,'K':K,'ntrain':ntrain}
print("\n"+"="*60); print(" جدول نهایی"); print("="*60)
print("%-30s %-8s %-8s %-8s"%("config","loss","params","ratio"))
for name,loss,nparams,ratio in results:
    print("%-30s %-8.4f %-8d %-8.2fx"%(name,loss,nparams,ratio))
print("\n"+"="*60); print(" تحلیل مقیاس‌پذیری"); print("="*60)
losses={name:loss for name,loss,_,_ in results}
samples_effect=losses['B: K=16, 4D, 400s']-losses['A: K=8,  4D, 400s']
print("  ۱) K=16 vs K=8 (با ۴۰۰ نمونه): Δ=%+.4f"%samples_effect)
if samples_effect<0:
    print("     ✓ K=16 الان بهتره → overfit با ۸۰ نمونه بود، مشکلی نیست")
else:
    print("     ✗ K=16 هنوز بدتره → مشکل از K زیاد نیست، از چیز دیگه‌ست")
dim_effect=losses['C: K=16, 8D, 400s']
print("  ۲) ۸-بعدی، K=16: loss=%.4f (%.2fx از baseline)"%(dim_effect,dim_effect/0.0226))
if dim_effect<0.05:
    print("     ✓ ابعاد بالا مشکل نبود → قابل تعمیم به زبان")
elif dim_effect<0.1:
    print("     ~ افت داره ولی کار می‌کنه → باید K رو با بعد زیاد کرد")
else:
    print("     ✗ curse of dimensionality → در ۸ بعد هم افتضاح")
frozen_effect=losses['D: K=16, 4D, c/s-frozen']-losses['B: K=16, 4D, 400s']
print("  ۳) c/s فریز: Δ=%+.4f (0=بی‌تأثیر، + یعنی بدتر)"%frozen_effect)
if abs(frozen_effect)<0.005:
    print("     ✓ c,σ واقعاً لازم نیستن → پارامتر قابل‌آموزش فقط a,b")
else:
    print("     ~ c,σ کمی کمک می‌کنن")
open('/data/data/com.termux/files/home/lab13_result.json','w').write(json.dumps(R,indent=2))
print("\n✅ lab13_result.json ذخیره شد")
