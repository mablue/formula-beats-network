import random, math, json, time
def randn(rng,m,n,s=0.1): return [[rng.gauss(0,s) for _ in range(n)] for _ in range(m)]
def mm(A,B):
    m,n,p=len(A),len(A[0]),len(B[0]); C=[[0.0]*p for _ in range(m)]
    for i in range(m):
        Ai,Ci=A[i],C[i]
        for k in range(n):
            a=Ai[k]
            if a==0: continue
            Bk=B[k]
            for j in range(p): Ci[j]+=a*Bk[j]
    return C
def add(A,B): return [[a+b for a,b in zip(ra,rb)] for ra,rb in zip(A,B)]
def mul(A,B): return [[a*b for a,b in zip(ra,rb)] for ra,rb in zip(A,B)]
def relu(A): return [[max(0.0,x) for x in r] for r in A]
def rgrad(A): return [[1.0 if x>0 else 0.0 for x in r] for r in A]
def T(A): return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]
def mse(A,B):
    t=0.0; n=0
    for ra,rb in zip(A,B):
        for a,b in zip(ra,rb): t+=(a-b)**2; n+=1
    return t/n
def mse_grad(A,B):
    n=len(A)*len(A[0])
    return [[2*(a-b)/n for a,b in zip(ra,rb)] for ra,rb in zip(A,B)]
def make_data(rng,n,task):
    X=[]; Y=[]
    for _ in range(n):
        x=[rng.uniform(-1,1) for _ in range(4)]
        X.append(x)
        if task=='sin':
            Y.append([math.sin(x[0]*x[1]), x[2]*x[2]+x[3]])
        else: # composite: harder
            Y.append([math.sin(math.cos(x[0]*x[1])+x[2]), math.tanh(x[3]*x[0]+x[2]*x[2])])
    return X,Y
def forward(X,Emb,Head,layers,shared):
    h=mm(X,Emb); caches=[]
    L=len(layers) if not shared else shared
    for i in range(L):
        W1,W2=layers[0] if shared else layers[i]
        z=mm(h,W1); a=relu(z); h=add(h,mm(a,W2))
        caches.append((z,a))
    return mm(h,Head),caches
def backward(X,Y,Emb,Head,layers,shared,caches):
    h=mm(X,Emb); snaps=[h]
    L=len(layers) if not shared else shared
    for i in range(L):
        W1,W2=layers[0] if shared else layers[i]
        h=add(h,mm(caches[i][1],W2)); snaps.append(h)
    y=mm(h,Head); dy=mse_grad(y,Y)
    dHead=mm(T(snaps[-1]),dy); dh=mm(dy,T(Head))
    dLayers=[None]*L
    for i in range(L-1,-1,-1):
        W1,W2=layers[0] if shared else layers[i]
        z,a=caches[i]
        dW2=mm(T(a),dh); da=mm(dh,T(W2)); dz=mul(da,rgrad(z))
        dW1=mm(T(snaps[i]),dz); dh=add(dh,mm(dz,T(W1)))
        dLayers[i]=(dW1,dW2)
    dEmb=mm(T(X),dh)
    if shared:
        dW1s=[[0.0]*len(dLayers[0][0][0]) for _ in range(len(dLayers[0][0]))]
        dW2s=[[0.0]*len(dLayers[0][1][0]) for _ in range(len(dLayers[0][1]))]
        for dW1,dW2 in dLayers:
            for r in range(len(dW1)):
                for c in range(len(dW1[0])): dW1s[r][c]+=dW1[r][c]
            for r in range(len(dW2)):
                for c in range(len(dW2[0])): dW2s[r][c]+=dW2[r][c]
        dLayers=[(dW1s,dW2s)]
    return dEmb,dHead,dLayers
def train(Emb,Head,layers,shared,X_tr,Y_tr,X_te,Y_te,epochs=300,lr=0.02):
    Emb=[r[:] for r in Emb]; Head=[r[:] for r in Head]
    layers=[([r[:] for r in w1],[r[:] for r in w2]) for w1,w2 in layers]
    losses=[]
    for ep in range(epochs):
        y,c=forward(X_tr,Emb,Head,layers,shared)
        dE,dH,dL=backward(X_tr,Y_tr,Emb,Head,layers,shared,c)
        for r in range(len(Emb)):
            for cc in range(len(Emb[0])): Emb[r][cc]-=lr*dE[r][cc]
        for r in range(len(Head)):
            for cc in range(len(Head[0])): Head[r][cc]-=lr*dH[r][cc]
        for i,(dW1,dW2) in enumerate(dL):
            W1,W2=layers[i]
            for r in range(len(W1)):
                for cc in range(len(W1[0])): W1[r][cc]-=lr*dW1[r][cc]
            for r in range(len(W2)):
                for cc in range(len(W2[0])): W2[r][cc]-=lr*dW2[r][cc]
        if ep in (50,150,299):
            yy,_=forward(X_te,Emb,Head,layers,shared)
            losses.append((ep,mse(yy,Y_te)))
    y,_=forward(X_te,Emb,Head,layers,shared)
    return mse(y,Y_te), losses
D_in=4; D_out=2
R={'experiments':[]}
print("="*60); print(" lab7: اعتبارسنجی وزن مشترک"); print("="*60)

def run(name,D_h,L,shared,task,seed,X_tr,Y_tr,X_te,Y_te):
    rng=random.Random(seed)
    Emb=randn(rng,D_in,D_h,0.3); Head=randn(rng,D_h,D_out,0.3)
    n_layers=1 if shared else L
    layers=[(randn(rng,D_h,D_h,0.2),randn(rng,D_h,D_h,0.2)) for _ in range(n_layers)]
    loss,prog=train(Emb,Head,layers,shared if shared else None,X_tr,Y_tr,X_te,Y_te)
    n_params=D_in*D_h + D_h*D_out + n_layers*2*D_h*D_h
    return {'name':name,'D_h':D_h,'L':L,'shared':shared,'task':task,
            'seed':seed,'loss':loss,'params':n_params,
            'progress':[{'ep':e,'loss':l} for e,l in prog]}

print("\n--- بخش ۱: چند seed (D_h=16, L=4) ---")
for seed in [42,7,13]:
    rng=random.Random(seed)
    Xtr,Ytr=make_data(rng,80,'sin'); Xte,Yte=make_data(rng,40,'sin')
    rT=run('T_%d'%seed,16,4,False,'sin',seed,Xtr,Ytr,Xte,Yte)
    rS=run('S4w_%d'%seed,16,4,True,'sin',seed,Xtr,Ytr,Xte,Yte)
    R['experiments']+= [rT,rS]
    print("  seed %2d  T=%.4f  S4w=%.4f  فایده=%.1f%%  (params T=%d S=%d)"%(
        seed,rT['loss'],rS['loss'],(rT['loss']-rS['loss'])/rT['loss']*100,rT['params'],rS['params']))

print("\n--- بخش ۲: عمق مشترک (D_h=16) ---")
rng=random.Random(7)
Xtr,Ytr=make_data(rng,80,'sin'); Xte,Yte=make_data(rng,40,'sin')
for L in [2,4,8,12]:
    rS=run('shared_L%d'%L,16,L,True,'sin',7,Xtr,Ytr,Xte,Yte)
    R['experiments'].append(rS)
    print("  shared L=%-2d: loss=%.4f  params=%d"%(
        L,rS['loss'],rS['params']))

print("\n--- بخش ۳: عرض بیشتر (D_h=32) ---")
rng=random.Random(7)
Xtr,Ytr=make_data(rng,80,'sin'); Xte,Yte=make_data(rng,40,'sin')
for L,shared in [(4,False),(4,True),(8,True)]:
    r=run('w32_L%d_%s'%(L,shared),32,L,shared,'sin',7,Xtr,Ytr,Xte,Yte)
    R['experiments'].append(r)
    print("  D_h=32 L=%d %s: loss=%.4f params=%d"%(
        L,'shared' if shared else 'distinct',r['loss'],r['params']))

print("\n--- بخش ۴: وظیفه سخت‌تر (composite) ---")
rng=random.Random(7)
Xtr,Ytr=make_data(rng,100,'comp'); Xte,Yte=make_data(rng,50,'comp')
for L,shared in [(4,False),(4,True),(8,True),(12,True)]:
    r=run('comp_L%d_%s'%(L,shared),16,L,shared,'comp',7,Xtr,Ytr,Xte,Yte)
    R['experiments'].append(r)
    print("  comp L=%d %s: loss=%.4f"%(
        L,'shared' if shared else 'distinct',r['loss']))

print("\n"+"="*60); print(" جمع‌بندی"); print("="*60)
# تحلیل seed
seed_pairs=[(e for e in R['experiments'] if e['name'].startswith('T_')),
            (e for e in R['experiments'] if e['name'].startswith('S4w_'))]
siny={e['name']:e['loss'] for e in R['experiments'] if e['name'].startswith(('T_','S4w_'))}
wins=sum(1 for k in siny if k.startswith('T_') and siny[k]>siny.get('S4w_'+k[2:],9))
print("  پیروزی S4w در %d از ۳ seed اول"%wins)
# تحلیل عمق
depth_s={e['L']:e['loss'] for e in R['experiments'] if e['name'].startswith('shared_L')}
if len(depth_s)>=2:
    best_depth=min(depth_s,key=depth_s.get)
    print("  بهترین عمق مشترک: L=%d (loss=%.4f)"%(best_depth,depth_s[best_depth]))
    if depth_s[max(depth_s)] > depth_s[min(depth_s)]*1.1:
        print("  ➜ عمق مشترک اشباع می‌شود (بیشتر از %d بی‌فایده)"%best_depth)
# تحلیل عرض
w32={e['name']:e['loss'] for e in R['experiments'] if 'w32' in e['name']}
print("  در D_h=32: shared_L4=%.4f  distinct_L4=%.4f  shared_L8=%.4f"%(
    w32.get('w32_L4_True',0),w32.get('w32_L4_False',0),w32.get('w32_L8_True',0)))
# تحلیل سخت
comp={e['name']:e['loss'] for e in R['experiments'] if e['task']=='comp'}
print("  روی وظیفه سخت: shared_L4=%.4f  distinct_L4=%.4f  shared_L8=%.4f  shared_L12=%.4f"%(
    comp.get('comp_L4_True',0),comp.get('comp_L4_False',0),comp.get('comp_L8_True',0),comp.get('comp_L12_True',0)))
open('/data/data/com.termux/files/home/lab7_result.json','w').write(json.dumps(R,indent=2))
print("\n✅ lab7_result.json ذخیره شد")
