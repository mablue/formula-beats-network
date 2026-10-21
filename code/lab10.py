import random, math, json
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
def make_data(rng,n):
    X=[]; Y=[]
    for _ in range(n):
        x=[rng.uniform(-1,1) for _ in range(4)]
        X.append(x); Y.append([math.sin(x[0]*x[1]),x[2]*x[2]+x[3]])
    return X,Y
def forward(X,Emb,Head,layers):
    h=mm(X,Emb); snaps=[h]; acts=[]
    for W1,W2 in layers:
        z=mm(h,W1); a=relu(z); h=add(h,mm(a,W2))
        snaps.append(h); acts.append((z,a))
    return mm(h,Head),snaps,acts
def backward(X,Y,Emb,Head,layers,snaps,acts,frozen):
    y=mm(snaps[-1],Head)
    dy=mse_grad(y,Y)
    dHead=mm(T(snaps[-1]),dy)
    dh=mm(dy,T(Head))
    L=len(layers); dL=[None]*L
    for i in range(L-1,-1,-1):
        if i in frozen: continue
        W1,W2=layers[i]; z,a=acts[i]
        dW2=mm(T(a),dh)
        da=mm(dh,T(W2))
        dz=mul(da,rgrad(z))
        dW1=mm(T(snaps[i]),dz)
        dh=add(dh,mm(dz,T(W1)))
        dL[i]=(dW1,dW2)
    dEmb=mm(T(X),dh)
    return dEmb,dHead,dL
def train(Emb,Head,layers,frozen,X_tr,Y_tr,epochs=500,lr=0.02):
    Emb=[r[:] for r in Emb]; Head=[r[:] for r in Head]
    layers=[([r[:] for r in w1],[r[:] for r in w2]) for w1,w2 in layers]
    for ep in range(epochs):
        y,s,a=forward(X_tr,Emb,Head,layers)
        dE,dH,dL=backward(X_tr,Y_tr,Emb,Head,layers,s,a,frozen)
        for r in range(len(Emb)):
            for cc in range(len(Emb[0])): Emb[r][cc]-=lr*dE[r][cc]
        for r in range(len(Head)):
            for cc in range(len(Head[0])): Head[r][cc]-=lr*dH[r][cc]
        for i,dl in enumerate(dL):
            if dl is None: continue
            dW1,dW2=dl; W1,W2=layers[i]
            for r in range(len(W1)):
                for cc in range(len(W1[0])): W1[r][cc]-=lr*dW1[r][cc]
            for r in range(len(W2)):
                for cc in range(len(W2[0])): W2[r][cc]-=lr*dW2[r][cc]
    return Emb,Head,layers
def eval_net(Emb,Head,layers,X_te,Y_te):
    y,_,_=forward(X_te,Emb,Head,layers); return mse(y,Y_te)
random.seed(1); rng=random.Random(42)
D_in=4; D_h=16; D_out=2; L=4
X_tr,Y_tr=make_data(rng,80); X_te,Y_te=make_data(rng,40)
R={}
print("="*60); print(" lab10: Reservoir — لایه وسط = تصادفی ثابت"); print("="*60)
# Baseline
Emb=randn(rng,D_in,D_h,0.3); Head=randn(rng,D_h,D_out,0.3)
layers=[(randn(rng,D_h,D_h,0.2),randn(rng,D_h,D_h,0.2)) for _ in range(L)]
Emb,Head,layers=train(Emb,Head,layers,(),X_tr,Y_tr)
base=eval_net(Emb,Head,layers,X_te,Y_te)
R['baseline']={'loss':base,'ratio':1.0}
print("\nBaseline: %.4f"%base)

def make_layer(rng,scale): return (randn(rng,D_h,D_h,scale),randn(rng,D_h,D_h,scale))
def rand_emb_head(rng): return randn(rng,D_in,D_h,0.3), randn(rng,D_h,D_out,0.3)

print("\n--- A: L1, L2 ثابت تصادفی، L0 و L3 آموزش ---")
for scale in [0.1, 0.2, 0.5]:
    Emb,Head=rand_emb_head(rng)
    layers=[make_layer(rng,0.2), make_layer(rng,scale), make_layer(rng,scale), make_layer(rng,0.2)]
    Emb,Head,layers=train(Emb,Head,layers,(1,2),X_tr,Y_tr)
    loss=eval_net(Emb,Head,layers,X_te,Y_te)
    print("  scale=%.1f: loss=%.4f  (%.2fx)"%(scale,loss,loss/base))
    R['A_scale_%.1f'%scale]={'loss':loss,'ratio':loss/base}

print("\n--- B: L1,L2,L3 ثابت تصادفی، فقط L0 آموزش ---")
for scale in [0.1, 0.2, 0.5]:
    Emb,Head=rand_emb_head(rng)
    layers=[make_layer(rng,0.2), make_layer(rng,scale), make_layer(rng,scale), make_layer(rng,scale)]
    Emb,Head,layers=train(Emb,Head,layers,(1,2,3),X_tr,Y_tr)
    loss=eval_net(Emb,Head,layers,X_te,Y_te)
    print("  scale=%.1f: loss=%.4f  (%.2fx)"%(scale,loss,loss/base))
    R['B_scale_%.1f'%scale]={'loss':loss,'ratio':loss/base}

print("\n--- C: فقط L1 تصادفی، بقیه آموزش ---")
for scale in [0.2, 0.5]:
    Emb,Head=rand_emb_head(rng)
    layers=[make_layer(rng,0.2), make_layer(rng,scale), make_layer(rng,0.2), make_layer(rng,0.2)]
    Emb,Head,layers=train(Emb,Head,layers,(1,),X_tr,Y_tr)
    loss=eval_net(Emb,Head,layers,X_te,Y_te)
    print("  scale=%.1f: loss=%.4f  (%.2fx)"%(scale,loss,loss/base))
    R['C_scale_%.1f'%scale]={'loss':loss,'ratio':loss/base}

print("\n"+"="*60); print(" نتیجه"); print("="*60)
keys=[k for k in R if k!='baseline']
best_key=min(keys,key=lambda k:R[k]['ratio'])
print("  بهترین: %s  (loss=%.4f, ratio=%.2fx)"%(best_key,R[best_key]['loss'],R[best_key]['ratio']))
if R[best_key]['ratio']<1.5:
    print("  ✓ RESERVOIR جواب داد → تأیید نهایی فرضیه نویز")
elif R[best_key]['ratio']<2.5:
    print("  ~ نیمه‌موفق: reservoir با افت کار می‌کنه")
else:
    print("  ✗ reservoir هم کار نکرد → مسیر نویز بسته شد")
open('/data/data/com.termux/files/home/lab10_result.json','w').write(json.dumps(R,indent=2))
print("\n✅ lab10_result.json ذخیره شد")
