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
def forward(X,Emb,Head,layers,shared):
    h=mm(X,Emb); caches=[]
    L=len(layers) if not shared else shared
    for i in range(L):
        W1,W2=layers[0] if shared else layers[i]
        z=mm(h,W1); a=relu(z); f=mm(a,W2); h=add(h,f)
        caches.append((z,a))
    return mm(h,Head),caches
def backward(X,Y,Emb,Head,layers,shared,caches):
    h=mm(X,Emb)
    # snapshot هر مرحله
    snaps=[h]
    L=len(layers) if not shared else shared
    for i in range(L):
        W1,W2=layers[0] if shared else layers[i]
        z,a=caches[i]
        h=add(h,mm(a,W2)); snaps.append(h)
    y=mm(h,Head); dy=mse_grad(y,Y)
    dHead=mm(T(snaps[-1]),dy); dh=mm(dy,T(Head))
    dLayers=[None]*L
    for i in range(L-1,-1,-1):
        W1,W2=layers[0] if shared else layers[i]
        z,a=caches[i]
        dW2=mm(T(a),dh); da=mm(dh,T(W2)); dz=mul(da,rgrad(z))
        dW1=mm(T(snaps[i]),dz); dh_new=add(dh,mm(dz,T(W1))); dh=dh_new
        dLayers[i]=(dW1,dW2)
    dEmb=mm(T(X),dh)
    if shared:
        dW1sum=[[0.0]*len(dLayers[0][0][0]) for _ in range(len(dLayers[0][0]))]
        dW2sum=[[0.0]*len(dLayers[0][1][0]) for _ in range(len(dLayers[0][1]))]
        for dW1,dW2 in dLayers:
            for r in range(len(dW1)):
                for c in range(len(dW1[0])): dW1sum[r][c]+=dW1[r][c]
            for r in range(len(dW2)):
                for c in range(len(dW2[0])): dW2sum[r][c]+=dW2[r][c]
        dLayers=[(dW1sum,dW2sum)]
    return dEmb,dHead,dLayers
def train(name,Emb_init,Head_init,layers_init,shared,X_tr,Y_tr,X_te,Y_te,epochs=400,lr=0.02):
    Emb=[r[:] for r in Emb_init]; Head=[r[:] for r in Head_init]
    layers=[( [r[:] for r in w1], [r[:] for r in w2] ) for w1,w2 in layers_init]
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
    y,_=forward(X_te,Emb,Head,layers,shared)
    return mse(y,Y_te),layers,Emb,Head
random.seed(1); rng=random.Random(42)
D_in=4; D_h=16; D_out=2
X_tr,Y_tr=make_data(rng,80); X_te,Y_te=make_data(rng,40)
def init_all(L):
    return (randn(rng,D_in,D_h,0.3), randn(rng,D_h,D_out,0.3),
            [(randn(rng,D_h,D_h,0.2),randn(rng,D_h,D_h,0.2)) for _ in range(L)])
print("="*60); print(" lab6: آیا عمق لازمه؟"); print("="*60)
R={}
# T: 4 لایه
E,H,Ls=init_all(4)
loss_T,_,_,_=train('T',E,H,Ls,None,X_tr,Y_tr,X_te,Y_te)
R['T_4layer']={'loss':loss_T,'layers':4,'shared':False,'params':4*2*D_h*D_h}
print("T (4 لایه)         : loss=%.4f"%loss_T)
# S1: 1 لایه
E,H,Ls=init_all(1)
loss_S1,_,_,_=train('S1',E,H,Ls,None,X_tr,Y_tr,X_te,Y_te)
R['S1_1layer']={'loss':loss_S1,'layers':1,'shared':False,'params':2*D_h*D_h}
print("S1 (1 لایه)        : loss=%.4f"%loss_S1)
# S2: 2 لایه
E,H,Ls=init_all(2)
loss_S2,_,_,_=train('S2',E,H,Ls,None,X_tr,Y_tr,X_te,Y_te)
R['S2_2layer']={'loss':loss_S2,'layers':2,'shared':False,'params':2*2*D_h*D_h}
print("S2 (2 لایه)        : loss=%.4f"%loss_S2)
# S4w: 4 لایه با وزن مشترک
E,H,Ls=init_all(1)
loss_S4w,_,_,_=train('S4w',E,H,Ls,4,X_tr,Y_tr,X_te,Y_te)
R['S4w_shared']={'loss':loss_S4w,'layers':4,'shared':True,'params':2*D_h*D_h}
print("S4w (4 لایه مشترک) : loss=%.4f"%loss_S4w)
print("\n"+"="*60); print(" تحلیل"); print("="*60)
base=loss_T
for k in ['S1_1layer','S2_2layer','S4w_shared']:
    v=R[k]['loss']
    ratio=v/base
    verdict='✓ کافی' if ratio<1.3 else ('~ نزدیک' if ratio<2.0 else '✗ ناکافی')
    print("  %-12s: %.4f  (%.2fx از T)  %s"%(k,v,ratio,verdict))
print("\nجمع‌بندی:")
if R['S1_1layer']['loss']/base<1.3:
    print("  ➜ عمق لازم نیست! 1 لایه کافیه = ۴ برابر سریع‌تر بدون افت")
elif R['S2_2layer']['loss']/base<1.3:
    print("  ➜ 2 لایه کافیه = ۲ برابر سریع‌تر بدون افت")
elif R['S4w_shared']['loss']/base<1.3:
    print("  ➜ وزن مشترک کافیه = ۴ برابر کم‌پارامتر بدون افت")
else:
    print("  ➜ عمق واقعاً لازمه، هیچ shortcut جواب نداد")
open('/data/data/com.termux/files/home/lab6_result.json','w').write(json.dumps(R,indent=2))
print("\n✅ lab6_result.json ذخیره شد")
