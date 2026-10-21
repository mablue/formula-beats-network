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
def res_fwd(h,W1,W2):
    z=mm(h,W1); a=relu(z); f=mm(a,W2); h_next=add(h,f)
    return h_next,z,a
def res_bwd(h_in,dh_out,W1,W2,z,a):
    dW2=mm(T(a),dh_out)
    da=mm(dh_out,T(W2))
    dz=mul(da,rgrad(z))
    dW1=mm(T(h_in),dz)
    dh_in=add(dh_out,mm(dz,T(W1)))
    return dh_in,dW1,dW2
def make_data(rng,n):
    X=[]; Y=[]
    for _ in range(n):
        x=[rng.uniform(-1,1) for _ in range(4)]
        y=[math.sin(x[0]*x[1]), x[2]*x[2]+x[3]]
        X.append(x); Y.append(y)
    return X,Y
random.seed(1); rng=random.Random(42)
D_in=4; D_h=16; D_out=2; L=4
W1s=[randn(rng,D_h,D_h,0.2) for _ in range(L)]
W2s=[randn(rng,D_h,D_h,0.2) for _ in range(L)]
Emb=randn(rng,D_in,D_h,0.3); Head=randn(rng,D_h,D_out,0.3)
def forward(X):
    h=mm(X,Emb); snapshots=[h]
    caches=[]
    for i in range(L):
        h,z,a=res_fwd(h,W1s[i],W2s[i])
        caches.append((h,z,a))
        snapshots.append(h)
    y=mm(h,Head)
    return y,caches,snapshots
def backward(X,Y,caches,snapshots):
    y=mm(snapshots[-1],Head)
    dy=mse_grad(y,Y)
    dHead=mm(T(snapshots[-1]),dy)
    dh=mm(dy,T(Head))
    dW1s=[None]*L; dW2s=[None]*L
    for i in range(L-1,-1,-1):
        h_in=snapshots[i]
        h_out,z,a=caches[i]
        dh,dW1,dW2=res_bwd(h_in,dh,W1s[i],W2s[i],z,a)
        dW1s[i]=dW1; dW2s[i]=dW2
    dEmb=mm(T(X),dh)
    return dEmb,dHead,dW1s,dW2s
X_tr,Y_tr=make_data(rng,80); X_te,Y_te=make_data(rng,40)
lr=0.02
print("در حال آموزش...")
for ep in range(400):
    y,c,s=forward(X_tr)
    dE,dH,dW1,dW2=backward(X_tr,Y_tr,c,s)
    for i in range(L):
        for r in range(D_h):
            for cc in range(D_h):
                W1s[i][r][cc]-=lr*dW1[i][r][cc]
                W2s[i][r][cc]-=lr*dW2[i][r][cc]
    for r in range(D_in):
        for cc in range(D_h): Emb[r][cc]-=lr*dE[r][cc]
    for r in range(D_h):
        for cc in range(D_out): Head[r][cc]-=lr*dH[r][cc]
    if ep%100==0: print("  epoch %d: loss=%.4f"%(ep,mse(y,Y_tr)))
y_te,_,_=forward(X_te)
base=mse(y_te,Y_te)
print("\nBaseline test loss: %.4f"%base)
R={'baseline':base,'per_layer':{}}
print("\n"+"="*60); print(" تست جایگزینی/کوانت هر لایه"); print("="*60)
print("%-8s %-10s %-10s %-10s %-10s"%("layer","random","identity","quant4","contribution"))
for li in range(L):
    orig_W1=[r[:] for r in W1s[li]]; orig_W2=[r[:] for r in W2s[li]]
    std=math.sqrt(sum(x*x for r in orig_W1 for x in r)/(D_h*D_h))
    W1s[li]=randn(rng,D_h,D_h,std); W2s[li]=randn(rng,D_h,D_h,std)
    y,_,_=forward(X_te); rnd_loss=mse(y,Y_te)
    W1s[li]=[[1.0 if i==j else 0.0 for j in range(D_h)] for i in range(D_h)]
    W2s[li]=[[0.0]*D_h for _ in range(D_h)]
    y,_,_=forward(X_te); id_loss=mse(y,Y_te)
    mx=max(abs(x) for r in orig_W1 for x in r) or 1e-9; half=8
    W1s[li]=[[round(orig_W1[i][j]/mx*half)/half*mx for j in range(D_h)] for i in range(D_h)]
    mx2=max(abs(x) for r in orig_W2 for x in r) or 1e-9
    W2s[li]=[[round(orig_W2[i][j]/mx2*half)/half*mx2 for j in range(D_h)] for i in range(D_h)]
    y,_,_=forward(X_te); q_loss=mse(y,Y_te)
    W1s[li]=orig_W1; W2s[li]=orig_W2
    contribution=abs(rnd_loss-base)+abs(id_loss-base)+abs(q_loss-base)
    print("%-8d %-10.4f %-10.4f %-10.4f %-10.4f"%(li,rnd_loss,id_loss,q_loss,contribution))
    R['per_layer'][li]={'random':rnd_loss,'identity':id_loss,'quant4':q_loss,'contribution':contribution}
# حذف کامل همه وسط‌ها
for li in range(L):
    W1s[li]=[[1.0 if i==j else 0.0 for j in range(D_h)] for i in range(D_h)]
    W2s[li]=[[0.0]*D_h for _ in range(D_h)]
y,_,_=forward(X_te); all_id=mse(y,Y_te)
print("\nحذف کامل همه لایه‌های وسط: %.4f"%all_id)
R['all_identity']=all_id
open('/data/data/com.termux/files/home/lab4_result.json','w').write(json.dumps(R,indent=2))
print("\n✅ lab4_result.json ذخیره شد")
