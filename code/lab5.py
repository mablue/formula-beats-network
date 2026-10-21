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
def gs(M):
    m,k=len(M),len(M[0]); Q=[[M[i][j] for j in range(k)] for i in range(m)]
    for j in range(k):
        for p in range(j):
            d=sum(Q[i][j]*Q[i][p] for i in range(m))
            for i in range(m): Q[i][j]-=d*Q[i][p]
        n=math.sqrt(sum(Q[i][j]**2 for i in range(m)))
        if n>1e-9:
            for i in range(m): Q[i][j]/=n
    return Q
def lowrank(W,k,iters=20):
    m,n=len(W),len(W[0])
    if k>=min(m,n): return W
    Q=[[random.random()-0.5 for _ in range(k)] for _ in range(n)]
    Wt=T(W)
    for _ in range(iters): Q=gs(mm(Wt,mm(W,Q)))
    return mm(W,mm(Q,T(Q)))
def res_fwd(h,W1,W2):
    z=mm(h,W1); a=relu(z); f=mm(a,W2)
    return add(h,f),z,a
def res_bwd(h_in,dh,W1,W2,z,a):
    dW2=mm(T(a),dh); da=mm(dh,T(W2)); dz=mul(da,rgrad(z))
    dW1=mm(T(h_in),dz); dh_in=add(dh,mm(dz,T(W1)))
    return dh_in,dW1,dW2
def make_data(rng,n):
    X=[]; Y=[]
    for _ in range(n):
        x=[rng.uniform(-1,1) for _ in range(4)]
        X.append(x); Y.append([math.sin(x[0]*x[1]),x[2]*x[2]+x[3]])
    return X,Y
random.seed(1); rng=random.Random(42)
D_in=4; D_h=16; D_out=2; L=4
W1s=[randn(rng,D_h,D_h,0.2) for _ in range(L)]
W2s=[randn(rng,D_h,D_h,0.2) for _ in range(L)]
Emb=randn(rng,D_in,D_h,0.3); Head=randn(rng,D_h,D_out,0.3)
def forward(X):
    h=mm(X,Emb); snaps=[h]; caches=[]
    for i in range(L):
        hn,z,a=res_fwd(h,W1s[i],W2s[i]); caches.append((h,z,a)); h=hn; snaps.append(h)
    return mm(h,Head),caches,snaps
X_tr,Y_tr=make_data(rng,80); X_te,Y_te=make_data(rng,40)
lr=0.02
print("در حال آموزش...")
for ep in range(400):
    y,c,s=forward(X_tr)
    dy=mse_grad(y,Y_tr); dH=mm(T(s[-1]),dy); dh=mm(dy,T(Head))
    dE=None; dW1s=[None]*L; dW2s=[None]*L
    for i in range(L-1,-1,-1):
        dh,dW1,dW2=res_bwd(s[i],dh,W1s[i],W2s[i],c[i][1],c[i][2])
        dW1s[i]=dW1; dW2s[i]=dW2
    dE=mm(T(X_tr),dh)
    for i in range(L):
        for r in range(D_h):
            for cc in range(D_h):
                W1s[i][r][cc]-=lr*dW1s[i][r][cc]
                W2s[i][r][cc]-=lr*dW2s[i][r][cc]
    for r in range(D_in):
        for cc in range(D_h): Emb[r][cc]-=lr*dE[r][cc]
    for r in range(D_h):
        for cc in range(D_out): Head[r][cc]-=lr*dH[r][cc]
    if ep%100==0: print("  epoch %d: loss=%.4f"%(ep,mse(y,Y_tr)))
y_te,_,_=forward(X_te); base=mse(y_te,Y_te)
print("\nBaseline test loss: %.4f"%base)
R={'baseline':base,'results':{}}
print("\n"+"="*60); print(" تست low-rank"); print("="*60)
print("%-6s %-10s %-8s %-8s"%("rank","test_loss","params/lay","vs base"))
W1_orig=[r[:] for r in W1s]; W2_orig=[r[:] for r in W2s]
for k in [1,2,4,8]:
    total=0.0
    for i in range(L):
        W1s[i]=lowrank(W1_orig[i],k); W2s[i]=lowrank(W2_orig[i],k)
    y,_,_=forward(X_te); loss=mse(y,Y_te)
    par=2*D_h*k
    R['results'][k]={'loss':loss,'params_per_layer':par,'vs_baseline':loss/base}
    print("%-6d %-10.4f %-8d %-8.2fx"%(k,loss,par,loss/base))
for i in range(L):
    W1s[i]=W1_orig[i][:]; W2s[i]=W2_orig[i][:]
print("\n--- بهترین rank + کوانت ۴ بیت ---")
best_k=min(R['results'],key=lambda k:R['results'][k]['loss'])
for i in range(L):
    W1s[i]=lowrank(W1_orig[i],best_k); W2s[i]=lowrank(W2_orig[i],best_k)
def qrow(row,bits=4):
    mx=max(abs(z) for z in row) or 1e-9; h=2**(bits-1)
    return [round(z/mx*h)/h*mx for z in row]
W1s=[ [qrow(r) for r in W] for W in W1s]
W2s=[ [qrow(r) for r in W] for W in W2s]
y,_,_=forward(X_te); combined=mse(y,Y_te)
R['combined']={'rank':best_k,'bits':4,'loss':combined,
               'params_total':L*2*D_h*best_k*4, 'vs_baseline':combined/base}
print("rank=%d + 4bit: loss=%.4f  (%.2fx از baseline)"%(best_k,combined,combined/base))
print("\n"+"="*60); print(" نتیجه"); print("="*60)
ok=combined<base*1.5
print("  %s"%("✓ ترکیب rank-%d + 4bit عملاً بی‌ضرر"%best_k if ok else "✗ ترکیب خراب می‌کند"))
print("  فشرده‌سازی نسبت به fp32+full-rank: %.0fx"%(32*D_h/(best_k*4*2/2)))
open('/data/data/com.termux/files/home/lab5_result.json','w').write(json.dumps(R,indent=2))
print("\n✅ lab5_result.json ذخیره شد")
