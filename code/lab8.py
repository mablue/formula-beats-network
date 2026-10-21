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
def forward(X,Emb,Head,layers,mask=None):
    h=mm(X,Emb); caches=[]
    for li,(W1,W2) in enumerate(layers):
        z=mm(h,W1); a=relu(z)
        if mask is not None and li in mask:
            keep=mask[li]
            a=[[a[r][i] if i in keep else 0.0 for i in range(len(a[0]))] for r in range(len(a))]
        h=add(h,mm(a,W2)); caches.append((z,a))
    return mm(h,Head),caches
def backward(X,Y,Emb,Head,layers,caches):
    h=mm(X,Emb); snaps=[h]
    for i in range(len(layers)):
        h=add(h,mm(caches[i][1],layers[i][1])); snaps.append(h)
    y=mm(h,Head); dy=mse_grad(y,Y)
    dHead=mm(T(snaps[-1]),dy); dh=mm(dy,T(Head))
    dL=[None]*len(layers)
    for i in range(len(layers)-1,-1,-1):
        W1,W2=layers[i]; z,a=caches[i]
        dW2=mm(T(a),dh); da=mm(dh,T(W2)); dz=mul(da,rgrad(z))
        dW1=mm(T(snaps[i]),dz); dh=add(dh,mm(dz,T(W1)))
        dL[i]=(dW1,dW2)
    dEmb=mm(T(X),dh)
    return dEmb,dHead,dL
def train(Emb,Head,layers,X_tr,Y_tr,epochs=400,lr=0.02):
    Emb=[r[:] for r in Emb]; Head=[r[:] for r in Head]
    layers=[([r[:] for r in w1],[r[:] for r in w2]) for w1,w2 in layers]
    for ep in range(epochs):
        y,c=forward(X_tr,Emb,Head,layers)
        dE,dH,dL=backward(X_tr,Y_tr,Emb,Head,layers,c)
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
    return Emb,Head,layers
random.seed(1); rng=random.Random(42)
D_in=4; D_h=16; D_out=2; L=4
X_tr,Y_tr=make_data(rng,80); X_te,Y_te=make_data(rng,40)
Emb=randn(rng,D_in,D_h,0.3); Head=randn(rng,D_h,D_out,0.3)
layers=[(randn(rng,D_h,D_h,0.2),randn(rng,D_h,D_h,0.2)) for _ in range(L)]
print("در حال آموزش...")
Emb,Head,layers=train(Emb,Head,layers,X_tr,Y_tr)
y,_=forward(X_te,Emb,Head,layers); base=mse(y,Y_te)
print("Baseline loss: %.4f\n"%base)
R={'baseline':base}
print("="*60); print(" بخش ۱: اهمیت هر نرون (صفر کردن یکی‌یکی)"); print("="*60)
imps=[]
for li in range(L):
    li_imps=[]
    for n in range(D_h):
        keep=set(range(D_h))-{n}
        y,_=forward(X_te,Emb,Head,layers,{li:keep})
        li_imps.append(mse(y,Y_te)-base)
    imps.append(li_imps)
    s=sorted(li_imps,reverse=True)
    print("  layer %d: max=%.4f  min=%.4f  median=%.4f  >0.01: %d/%d"%(
        li,s[0],s[-1],s[len(s)//2],sum(1 for x in li_imps if x>0.01),D_h))
R['neuron_importance']=imps
print("\n"+"="*60); print(" بخش ۲: top-k فعال (k نرون قوی‌تر)"); print("="*60)
print("%-6s %-10s %-10s %-8s"%("k","top-k","random-k","فایده"))
R['sparsity']=[]
for k in [2,4,6,8,12]:
    y,_=forward(X_te,Emb,Head,layers); y2,_=forward(X_te,Emb,Head,layers)
    def topk_forward(k,seed=0):
        h=mm(X_te,Emb)
        for li,(W1,W2) in enumerate(layers):
            z=mm(h,W1); a=relu(z)
            new=[]
            for row in a:
                idx=sorted(range(len(row)),key=lambda i:-abs(row[i]))[:k]
                keep=set(idx)
                new.append([row[i] if i in keep else 0.0 for i in range(len(row))])
            h=add(h,mm(new,W2))
        return mm(h,Head)
    def randk_forward(k,seed=0):
        r=random.Random(seed)
        h=mm(X_te,Emb)
        for li,(W1,W2) in enumerate(layers):
            z=mm(h,W1); a=relu(z)
            new=[]
            for row in a:
                idx=r.sample(range(len(row)),k); keep=set(idx)
                new.append([row[i] if i in keep else 0.0 for i in range(len(row))])
            h=add(h,mm(new,W2))
        return mm(h,Head)
    l_top=mse(topk_forward(k),Y_te); l_rnd=mse(randk_forward(k,42),Y_te)
    R['sparsity'].append({'k':k,'top':l_top,'random':l_rnd,'benefit':l_rnd/l_top})
    print("  %-6d %-10.4f %-10.4f %-8.2fx"%(k,l_top,l_rnd,l_rnd/l_top))
print("\n"+"="*60); print(" بخش ۳: آستانه (بالای میانگین)"); print("="*60)
h=mm(X_te,Emb)
for li,(W1,W2) in enumerate(layers):
    z=mm(h,W1); a=relu(z)
    kept=0; total=0
    for row in a:
        m=sum(row)/len(row)
        for x in row:
            total+=1
            if x>m: kept+=1
    print("  layer %d: %.1f%% نرون بالای میانگین"%(li,100*kept/total))
    new=[]
    for row in a:
        m=sum(row)/len(row)
        new.append([x if x>m else 0.0 for x in row])
    h=add(h,mm(new,W2))
y=mm(h,Head); thr_loss=mse(y,Y_te)
print("  threshold loss: %.4f (%.2fx از baseline)"%(thr_loss,thr_loss/base))
R['threshold_loss']=thr_loss
print("\n"+"="*60); print(" جمع‌بندی"); print("="*60)
all_imps=[x for li in imps for x in li]
all_imps_sorted=sorted(all_imps,reverse=True)
top20=all_imps_sorted[:len(all_imps_sorted)//5]
bottom80=all_imps_sorted[len(all_imps_sorted)//5:]
print("  میانگین اهمیت ۲۰%% بالایی: %.4f"%sum(top20)/len(top20))
print("  میانگین اهمیت ۸۰%% پایینی: %.4f"%sum(bottom80)/len(bottom80))
ratio=(sum(top20)/len(top20))/(sum(bottom80)/len(bottom80)+1e-9)
print("  نسبت اهمیت: %.1fx"%ratio)
if ratio>5:
    print("  ➜ ساختار قوی: توزیع نامتقارن → هرس/MoE جواب می‌ده")
elif ratio>2:
    print("  ➜ ساختار متوسط: هرس جزئی ممکنه جواب بده")
else:
    print("  ➜ ساختار ضعیف: نرون‌ها هم‌ارزن → مسیر MoE کمرنگه")
# بهترین sparsity
best_sp=max(R['sparsity'],key=lambda r:r['benefit'])
print("  بهترین k برای top-k: k=%d (top=%.4f, %.1fx)"%(
    best_sp['k'],best_sp['top'],best_sp['top']/base))
open('/data/data/com.termux/files/home/lab8_result.json','w').write(json.dumps(R,indent=2))
print("\n✅ lab8_result.json ذخیره شد")
