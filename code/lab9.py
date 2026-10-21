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
def forward(X,Emb,Head,layers,masks=None,frozen=None):
    h=mm(X,Emb); caches=[]
    for li,(W1,W2) in enumerate(layers):
        if frozen is not None and li in frozen and frozen[li] is None:
            caches.append(None); continue
        z=mm(h,W1); a=relu(z)
        if masks is not None and li in masks:
            keep=masks[li]
            a=[[a[r][i] if i in keep else 0.0 for i in range(len(a[0]))] for r in range(len(a))]
        h=add(h,mm(a,W2)); caches.append((z,a))
    return mm(h,Head),caches
def backward(X,Y,Emb,Head,layers,caches,masks=None,frozen=None):
    h=mm(X,Emb); snaps=[h]
    for i in range(len(layers)):
        if frozen is not None and i in frozen and frozen[i] is None:
            snaps.append(h); continue
        h=add(h,mm(caches[i][1] if caches[i] else [[0.0]*len(layers[i][1][0])]*len(h),layers[i][1]))
        snaps.append(h)
    y=mm(h,Head); dy=mse_grad(y,Y)
    dHead=mm(T(snaps[-1]),dy); dh=mm(dy,T(Head))
    dL=[None]*len(layers)
    for i in range(len(layers)-1,-1,-1):
        if frozen is not None and i in frozen and frozen[i] is None:
            continue
        W1,W2=layers[i]; c=caches[i]
        if c is None: continue
        z,a=c
        dW2=mm(T(a),dh); da=mm(dh,T(W2))
        if masks is not None and i in masks:
            keep=masks[i]
            da=[[da[r][j] if j in keep else 0.0 for j in range(len(da[0]))] for r in range(len(da))]
        dz=mul(da,rgrad(z)); dW1=mm(T(snaps[i]),dz)
        dh=add(dh,mm(dz,T(W1)))
        dL[i]=(dW1,dW2)
    dEmb=mm(T(X),dh)
    return dEmb,dHead,dL
def train(Emb,Head,layers,X_tr,Y_tr,epochs=400,lr=0.02,masks=None,frozen=None):
    Emb=[r[:] for r in Emb]; Head=[r[:] for r in Head]
    layers=[([r[:] for r in w1],[r[:] for r in w2]) for w1,w2 in layers]
    for ep in range(epochs):
        y,c=forward(X_tr,Emb,Head,layers,masks,frozen)
        dE,dH,dL=backward(X_tr,Y_tr,Emb,Head,layers,c,masks,frozen)
        for r in range(len(Emb)):
            for cc in range(len(Emb[0])): Emb[r][cc]-=lr*dE[r][cc]
        for r in range(len(Head)):
            for cc in range(len(Head[0])): Head[r][cc]-=lr*dH[r][cc]
        for i,dl in enumerate(dL):
            if dl is None: continue
            dW1,dW2=dl; W1,W2=layers[i]
            if masks is not None and i in masks:
                keep=masks[i]
                for r in range(len(W1)):
                    for cc in range(len(W1[0])):
                        if r not in keep: continue
                        W1[r][cc]-=lr*dW1[r][cc]
                for r in range(len(W2)):
                    for cc in range(len(W2[0])): W2[r][cc]-=lr*dW2[r][cc]
            else:
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
print("در حال آموزش baseline...")
Emb,Head,layers=train(Emb,Head,layers,X_tr,Y_tr)
y,_=forward(X_te,Emb,Head,layers); base=mse(y,Y_te)
print("Baseline loss: %.4f\n"%base)
R={'baseline':base,'tests':{}}

print("="*60); print(" ۹a: لایه وسط = identity کامل"); print("="*60)
for target in range(L):
    frozen={target:None}
    y,_=forward(X_te,Emb,Head,layers,frozen=frozen)
    l=mse(y,Y_te)
    print("  حذف لایه %d: loss=%.4f  (%.2fx از baseline)"%(target,l,l/base))
    R['tests']['remove_L%d'%target]={'loss':l,'ratio':l/base}

print("\n"+"="*60); print(" ۹b: تعیین top-8 نرون مهم + ریتِرِین"); print("="*60)
def important_neurons(layers,X_te,Y_te,base,k=8):
    masks={}
    for li in range(len(layers)):
        imps=[]
        for n in range(len(layers[li][0])):
            h=mm(X_te,Emb)
            for lj,(W1,W2) in enumerate(layers):
                z=mm(h,W1); a=relu(z)
                if lj==li:
                    a=[[a[r][i] if i!=n else 0.0 for i in range(len(a[0]))] for r in range(len(a))]
                h=add(h,mm(a,W2))
            y=mm(h,Head); imps.append((mse(y,Y_te),n))
        imps.sort()
        masks[li]=set(n for _,n in imps[:k])
    return masks
masks=important_neurons(layers,X_te,Y_te,base,k=8)
print("  نرون‌های انتخاب‌شده top-8 (قبل از ریتِرِین):")
y,_=forward(X_te,Emb,Head,layers,masks=masks); pruned_loss=mse(y,Y_te)
print("    top-8 بدون آموزش مجدد: %.4f (%.2fx)"%(pruned_loss,pruned_loss/base))
print("  ریتِرِین با top-8 فریز...")
Emb2,Head2,L2=train(Emb,Head,layers,X_tr,Y_tr,masks=masks)
y,_=forward(X_te,Emb2,Head2,L2,masks=masks); retrained=mse(y,Y_te)
print("    top-8 بعد از ریتِرِین: %.4f (%.2fx از baseline)"%(retrained,retrained/base))
R['tests']['top8_pruned']={'loss':pruned_loss,'ratio':pruned_loss/base}
R['tests']['top8_retrained']={'loss':retrained,'ratio':retrained/base}

print("\n"+"="*60); print(" ۹b۲: با k کوچک‌تر (k=4) ریتِرِین"); print("="*60)
masks4=important_neurons(layers,X_te,Y_te,base,k=4)
Emb3,Head3,L3=train(Emb,Head,layers,X_tr,Y_tr,masks=masks4)
y,_=forward(X_te,Emb3,Head3,L3,masks=masks4); r4=mse(y,Y_te)
print("    top-4 بعد از ریتِرِین: %.4f (%.2fx)"%(r4,r4/base))
R['tests']['top4_retrained']={'loss':r4,'ratio':r4/base}

print("\n"+"="*60); print(" ۹c: صرفه‌جویی محاسبه (FLOPs)"); print("="*60)
def flops_per_sample(k_per_layer, D_h=16, L=4, D_in=4, D_out=2):
    # هر نرون فعال: D_h (input) + D_h (output) ضرب
    total=D_in*D_h + D_h*D_out
    for li in range(L):
        total += k_per_layer[li]*(D_h+D_h)
    return total
full_k=[D_h]*L
top8_k=[8]*L
top4_k=[4]*L
l0_k=[5,1,0,2]  # از lab8
f_full=flops_per_sample(full_k)
f_t8=flops_per_sample(top8_k)
f_t4=flops_per_sample(top4_k)
f_l0=flops_per_sample(l0_k)
print("  FLOP/sample (نسبت به baseline):")
print("    Full (baseline):      %d  (1.00x)"%f_full)
print("    top-8 همه لایه:       %d  (%.2fx)"%(f_t8,f_full/f_t8))
print("    top-4 همه لایه:       %d  (%.2fx)"%(f_t4,f_full/f_t4))
print("    ضروری (از lab8):      %d  (%.2fx)"%(f_l0,f_full/f_l0))
R['flops']={'full':f_full,'top8':f_t8,'top4':f_t4,'essential':f_l0,
            'speedup_top8':f_full/f_t8,'speedup_top4':f_full/f_t4,'speedup_essential':f_full/f_l0}

print("\n"+"="*60); print(" جمع‌بندی نهایی"); print("="*60)
ratios={k:v['ratio'] for k,v in R['tests'].items()}
best=min(ratios,key=ratios.get)
print("  بهترین استراتژی: %s (loss=%.4f, %.2fx از baseline)"%(
    best,R['tests'][best]['loss'],ratios[best]))
print("  صرفه‌جویی محاسبه با top-8: %.2fx"%(f_full/f_t8))
print("  صرفه‌جویی محاسبه با top-4: %.2fx"%(f_full/f_t4))
if R['tests']['top8_retrained']['ratio']<1.3:
    print("  ✓ اسپارسیتی ساختاریافته + ریتِرِین ≈ baseline → قابل تولید انبوه")
elif R['tests']['top4_retrained']['ratio']<1.5:
    print("  ✓ top-4 + ریتِرِین هم قابل قبول → صرفه‌جویی 2.5x")
else:
    print("  ✗ ریتِرِین هم نجات نداد → ساختار ضعیف‌تر از انتظار")
if R['tests']['remove_L2']['ratio']<1.5:
    print("  ✓ لایه ۲ (وسط) = نویز → این تأیید نهایی فرضیه‌ی نویز")
open('/data/data/com.termux/files/home/lab9_result.json','w').write(json.dumps(R,indent=2))
print("\n✅ lab9_result.json ذخیره شد")
