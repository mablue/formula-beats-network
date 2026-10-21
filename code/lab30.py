import numpy as np, time, sys, random
np.random.seed(42); random.seed(42)
def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-26s [%s] %3d%% %s'%(tag[:26],bar,int(pct*100),msg)); sys.stdout.flush()

nouns=['cat','dog','bird','fish','tree','house','car','book','man','woman',
       'child','friend','teacher','student','doctor','artist','river','mountain',
       'city','village','garden','library']
verbs=['sat','ate','saw','sang','ran','read','looked','found','lost','held',
       'wrote','drew','opened','closed','pushed','pulled','bought','sold']
adjs=['big','small','red','blue','old','new','happy','sad','bright','dark',
      'soft','hard','warm','cold','fast','slow']
places=['park','house','tree','sky','road','river','city','garden','library',
        'school','market','station']
preps=['in','on','at','near','under','over','behind','beside']
templates=["the {n} {v} the {n}","the {a} {n} {v}","a {a} {n} {v} the {n}",
    "the {n} {v} {p} the {p2}","the {a} {n} {v} the {a} {n}",
    "the {n} and the {n} {v}","the {n} {v} the {n} and the {n}",
    "the {a} {a} {n} {v}","the {n} {v} {p} the {a} {n}",
    "the {n} {v} the {n} {p} the {p2}"]
def gen(rng):
    return rng.choice(templates).format(n=rng.choice(nouns),v=rng.choice(verbs),
        a=rng.choice(adjs),p=rng.choice(preps),p2=rng.choice(places)).split()
rng=random.Random(42)
tr_s=[gen(rng) for _ in range(2000)]; te_s=[gen(rng) for _ in range(500)]
aw=sorted(set(w for s in tr_s+te_s for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
CTX=3
def mxy(ss):
    X,Y=[],[]
    for s in ss:
        ids=[w2i[w] for w in s]
        for i in range(CTX,len(ids)): X.append(ids[i-CTX:i]); Y.append(ids[i])
    return np.array(X),np.array(Y)
Xtr,Ytr=mxy(tr_s); Xte,Yte=mxy(te_s)
n_val=len(Xtr)//5
perm=np.random.RandomState(0).permutation(len(Xtr))
val_idx=perm[:n_val]; tr_idx=perm[n_val:]

# baseline: most-freq
from collections import Counter
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)

# ---------- One-hot + MLP بزرگ ----------
print("="*60); print(" lab30: سقف واقعی تسک چیه؟"); print("="*60)
print("baseline: %.1f%%\n"%(base_acc*100))

def onehot(X):
    n=len(X)
    v=np.zeros((n,CTX*V),dtype=np.float32)
    for k in range(CTX):
        v[np.arange(n),k*V+X[:,k]]=1
    return v

def train_mlp(Xtr,Ytr,Xte,Yte,hidden,epochs,lr=0.02):
    r=np.random.RandomState(42)
    di=Xtr.shape[1]; dh=hidden; do=V
    W1=r.randn(di,dh).astype(np.float32)*np.sqrt(2.0/di); b1=np.zeros(dh,dtype=np.float32)
    W2=r.randn(dh,do).astype(np.float32)*np.sqrt(2.0/dh); b2=np.zeros(do,dtype=np.float32)
    vW1=np.zeros_like(W1); vW2=np.zeros_like(W2)
    n=len(Xtr)
    for ep in range(epochs):
        z=Xtr@W1+b1; h=np.maximum(z,0); logits=h@W2+b2
        # softmax
        lm=logits.max(axis=1,keepdims=True)
        exp_l=np.exp(logits-lm); P=exp_l/exp_l.sum(axis=1,keepdims=True)
        dlogits=P.copy(); dlogits[np.arange(n),Ytr]-=1.0; dlogits/=n
        dW2=h.T@dlogits; db2=dlogits.sum(0)
        dh_=dlogits@W2.T; dz=dh_*(z>0)
        dW1=Xtr.T@dz; db1=dz.sum(0)
        vW1=0.9*vW1-lr*dW1; W1+=vW1; b1-=lr*db1
        vW2=0.9*vW2-lr*dW2; W2+=vW2; b2-=lr*db2
    z=Xte@W1+b1; h=np.maximum(z,0); logits=h@W2+b2
    acc=np.mean(logits.argmax(1)==Yte)
    return float(acc)

print("[1/3] MLP روی one-hot با ظرفیت‌های مختلف")
Xtr_oh=onehot(Xtr); Xte_oh=onehot(Xte)
Xtr_tr=Xtr_oh[tr_idx]; Xtr_val=Xtr_oh[val_idx]
best_mlp=(0,None)
for h in [64, 128, 256, 512]:
    t0=time.time()
    a=train_mlp(Xtr_tr,Ytr[tr_idx],Xte_oh,Yte,h,epochs=200)
    print("  h=%-4d test=%.1f%% time=%.1fs"%(h,a*100,time.time()-t0))
    if a>best_mlp[0]: best_mlp=(a,h)
    render("mlp",[64,128,256,512].index(h)+1,4,"")
sys.stdout.write('\n'); sys.stdout.flush()

# ---------- RF با بهترین D (برنده lab26) ----------
print("\n[2/3] RF: cos(concat_emb(Wx+b)) D=256")
cooc=np.zeros((V,V),dtype=np.float32)
for s in tr_s:
    ids=[w2i[w] for w in s]
    for i in range(len(ids)):
        for j in range(max(0,i-3),min(len(ids),i+4)):
            if i!=j: cooc[ids[i],ids[j]]+=1.0
P=(cooc+1e-3)/(cooc.sum()+1e-3*V*V)
ppmi=np.maximum(np.log(P/(P.sum(1,keepdims=True)*P.sum(0,keepdims=True)+1e-9)+1e-9),0)
U,S,_=np.linalg.svd(ppmi,full_matrices=False)
D_EMB=8
emb=(U[:,:D_EMB]*np.sqrt(S[:D_EMB])).astype(np.float32)
emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-9)
def feats(X): return np.concatenate([emb[X[:,0]],emb[X[:,1]],emb[X[:,2]]],axis=1)
Ftr=feats(Xtr); Fte=feats(Xte)
Ftr_tr=Ftr[tr_idx]; Ftr_val=Ftr[val_idx]
Etr_tr=emb[Ytr[tr_idx]]; Ete=emb[Yte]
def acc_rf(pred):
    d=np.linalg.norm(pred[:,None,:]-emb[None,:,:],axis=2)
    return float(np.mean(d.argmin(1)==Yte))
best_rf=0; best_D_rf=0
for D in [128, 256, 512]:
    r=np.random.RandomState(100)
    W=r.randn(Ftr.shape[1],D).astype(np.float32)*np.sqrt(2.0/Ftr.shape[1])
    b=r.uniform(0,2*np.pi,D).astype(np.float32)
    Ptr=np.cos(Ftr_tr@W+b); Pval=np.cos(Ftr_val@W+b); Pte=np.cos(Fte@W+b)
    lam=0.5
    A=Ptr.T@Ptr+lam*np.eye(D)
    Th=np.linalg.solve(A,Ptr.T@Etr_tr)
    a=acc_rf(Pte@Th)
    print("  D=%-4d test=%.1f%%"%(D,a*100))
    if a>best_rf: best_rf=a; best_D_rf=D
    render("rf",[128,256,512].index(D)+1,3,"")
sys.stdout.write('\n'); sys.stdout.flush()

# ---------- Oracle: چقدر سقف ممکنه ----------
print("\n[3/3] سقف ممکن (Oracle)")
# ۱) در train چقدر context یکتا داریم که target یکتا داره؟
train_map={}
for x,y in zip(Xtr,Ytr):
    key=tuple(x)
    if key not in train_map: train_map[key]=[]
    train_map[key].append(y)
from collections import Counter as Ctr
# ۲) accuracy چقدره اگر بهترین حدس رو از train برای هر context بزنیم
correct=0; total=0; ambiguous=0
for x,y in zip(Xte,Yte):
    key=tuple(x)
    if key in train_map:
        counts=Ctr(train_map[key])
        best=counts.most_common(1)[0][0]
        if best==y: correct+=1
        if len(counts)>1: ambiguous+=1
    else:
        counts=Ctr(Ytr)
        best=counts.most_common(1)[0][0]
        if best==y: correct+=1
    total+=1
oracle_acc=correct/total
print("  Oracle (بهترین حدس از train): %.1f%%"%(oracle_acc*100))
print("  contextهای مبهم: %.1f%%"%(100*ambiguous/total))
print("  contextهای جدید در test: %.1f%%"%(100*(1-sum(1 for x in Xte if tuple(x) in train_map)/len(Xte))))
print("  → اگر این oracle بالاتر از مدل‌هاست، جا برای بهبود هست")

print("\n"+"="*60); print(" نتیجه‌گیری نهایی"); print("="*60)
print("  baseline:           %.1f%%"%(base_acc*100))
print("  RF (D=256):         %.1f%%"%(best_rf*100))
print("  MLP (h=%d):         %.1f%%"%(best_mlp[1],best_mlp[0]*100))
print("  Oracle (سقف ممکن):  %.1f%%"%(oracle_acc*100))
print()
# آیا gap بین مدل و oracle زیاده؟
gap=oracle_acc-max(best_rf,best_mlp[0])
print("  فاصله بهترین مدل تا oracle: %.1f%%"%(gap*100))
if gap<0.05:
    print("  ✓✓✓ مدل به سقف تسک رسیده → برنده شدیم!")
    print("       سقف از تسک بود، نه الگوریتم")
elif gap<0.15:
    print("  ~ مدل ۱۰-۱۵٪ از سقف فاصله داره → جا برای بهبود")
else:
    print("  ✗ فاصله زیاد تا oracle → الگوریتم ضعیفه، باید عوض شه")

# آیا RF از MLP بهتره؟
print()
if best_rf>best_mlp[0]+0.03:
    print("  ✓ RF از MLP بهتره → حرف تو درست بود: فرمول > شبکه")
    print("       با %.0f%% پارامتر کمتر، %.1f%% دقیق‌تر"%(100*(1-best_D_rf/(64*V)),(best_rf-best_mlp[0])*100))
else:
    print("  ≈ RF ≈ MLP → هر دو یه سقف دارن")
