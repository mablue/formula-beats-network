import numpy as np, sys, time, math, random
np.random.seed(42)

def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-24s [%s] %3d%% %s'%(tag[:24],bar,int(pct*100),msg))
    sys.stdout.flush()

# داده
rng=random.Random(42)
nouns=['cat','dog','bird','fish','tree','house','car','book']
verbs=['sat','ate','saw','sang','ran','read']
adjs=['big','small','red','blue','old','new']
places=['park','house','tree','sky']
def gen():
    return rng.choice([
        "the {n} {v} the {n}","the {a} {n} {v}",
        "a {a} {n} {v} the {n}","the {n} {v} in the {p}",
    ]).format(n=rng.choice(nouns),v=rng.choice(verbs),a=rng.choice(adjs),p=rng.choice(places)).split()
tr=[gen() for _ in range(300)]
te=[gen() for _ in range(60)]
aw=sorted(set(w for s in tr+te for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
print("="*60); print(" lab21: SINDy روی embedding"); print("="*60)
print("V=%d | train=%d جمله | test=%d جمله\n"%(V,len(tr),len(te)))

def make_xy(sents,ctx=2):
    X,Y=[],[]
    for s in sents:
        ids=[w2i[w] for w in s]
        for i in range(ctx,len(ids)):
            X.append(ids[i-ctx:i]); Y.append(ids[i])
    return np.array(X),np.array(Y)
Xtr,Ytr=make_xy(tr); Xte,Yte=make_xy(te)
print("train samples: %d | test samples: %d"%(len(Xtr),len(Xte)))

# ============================================================
# بخش ۱: ماتریس هم‌رخدادی برای یادگیری embedding
# ============================================================
print("\n[1/3] یادگیری embedding از هم‌رخدادی")
cooc=np.zeros((V,V),dtype=np.float32)
for s in tr:
    ids=[w2i[w] for w in s]
    for i in range(len(ids)):
        for j in range(max(0,i-3),min(len(ids),i+4)):
            if i!=j: cooc[ids[i],ids[j]]+=1
# PPMI
total=cooc.sum()
P=cooc/total
pi=P.sum(axis=1,keepdims=True)
pj=P.sum(axis=0,keepdims=True)
pmi=np.log((P+1e-9)/(pi*pj+1e-9))
ppmi=np.maximum(pmi,0)
# SVD → embedding
U,S,Vt=np.linalg.svd(ppmi,full_matrices=False)
d_emb=8
emb=U[:,:d_emb]*np.sqrt(S[:d_emb])
emb=emb.astype(np.float32)
print("  embedding: %d بعد، از PPMI"%(d_emb))

# ============================================================
# بخش ۲: SINDy روی embedding ها
# ============================================================
def build_features(x):
    """x = [e_i, e_j] در 2*d بعد، عبارت‌های SINDy"""
    d=len(x)//2
    a=x[:d]; b=x[d:]
    f=[1.0]; n=['1']
    for i in range(d):
        f.append(a[i]); n.append(f'a{i}')
        f.append(b[i]); n.append(f'b{i}')
    for i in range(d):
        f.append(a[i]*b[i]); n.append(f'a{i}·b{i}')
    for i in range(d):
        f.append(a[i]**2); n.append(f'a{i}²')
        f.append(b[i]**2); n.append(f'b{i}²')
    for i in range(d):
        for j in range(d):
            if i!=j: f.append(a[i]*b[j]); n.append(f'a{i}·b{j}')
    for i in range(min(3,d)):
        f.append(math.sin(a[i])); n.append(f'sin(a{i})')
        f.append(math.cos(b[i])); n.append(f'cos(b{i})')
    return np.array(f,dtype=np.float32),n

print("\n[2/3] ساخت دیکشنری ویژگی از embedding")
all_feats=[]; names=None
for k in range(len(Xtr)):
    fi,n=build_features(np.concatenate([emb[Xtr[k,0]],emb[Xtr[k,1]]]))
    all_feats.append(fi)
    if names is None: names=n
Phi_tr=np.array(all_feats,dtype=np.float32)
print("  تعداد ویژگی: %d"%len(names))

# نرمال‌سازی
norms=np.linalg.norm(Phi_tr,axis=0)+1e-9
Phi_tr_n=Phi_tr/norms

# ============================================================
# بخش ۳: پیش‌بینی از طریق رگرسیون به embedding کلمه بعدی
# ============================================================
# استراتژی: به جای پیش‌بینی مستقیم k، embedding کلمه بعدی رو پیش‌بینی می‌کنیم
# بعد نزدیک‌ترین کلمه رو انتخاب می‌کنیم (این با «توزیع‌شدگی» جور درمیاد)
print("\n[3/3] آموزش: پیش‌بینی embedding کلمه بعدی")
E_targets=emb[Ytr]  # (n, d_emb)
# ridge regression: Phi_w = E_targets
lam=0.1
A=Phi_tr_n.T@Phi_tr_n+lam*np.eye(Phi_tr_n.shape[1])
W=np.linalg.solve(A,Phi_tr_n.T@E_targets)
print("  W shape: %s"%(str(W.shape)))

# ارزیابی
def eval_model(X,Y):
    correct=0; total=0; mse_emb=0.0
    for k in range(len(X)):
        fi,_=build_features(np.concatenate([emb[X[k,0]],emb[X[k,1]]]))
        fi=fi/norms
        pred=fi@W  # embedding پیش‌بینی شده
        # نزدیک‌ترین کلمه
        dists=np.linalg.norm(emb-pred[None,:],axis=1)
        pred_word=dists.argmin()
        if pred_word==Y[k]: correct+=1
        mse_emb+=float(np.mean((pred-emb[Y[k]])**2))
        total+=1
    return correct/total, mse_emb/total

train_acc,train_mse=eval_model(Xtr,Ytr)
test_acc,test_mse=eval_model(Xte,Yte)

# baseline: most-frequent
from collections import Counter
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)

# baseline: 1-layer RF روی one-hot
def onehot(c):
    v=np.zeros(V*2,dtype=np.float32)
    v[c[0]]=1; v[V+c[1]]=1
    return v
Phi_oh=np.array([onehot(Xtr[k]) for k in range(len(Xtr))])
Phi_oh_te=np.array([onehot(Xte[k]) for k in range(len(Xte))])
lam=0.01
A=Phi_oh.T@Phi_oh+lam*np.eye(Phi_oh.shape[1])
W_oh=np.linalg.solve(A,Phi_oh.T@E_targets)  # همون هدف: embedding بعدی
def eval_oh(X,Y,W):
    correct=0; total=0
    for k in range(len(X)):
        v=onehot(X[k])
        pred=v@W
        dw=np.linalg.norm(emb-pred[None,:],axis=1).argmin()
        if dw==Y[k]: correct+=1
        total+=1
    return correct/total
oh_test=eval_oh(Xte,Yte,W_oh)

print("\n"+"="*60); print(" نتیجه"); print("="*60)
print("  Most-freq:           %.1f%%"%(base_acc*100))
print("  One-hot ridge:       %.1f%%"%(oh_test*100))
print("  SINDy+embedding:     %.1f%%"%(test_acc*100))
print("  (train acc: %.1f%%  mse_emb=%.6f)"%(train_acc*100,test_mse))

# تعداد پارامتر
sindy_params=d_emb*2*0 + W.shape[0]*W.shape[1]  # W
oh_params=(V*2)*W_oh.shape[1]
print("\n  پارامتر SINDy: %d"%sindy_params)
print("  پارامتر one-hot: %d"%oh_params)
print()
if test_acc>oh_test+0.05:
    print("  ✓✓✓ SINDy+embedding از one-hot بهتره → فرمول در فضای چگال زندگی می‌کنه")
    print("       حرف تو تأیید شد: معنا توزیع‌شده‌ست، نه نود-به-نود")
elif test_acc>oh_test-0.02:
    print("  ≈ هر دو نزدیک → embedding کمک زیادی نکرد")
else:
    print("  ✗ one-hot بهتره → embedding ضعیف ساخته شد")

# کدوم ویژگی‌ها زنده موندن؟
print("\n  وزن‌های W به ترتیب اهمیت (top 5):")
imp=np.abs(W).sum(axis=1)
order=np.argsort(-imp)[:5]
for i in order:
    print("    %-15s |w|=%.4f"%(names[i],imp[i]))
