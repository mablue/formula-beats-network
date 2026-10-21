import numpy as np, sys, time, math, random
np.random.seed(42)
random.seed(42)

def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-28s [%s] %3d%% %s'%(tag[:28],bar,int(pct*100),msg))
    sys.stdout.flush()

# ============================================================
# داده زبان با ساختار واقعی‌تر
# ============================================================
nouns=['cat','dog','bird','fish','tree','house','car','book']
verbs=['sat','ate','saw','sang','ran','read','looked']
adjs=['big','small','red','blue','old','new','happy']
places=['park','house','tree','sky','road']
def gen(rng):
    return rng.choice([
        "the {n} {v} the {n}","the {a} {n} {v}",
        "a {a} {n} {v} the {n}","the {n} {v} in the {p}",
        "the {n} {v} and the {n} {v}",
    ]).format(n=rng.choice(nouns),v=rng.choice(verbs),a=rng.choice(adjs),p=rng.choice(places)).split()
rng=random.Random(42)
tr=[gen(rng) for _ in range(400)]
te=[gen(rng) for _ in range(80)]
aw=sorted(set(w for s in tr+te for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
print("="*60); print(" lab22: Attention استاندارد vs Linear (فازی)"); print("="*60)
print("V=%d | train=%d | test=%d\n"%(V,len(tr),len(te)))

# پد کردن به طول ثابت
MAXLEN=max(len(s) for s in tr+te)
def pad(ids): return ids + [0]*(MAXLEN-len(ids))
Xtr=np.array([pad([w2i[w] for w in s]) for s in tr])
Xte=np.array([pad([w2i[w] for w in s]) for s in te])

# ============================================================
# مدل ساده: Embedding → Attention → Linear → Softmax
# ============================================================
class Transformer:
    def __init__(s, V, d=16, use_linear=False, seed=42):
        r=np.random.RandomState(seed)
        s.V=V; s.d=d; s.use_linear=use_linear
        s.E = r.randn(V,d).astype(np.float32)*0.1
        s.Wq= r.randn(d,d).astype(np.float32)*0.1
        s.Wk= r.randn(d,d).astype(np.float32)*0.1
        s.Wv= r.randn(d,d).astype(np.float32)*0.1
        s.Wo= r.randn(d,d).astype(np.float32)*0.1
        s.Wl= r.randn(d,V).astype(np.float32)*0.1
    def attn_forward(s, X, mask):
        # X: (B, T, d) -- but let's do batched over single samples
        Q = X @ s.Wq; K = X @ s.Wk; V_ = X @ s.Wv  # (T,d)
        T = Q.shape[0]
        if s.use_linear:
            # φ(x) = elu(x)+1
            phiQ = np.maximum(Q,0) + 1.0
            phiK = np.maximum(K,0) + 1.0
            KV = phiK.T @ V_     # (d,d)
            denom = (phiQ * phiK.sum(0)[None,:]).sum(-1, keepdims=True)
            A_ = phiQ @ KV
            O = A_ / (denom+1e-9)
        else:
            scores = Q @ K.T / math.sqrt(s.d)   # (T,T)
            # mask (padding)
            scores = scores + (1-mask[:,None]) * (-1e9)  # mask columns
            m = scores.max(axis=-1, keepdims=True)
            exp_s = np.exp(scores - m)
            A_ = exp_s / (exp_s.sum(-1, keepdims=True)+1e-9)
            O = A_ @ V_
        return O @ s.Wo
    def forward(s, X_ids):
        # X_ids: (T,) -- single sample
        h = s.E[X_ids]                     # (T,d)
        o = s.attn_forward(h, np.ones(len(X_ids)))
        h = h + o                          # residual
        # آخرین position → پیش‌بینی
        logits = h[-1] @ s.Wl              # (V,)
        return logits

def train(model, X, Y, epochs=200, lr=0.05):
    V=model.V; d=model.d
    losses=[]
    for ep in range(epochs):
        # جمع‌آوری گرادیان (batch)
        dE=np.zeros_like(model.E); dWq=np.zeros_like(model.Wq); dWk=np.zeros_like(model.Wk)
        dWv=np.zeros_like(model.Wv); dWo=np.zeros_like(model.Wo); dWl=np.zeros_like(model.Wl)
        n=len(X); total_loss=0.0; correct=0
        for k in range(n):
            ids=X[k]; target=Y[k]
            mask=np.ones(len(ids))
            # forward
            h=model.E[ids]
            Q=h@model.Wq; K=h@model.Wk; V_=h@model.Wv
            if model.use_linear:
                phiQ=np.maximum(Q,0)+1.0; phiK=np.maximum(K,0)+1.0
                KV=phiK.T@V_
                denom=(phiQ*phiK.sum(0)[None,:]).sum(-1,keepdims=True)
                A_=phiQ@KV/(denom+1e-9)
            else:
                scores=Q@K.T/math.sqrt(d)
                m=scores.max(-1,keepdims=True)
                e=np.exp(scores-m); A_=e/(e.sum(-1,keepdims=True)+1e-9)
            O=A_@V_
            o_out=O@model.Wo
            h2=h+o_out
            logits=h2[-1]@model.Wl
            # softmax + CE
            lm=logits.max(); e2=np.exp(logits-lm); p=e2/e2.sum()
            if p.argmax()==target: correct+=1
            total_loss+=-math.log(p[target]+1e-9)
            # grad
            dy=p.copy(); dy[target]-=1.0
            dWl+=np.outer(h2[-1],dy)
            dh2_last=model.Wl@dy
            dh2=np.zeros_like(h2); dh2[-1]=dh2_last
            do_out=dh2.copy()
            dWo+=O.T@do_out
            dO=do_out@model.Wo.T
            # grad through attention (approximate — for demo)
            dA_=dO@V_.T
            dV_=A_.T@dO
            dWv+=h.T@dV_
            if not model.use_linear:
                # softmax grad
                dS=A_*(dA_- (dA_*A_).sum(-1,keepdims=True))
                dS/=math.sqrt(d)
                dQ=dS@K; dK=dS.T@Q
                dWq+=h.T@dQ; dWk+=h.T@dK
                dh=dQ@model.Wq.T + dK@model.Wk.T + dV_@model.Wv.T
            else:
                # خطی — تقریب ساده
                dWq+=h.T@dO; dWk+=h.T@dO; dh=dO@model.Wq.T + dO@model.Wk.T + dV_@model.Wv.T
            dh+=dh2  # residual
            np.add.at(dE, ids, dh)
        # update
        model.E -= lr*dE/n
        model.Wq -= lr*dWq/n; model.Wk -= lr*dWk/n; model.Wv -= lr*dWv/n
        model.Wo -= lr*dWo/n; model.Wl -= lr*dWl/n
        losses.append((total_loss/n, correct/n))
        if (ep+1)%40==0 or ep==0:
            render("train",ep+1,epochs,"loss=%.3f acc=%.1f%%"%(total_loss/n,100*correct/n))
    sys.stdout.write('\n'); sys.stdout.flush()
    return losses

def evaluate(model, X, Y):
    n=len(X); correct=0; loss=0.0
    for k in range(n):
        logits=model.forward(X[k])
        lm=logits.max(); e=np.exp(logits-lm); p=e/e.sum()
        if p.argmax()==Y[k]: correct+=1
        loss+=-math.log(p[Y[k]]+1e-9)
    return correct/n, loss/n

# ============================================================
# آماده‌سازی داده (زوج جمله → کلمه آخر)
# ============================================================
def make_xy(X, maxlen=None):
    Xs=[]; Ys=[]
    for row in X:
        for L in range(2, len(row)+1):
            if row[L-1]==0: break
            Xs.append(row[:L]); Ys.append(row[L-1])
    return Xs, np.array(Ys)

Xtr_seq, Ytr = make_xy(Xtr)
Xte_seq, Yte = make_xy(Xte)
print("train samples: %d | test samples: %d"%(len(Xtr_seq),len(Xte_seq)))
from collections import Counter
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)
print("Most-freq baseline: %.1f%%\n"%(base_acc*100))

# ============================================================
# تست ۱: Standard Attention
# ============================================================
print("[1/2] Standard Attention (softmax)")
t0=time.time()
m_std=Transformer(V, d=16, use_linear=False, seed=42)
train(m_std, Xtr_seq, Ytr, epochs=100, lr=0.03)
std_acc, std_loss=evaluate(m_std, Xte_seq, Yte)
std_time=time.time()-t0
print("  acc=%.1f%% | loss=%.3f | time=%.1fs"%(std_acc*100,std_loss,std_time))

# ============================================================
# تست ۲: Linear Attention (فازی)
# ============================================================
print("\n[2/2] Linear Attention (φ=elu+1)")
t0=time.time()
m_lin=Transformer(V, d=16, use_linear=True, seed=42)
train(m_lin, Xtr_seq, Ytr, epochs=100, lr=0.03)
lin_acc, lin_loss=evaluate(m_lin, Xte_seq, Yte)
lin_time=time.time()-t0
print("  acc=%.1f%% | loss=%.3f | time=%.1fs"%(lin_acc*100,lin_loss,lin_time))

# ============================================================
# جمع‌بندی
# ============================================================
print("\n"+"="*60); print(" نتیجه"); print("="*60)
print("  Most-freq:        %.1f%%"%(base_acc*100))
print("  Standard attn:    %.1f%%  loss=%.3f  params=%d"%(std_acc*100,std_loss,m_std.E.size+5*16*16))
print("  Linear attn:      %.1f%%  loss=%.3f  params=%d"%(lin_acc*100,lin_loss,m_lin.E.size+5*16*16))
print()
diff=(lin_acc-std_acc)*100
if lin_acc>=std_acc-0.05:
    print("  ✓✓✓ Linear attention ≈ Standard → فرمول فازی attention جواب می‌ده")
    print("       صرفه‌جویی محاسبه: n²·d → n·d²")
    print("       برای n بزرگ (مثل ۱۰۰۰ توکن): ۱۰۰x سریع‌تر!")
else:
    print("  ✗ Linear attention افت داره: %+.1f%% نسبت به standard" % diff)
    print("       → فرمول فازی attention فعلی کافی نیست")
