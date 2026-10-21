import numpy as np, sys, time, math, random
np.random.seed(42); random.seed(42)
def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-24s [%s] %3d%% %s'%(tag[:24],bar,int(pct*100),msg)); sys.stdout.flush()

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
tr=[gen(rng) for _ in range(200)]; te=[gen(rng) for _ in range(50)]
aw=sorted(set(w for s in tr+te for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
MAXLEN=max(len(s) for s in tr+te)
def pad(ids): return ids+[0]*(MAXLEN-len(ids))
def make_xy(X):
    Xs=[]; Ys=[]
    for row in X:
        for L in range(2,len(row)+1):
            if row[L-1]==0: break
            Xs.append(row[:L]); Ys.append(row[L-1])
    return Xs, np.array(Ys)
Xtr_seq,Ytr=make_xy([pad([w2i[w] for w in s]) for s in tr])
Xte_seq,Yte=make_xy([pad([w2i[w] for w in s]) for s in te])
from collections import Counter
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)
print("="*60); print(" lab22b: Standard vs Linear Attention (fix)"); print("="*60)
print("V=%d train=%d test=%d Most-freq=%.1f%%\n"%(V,len(Xtr_seq),len(Xte_seq),base_acc*100))

class T:
    def __init__(s,V,d,linear,seed=42):
        r=np.random.RandomState(seed)
        s.V=V; s.d=d; s.linear=linear
        s.E=r.randn(V,d).astype(np.float32)*0.1
        s.Wq=r.randn(d,d).astype(np.float32)*0.1
        s.Wk=r.randn(d,d).astype(np.float32)*0.1
        s.Wv=r.randn(d,d).astype(np.float32)*0.1
        s.Wl=r.randn(d,V).astype(np.float32)*0.1
    def fwd(s, ids):
        Tn=len(ids); d=s.d
        h=s.E[ids]
        Q=h@s.Wq; K=h@s.Wk; V_=h@s.Wv
        if s.linear:
            phiQ=np.maximum(Q,0)+1.0    # (T,d)
            phiK=np.maximum(K,0)+1.0    # (T,d)
            KV = phiK.T @ V_            # (d,d)
            O = phiQ @ KV               # (T,d)  ← درست!
            denom = phiK.sum(0)         # (d,)
            O = O / (denom[None,:] + 1e-9)
        else:
            scores=Q@K.T/math.sqrt(d)
            m=scores.max(-1,keepdims=True)
            e=np.exp(scores-m); A_=e/(e.sum(-1,keepdims=True)+1e-9)
            O=A_@V_
        h2=h+O
        logits=h2[-1]@s.Wl
        return logits

def ev(m,X,Y):
    c=0; L=0.0
    for k in range(len(X)):
        lo=m.fwd(X[k]); lm=lo.max(); e=np.exp(lo-lm); p=e/e.sum()
        if p.argmax()==Y[k]: c+=1
        L+=-math.log(p[Y[k]]+1e-9)
    return c/len(X), L/len(X)

def train(m,X,Y,epochs,lr,tag):
    d=m.d; n=len(X)
    for ep in range(epochs):
        dE=np.zeros_like(m.E)
        dWq=np.zeros_like(m.Wq); dWk=np.zeros_like(m.Wk); dWv=np.zeros_like(m.Wv); dWl=np.zeros_like(m.Wl)
        tl=0.0; corr=0
        for k in range(n):
            ids=X[k]; tgt=Y[k]
            h=m.E[ids]
            Q=h@m.Wq; K=h@m.Wk; V_=h@m.Wv
            if m.linear:
                phiQ=np.maximum(Q,0)+1.0; phiK=np.maximum(K,0)+1.0
                KV=phiK.T@V_; O=phiQ@KV/(phiK.sum(0)[None,:]+1e-9)
            else:
                scores=Q@K.T/math.sqrt(d); mm=scores.max(-1,keepdims=True)
                e=np.exp(scores-mm); A_=e/(e.sum(-1,keepdims=True)+1e-9); O=A_@V_
            h2=h+O; logits=h2[-1]@m.Wl
            lm=logits.max(); e2=np.exp(logits-lm); p=e2/e2.sum()
            if p.argmax()==tgt: corr+=1
            tl+=-math.log(p[tgt]+1e-9)
            # backprop (simplified)
            dy=p.copy(); dy[tgt]-=1.0
            dWl+=np.outer(h2[-1],dy)
            dh2=np.zeros_like(h2); dh2[-1]=m.Wl@dy
            dO=dh2.copy()
            dV_=dO  # rough
            dWv+=V_.T@dV_
            if m.linear:
                dphiQ=dO@KV.T/(phiK.sum(0)[None,:].mean()+1e-9)
                dphiK=np.zeros_like(phiK)
                dKV=phiQ.T@dO/(phiK.sum(0)[None,:]+1e-9)
                dphiK+=V_@dKV.T
                dQ=dphiQ*(Q>0); dK=dphiK*(K>0)
            else:
                dA_=dO@V_.T
                dS=A_*(dA_-(dA_*A_).sum(-1,keepdims=True))/math.sqrt(d)
                dQ=dS@K; dK=dS.T@Q
            dWq+=h.T@dQ; dWk+=h.T@dK
            dh=dQ@m.Wq.T+dK@m.Wk.T+dV_@m.Wv.T+dh2
            np.add.at(dE,ids,dh)
        m.E-=lr*dE/n
        m.Wq-=lr*dWq/n; m.Wk-=lr*dWk/n; m.Wv-=lr*dWv/n; m.Wl-=lr*dWl/n
        if (ep+1)%25==0 or ep==0:
            a,l=ev(m,Xte_seq,Yte); render(tag,ep+1,epochs,"loss=%.3f acc=%.1f%%"%(l,100*a))
    sys.stdout.write('\n'); sys.stdout.flush()

print("[1/2] Standard Attention")
t0=time.time(); m_std=T(V,16,False); train(m_std,Xtr_seq,Ytr,50,0.03,"std")
a_std,l_std=ev(m_std,Xte_seq,Yte); t_std=time.time()-t0
print("  → acc=%.1f%% loss=%.3f time=%.1fs"%(a_std*100,l_std,t_std))

print("\n[2/2] Linear Attention (فازی)")
t0=time.time(); m_lin=T(V,16,True); train(m_lin,Xtr_seq,Ytr,50,0.03,"linear")
a_lin,l_lin=ev(m_lin,Xte_seq,Yte); t_lin=time.time()-t0
print("  → acc=%.1f%% loss=%.3f time=%.1fs"%(a_lin*100,l_lin,t_lin))

print("\n"+"="*60); print(" نتیجه"); print("="*60)
print("  Most-freq:     %.1f%%"%(base_acc*100))
print("  Standard:      %.1f%%  loss=%.3f"%(a_std*100,l_std))
print("  Linear:        %.1f%%  loss=%.3f"%(a_lin*100,l_lin))
print()
if a_lin>=a_std-0.05:
    print("  ✓✓✓ Linear ≈ Standard → فرمول فازی attention جواب داد")
else:
    print("  ✗ Linear %.1f%% ضعیف‌تر از Standard" % ((a_lin-a_std)*100))
