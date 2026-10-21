import numpy as np, time, sys, random
np.random.seed(42); random.seed(42)
def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-26s [%s] %3d%% %s'%(tag[:26],bar,int(pct*100),msg)); sys.stdout.flush()

# ---------- داده ----------
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
# split داخلی برای validation
n_val=len(Xtr)//5
perm=np.random.RandomState(0).permutation(len(Xtr))
val_idx=perm[:n_val]; tr_idx=perm[n_val:]

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
Etr_tr=emb[Ytr[tr_idx]]; Etr_val=emb[Ytr[val_idx]]
Etr=emb[Ytr]; Ete=emb[Yte]
from collections import Counter
mc=Counter(Ytr).most_common(1)[0][0]
base_acc=sum(1 for y in Yte if y==mc)/len(Yte)

def acc_on(pred, Y):
    d=np.linalg.norm(pred[:,None,:]-emb[None,:,:],axis=2)
    return float(np.mean(d.argmin(1)==Y))

def build_features(F, layers, Ds, seed):
    r=np.random.RandomState(seed)
    cur=F
    for i,(din,dout) in enumerate(zip([F.shape[1]]+Ds[:-1], Ds)):
        W=r.randn(din,dout).astype(np.float32)*np.sqrt(2.0/din)
        b=r.uniform(0,2*np.pi,dout).astype(np.float32)
        cur=np.cos(cur@W+b)
    return cur

def fit_and_eval(layers, Ds, seed=100, lam=0.5):
    # fit on train subset
    P_tr=build_features(Ftr_tr, layers, Ds, seed)
    P_val=build_features(Ftr_val, layers, Ds, seed)
    P_te =build_features(Fte, layers, Ds, seed)
    d=P_tr.shape[1]
    A=P_tr.T@P_tr+lam*np.eye(d)
    Th=np.linalg.solve(A,P_tr.T@Etr_tr)
    val_pred=P_val@Th; val_acc=acc_on(val_pred,Ytr[val_idx])
    te_pred=P_te@Th; te_acc=acc_on(te_pred,Yte); te_loss=float(np.mean((te_pred-Ete)**2))
    par=Ftr.shape[1]*Ds[0]
    for i in range(1,len(Ds)): par+=Ds[i-1]*Ds[i]
    par+=Ds[-1]*D_EMB
    return val_acc, te_acc, te_loss, par

print("="*60); print(" lab26: ساختار خودش رو کشف می‌کنه"); print("="*60)
print("baseline most-freq=%.1f%%\n"%(base_acc*100))

THRESH=0.01  # ۱٪ بهبود حداقل

# ---------- A: رشد بعد ----------
print("[A/3] رشد بعد — از D=8 تا D=1024")
prev_val=0; chosen_D=0; history=[]
for D in [8,16,32,64,128,256,512,1024]:
    v,t,l,p=fit_and_eval([1],[D])
    history.append((D,v,t,p))
    delta=v-prev_val
    verdict="✓ قبول" if delta>THRESH or D==8 else "✗ رد"
    print("  D=%-5d val=%.1f%% test=%.1f%% params=%-7d delta=%+.1f%% %s"%(
        D,v*100,t*100,p,delta*100,verdict))
    if D==8 or delta>THRESH:
        prev_val=v; chosen_D=D
    render("A",[8,16,32,64,128,256,512,1024].index(D)+1,8,"")
sys.stdout.write('\n'); sys.stdout.flush()
print("  → انتخاب شد: D=%d (با val=%.1f%%)"%(chosen_D,prev_val*100))

# ---------- B: رشد لایه (در D=256 ثابت) ----------
print("\n[B/3] رشد لایه — از ۱ تا ۴ (D=256 هر لایه)")
prev_val=0; chosen_L=1; history_L=[]
for L in [1,2,3,4]:
    Ds=[256]*L
    v,t,l,p=fit_and_eval([1]*L,Ds)
    history_L.append((L,v,t,p))
    delta=v-prev_val
    verdict="✓ قبول" if delta>THRESH or L==1 else "✗ رد"
    print("  L=%d  val=%.1f%% test=%.1f%% params=%-7d delta=%+.1f%% %s"%(
        L,v*100,t*100,p,delta*100,verdict))
    if L==1 or delta>THRESH:
        prev_val=v; chosen_L=L
    render("B",L,4,"")
sys.stdout.write('\n'); sys.stdout.flush()
print("  → انتخاب شد: L=%d (با val=%.1f%%)"%(chosen_L,prev_val*100))

# ---------- C: ترکیب ----------
print("\n[C/3] رشد ترکیبی — بعد و لایه همزمان")
# از D=chosen_D شروع، هر گام: یا D رو ۲x کن، یا L رو +1، انتخاب کن چی بهتره
cur_D=chosen_D; cur_L=1
v,t,l,p=fit_and_eval([1],[cur_D])
best_val=v
print("  شروع: D=%d L=1 → val=%.1f%%"%(cur_D,best_val*100))
for step in range(4):
    options=[]
    # گزینه ۱: D دو برابر
    if cur_D<1024:
        v2,t2,l2,p2=fit_and_eval([1],[cur_D*2])
        options.append(("D→%d"%(cur_D*2),v2,t2,p2,cur_D*2,cur_L))
    # گزینه ۲: L +1
    if cur_L<3:
        v2,t2,l2,p2=fit_and_eval([1]*cur_L,[cur_D]*cur_L)
    # better: L+1 با همان D
    if cur_L<3:
        v2,t2,l2,p2=fit_and_eval([1]*(cur_L+1),[cur_D]*(cur_L+1))
        options.append(("L→%d"%(cur_L+1),v2,t2,l2,p2,cur_D,cur_L+1))
    if not options: break
    best_opt=max(options,key=lambda x:x[1])
    delta=best_opt[1]-best_val
    if delta>THRESH:
        cur_D=best_opt[4]; cur_L=best_opt[5]; best_val=best_opt[1]
        print("  گام %d: %s → val=%.1f%% (delta=+%.1f%%) ✓"%(
            step+1,best_opt[0],best_opt[1]*100,delta*100))
    else:
        print("  گام %d: بهترین گزینه=%s delta=+%.1f%% < آستانه → توقف"%(
            step+1,best_opt[0],delta*100))
        break
    render("C",step+1,4,"")
sys.stdout.write('\n'); sys.stdout.flush()

# ---------- نتیجه نهایی ----------
print("\n"+"="*60); print(" مقایسه: ساختار کشف‌شده vs ساختار تحمیلی"); print("="*60)
grown_val,grown_te,grown_loss,grown_par=fit_and_eval([1]*cur_L,[cur_D]*cur_L)
print("  ساختار کشف‌شده:    D=%d L=%d  → val=%.1f%% test=%.1f%% params=%d"%(
    cur_D,cur_L,grown_val*100,grown_te*100,grown_par))

# تحمیلی: D=512 L=1 (بهترین single از lab24)
imp_v,imp_t,imp_l,imp_p=fit_and_eval([1],[512])
print("  تحمیلی (D=512 L=1):               → val=%.1f%% test=%.1f%% params=%d"%(
    imp_v*100,imp_t*100,imp_p))

# تحمیلی: D=256 L=2
imp2_v,imp2_t,imp2_l,imp2_p=fit_and_eval([1,1],[256,256])
print("  تحمیلی (D=256 L=2):               → val=%.1f%% test=%.1f%% params=%d"%(
    imp2_v*100,imp2_t*100,imp2_p))

print()
if grown_te>=imp_t-0.01 and grown_par<=imp_p:
    print("  ✓✓✓ ساختار کشف‌شده با پارامتر کمتر/مساوی، accuracy مشابه/بهتر گرفت")
    print("       حرف تو اثبات شد: ساختار نباید تحمیل شود")
elif grown_te>imp_t+0.02:
    print("  ✓✓✓ ساختار کشف‌شده حتی بهتر از تحمیلی کار کرد")
    print("       حرف تو اثبات شد: کشف ساختار > تحمیل ساختار")
else:
    print("  ~ هر دو نزدیک → کشف ساختار تفاوت زیادی نداد")

print("\n"+"="*60); print(" خلاصه کشف"); print("="*60)
print("  عمق لازم: %d لایه"%chosen_L)
print("  بعد لازم: %d"%chosen_D)
print("  ترکیب نهایی: D=%d L=%d"%(cur_D,cur_L))
