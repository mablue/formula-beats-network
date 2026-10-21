import numpy as np, time, sys, random
np.random.seed(42); random.seed(42)
def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-26s [%s] %3d%% %s'%(tag[:26],bar,int(pct*100),msg)); sys.stdout.flush()

# داده (همون lab26)
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

def acc_on(pred,Y):
    d=np.linalg.norm(pred[:,None,:]-emb[None,:,:],axis=2)
    return float(np.mean(d.argmin(1)==Y))

def make_formula(F, D, kernel, seed):
    r=np.random.RandomState(seed)
    W=r.randn(F.shape[1],D).astype(np.float32)*np.sqrt(2.0/F.shape[1])
    b=r.uniform(0,2*np.pi,D).astype(np.float32)
    z=F@W+b
    if kernel=='cos': return np.cos(z)
    if kernel=='sin': return np.sin(z)
    if kernel=='rbf': return np.exp(-z*z)
    if kernel=='elu': return np.maximum(z,0)+1.0
    return np.cos(z)

def eval_ensemble(formulas):
    """formulas = [(kernel, D, seed), ...] → concat features → ridge"""
    if not formulas: return 0,0,0,0
    P_tr=np.concatenate([make_formula(Ftr_tr,D,k,s) for k,D,s in formulas],axis=1)
    P_val=np.concatenate([make_formula(Ftr_val,D,k,s) for k,D,s in formulas],axis=1)
    P_te=np.concatenate([make_formula(Fte,D,k,s) for k,D,s in formulas],axis=1)
    d=P_tr.shape[1]; lam=0.5
    A=P_tr.T@P_tr+lam*np.eye(d)
    Th=np.linalg.solve(A,P_tr.T@Etr_tr)
    val_pred=P_val@Th; val_acc=acc_on(val_pred,Ytr[val_idx])
    te_pred=P_te@Th; te_acc=acc_on(te_pred,Yte); te_loss=float(np.mean((te_pred-Ete)**2))
    par=sum(Ftr.shape[1]*D + D*D_EMB for k,D,s in formulas)
    return val_acc,te_acc,te_loss,par

print("="*60); print(" lab27: کشف چند فرمول با تست و خطا"); print("="*60)
print("baseline most-freq=%.1f%%"% (base_acc*100))
print("شروع: ۱ فرمول cos D=64\n")

THRESH=0.005
formulas=[]
current_val=0
max_formulas=6
D_per=64

for step in range(max_formulas):
    # پیشنهاد: یک فرمول جدید با ۴ کرنل مختلف
    candidates=[]
    for k in ['cos','sin','rbf','elu']:
        seed=100+step*10+hash(k)%1000
        cand=formulas+[(k,D_per,seed)]
        v,t,l,p=eval_ensemble(cand)
        candidates.append((k,seed,v,t,l,p,cand))
    # بهترین
    best=max(candidates,key=lambda x:x[2])
    delta=best[2]-current_val
    if step==0:
        # اولین فرمول — همیشه قبول
        formulas=best[6]; current_val=best[2]
        print("  گام %d: %s D=%d  val=%.1f%% test=%.1f%% params=%d (اولین)"%(
            step+1,best[0],D_per,best[2]*100,best[3]*100,best[5]))
    else:
        status="✓ قبول" if delta>THRESH else "✗ رد"
        print("  گام %d: بهترین گزینه=%s  val=%.1f%% (delta=%+.1f%%) %s"%(
            step+1,best[0],best[2]*100,delta*100,status))
        if delta>THRESH:
            formulas=best[6]; current_val=best[2]
        else:
            print("       → توقف: فرمول‌های بیشتر کمک نمی‌کنن")
            break
    render("grow",step+1,max_formulas,"")
sys.stdout.write('\n'); sys.stdout.flush()

# ارزیابی نهایی
final_val,final_te,final_l,final_p=eval_ensemble(formulas)
print("\n"+"="*60); print(" نتیجه کشف"); print("="*60)
print("  تعداد فرمول کشف‌شده: %d"%len(formulas))
print("  ترکیب نهایی:")
kinds={}
for k,D,s in formulas: kinds[k]=kinds.get(k,0)+1
for k,c in kinds.items(): print("    %s × %d"%(k,c))
print("  val=%.1f%%  test=%.1f%%  params=%d"%(final_val*100,final_te*100,final_p))

# مقایسه با تک‌فرمول بزرگ‌تر
print("\n"+"="*60); print(" مقایسه: چند فرمول کوچک vs یک فرمول بزرگ"); print("="*60)
N=len(formulas)
D_total=N*D_per
v1,t1,l1,p1=eval_ensemble([('cos',D_total,100)])
print("  ۱ فرمول cos D=%d:               test=%.1f%% params=%d"%(D_total,t1*100,p1))
v2,t2,l2,p2=eval_ensemble([('cos',D_total//2,100),('rbf',D_total//2,200)])
print("  cos+rbf (جمع=%d):                test=%.1f%% params=%d"%(D_total,t2*100,p2))
print("  کشف‌شده (%d فرمول × D=%d):       test=%.1f%% params=%d"%(N,D_per,final_te*100,final_p))
print()
if len(formulas)>1 and final_te>t1+0.02:
    print("  ✓✓✓ کشف کرد که چند فرمول لازمه → حرف تو تأیید شد")
    print("       «هوشمندی = توان کشف تعداد و نوع فرمول‌ها»")
elif len(formulas)==1:
    print("  ✓ کشف کرد ۱ فرمول کافیه → نیازی به ترکیب نیست")
else:
    print("  ~ ترکیب چند فرمول بهتر از یک فرمول بزرگ نبود")
