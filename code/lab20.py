import numpy as np, sys, time, math
np.random.seed(42)

def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    sys.stdout.write('\r  %-24s [%s] %3d%% %s'%(tag[:24],bar,int(pct*100),msg))
    sys.stdout.flush()

# ============================================================
# داده زبان
# ============================================================
import random
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
sents=[gen() for _ in range(300)]
aw=sorted(set(w for s in sents for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
print("="*60); print(" lab20: فرمول تانسوری زبان"); print("="*60)
print("V=%d | %d جمله | %d توکن\n"%(V,len(sents),sum(len(s) for s in sents)))

# ============================================================
# ساخت تانسور سه‌بعدی P(k | i,j)
# ============================================================
T = np.zeros((V,V,V), dtype=np.float32)
for s in sents:
    ids=[w2i[w] for w in s]
    for i in range(2,len(ids)):
        T[ids[i-2], ids[i-1], ids[i]] += 1.0
# نرمال‌سازی: هر (i,j) یه توزیع احتمال
counts = T.sum(axis=2, keepdims=True) + 1e-9
T_prob = T / counts

# train/test split
mask = np.random.rand(V,V) < 0.85
T_train = T_prob * mask[:,:,None]
T_test  = T_prob * (~mask)[:,:,None]

# baseline: most-frequent
flat = T.reshape(-1,V)
pred_base = flat.argmax(axis=1)
print("Baseline (most-frequent per (i,j)):")
base_loss = 0.0; base_n = 0
for i in range(V):
    for j in range(V):
        if not mask[i,j]: continue
        k = T_prob[i,j].argmax()
        # cross-entropy
        p = T_prob[i,j,k] + 1e-9
        base_loss += -math.log(p); base_n += 1
base_acc = base_n
print("  خالی (چون فقط احتمال max رو داره، معیار بهتر: MSE)")
mse_base = float(np.mean((flat.argmax(axis=1).reshape(V,V,1) - T_prob)**2))
print("  loss (MSE): %.4f" % mse_base)

# ============================================================
# CP Decomposition (ALS — Alternating Least Squares)
# ============================================================
def cp_als(T, r, n_iter=50, reg=1e-3):
    V = T.shape[0]
    A = np.random.randn(V,r).astype(np.float32)*0.1
    B = np.random.randn(V,r).astype(np.float32)*0.1
    C = np.random.randn(V,r).astype(np.float32)*0.1
    T_mat = T.reshape(V*V, V)  # (V*V, V)
    for it in range(n_iter):
        # A update: reshape as (V, V*V) with i as first dim
        T_i = T.transpose(0,1,2).reshape(V, V*V)  # (V, V*V)
        M = (B[:,None,:] * C[None,:,:]).reshape(V*V, r)  # (V*V, r)
        A = np.linalg.solve(M.T @ M + reg*np.eye(r), M.T @ T_i.T).T
        # B update
        T_j = T.transpose(1,0,2).reshape(V, V*V)
        M = (A[:,None,:] * C[None,:,:]).reshape(V*V, r)
        B = np.linalg.solve(M.T @ M + reg*np.eye(r), M.T @ T_j.T).T
        # C update (direct)
        M = (A[:,None,:] * B[None,:,:]).reshape(V*V, r)
        C = np.linalg.solve(M.T @ M + reg*np.eye(r), M.T @ T_mat).T
    return A, B, C

def cp_predict(A,B,C):
    """T_pred[i,j,k] = Σ_r A[i,r]·B[j,r]·C[k,r]"""
    return np.einsum('ir,jr,kr->ijk', A, B, C)

def eval_cp(A,B,C):
    T_pred = cp_predict(A,B,C)
    mse = float(np.mean((T_pred - T_prob)**2))
    # cross-entropy روی نمونه‌های test
    ce = 0.0; n = 0
    for i in range(V):
        for j in range(V):
            if mask[i,j]: continue
            logits = T_pred[i,j]
            logits = logits - logits.max()
            p = np.exp(logits) / (np.exp(logits).sum() + 1e-9)
            true_k = T_prob[i,j].argmax()
            ce += -math.log(p[true_k] + 1e-9); n += 1
    return mse, ce/max(n,1)

# ============================================================
# تست رتبه‌های مختلف
# ============================================================
print("\n"+"="*60); print(" تست CP با رتبه‌های مختلف"); print("="*60)
print("%-6s %-12s %-12s %-8s"%("r","MSE","CE(test)","params"))
results = []
for r in [2, 3, 5, 8, 12, 20]:
    t0=time.time()
    A,B,C = cp_als(T_train, r, n_iter=30)
    mse, ce = eval_cp(A,B,C)
    params = 3 * V * r
    results.append((r, mse, ce, params, time.time()-t0))
    print("%-6d %-12.6f %-12.4f %-8d (%.1fs)"%(r, mse, ce, params, time.time()-t0))

# baseline
A_rand=np.random.randn(V,1)*0.1
mse_rand = float(np.mean((cp_predict(A_rand,A_rand,A_rand)-T_prob)**2))
print("\nBaseline تصادفی (r=1): MSE=%.6f" % mse_rand)
print("Baseline ماتریس کامل: params=%d" % (V*V*V))

# ============================================================
# تحلیل
# ============================================================
print("\n"+"="*60); print(" تحلیل"); print("="*60)
best = min(results, key=lambda x: x[1])
print("  بهترین رتبه: r=%d  (MSE=%.6f)" % (best[0], best[1]))
print("  پارامترهای CP: %d (در برابر %d ماتریس کامل → %.1fx فشرده‌سازی)" % (
    best[3], V*V*V, V*V*V/best[3]))
# آیا افزایش رتبه خیلی کمک می‌کنه؟
mse_r2 = results[0][1]
mse_r20 = results[-1][1]
gain = (mse_r2 - mse_r20) / mse_r2 * 100
print("  افزایش r از %d به %d: %.1f%% بهبود" % (results[0][0], results[-1][0], gain))
print()
if gain < 20:
    print("  ✓✓✓ زبان رتبه‌ی پایین داره → فرمول تانسوری ساده کافیه")
    print("       یعنی: %d عامل، کل توزیع رو توضیح می‌ده" % best[0])
else:
    print("  ✗ افزایش رتبه مدام بهتر می‌شه → زبان رتبه‌ی بالا داره")
    print("       فرمول تانسوری ساده کافی نیست، r بزرگ لازمه")
