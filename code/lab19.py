import numpy as np, math, sys, time
from collections import Counter
np.random.seed(42)

def render(tag,step,total,msg,w=15):
    pct=step/total; n=int(pct*w); bar='#'*n+'-'*(w-n)
    line='  %-22s [%s] %3d%% %s'%(tag[:22],bar,int(pct*100),msg)
    sys.stdout.write('\r'+line.ljust(80)); sys.stdout.flush()

# ============================================================
# دیکشنری عبارات (پایه‌های فرمول)
# ============================================================
def build_dict(x, freqs=[1,2,3]):
    """برای بردار x، همه‌ی عبارت‌های ممکن رو تولید می‌کنه"""
    n = len(x); terms = []; names = []
    terms.append(1.0); names.append('1')
    for i in range(n):
        terms.append(x[i]); names.append(f'x{i}')
    for i in range(n):
        terms.append(x[i]**2); names.append(f'x{i}²')
    for i in range(n):
        terms.append(x[i]**3); names.append(f'x{i}³')
    for i in range(n):
        for j in range(i+1,n):
            terms.append(x[i]*x[j]); names.append(f'x{i}·x{j}')
    for f in freqs:
        for i in range(n):
            terms.append(math.sin(f*x[i])); names.append(f'sin({f}x{i})')
            terms.append(math.cos(f*x[i])); names.append(f'cos({f}x{i})')
    for i in range(n):
        terms.append(math.exp(-x[i]**2)); names.append(f'exp(-x{i}²)')
    # عبارات ترکیبی
    for i in range(n):
        for j in range(n):
            if i!=j:
                terms.append(x[i]*math.sin(x[j])); names.append(f'x{i}·sin(x{j})')
                terms.append(x[i]*math.cos(x[j])); names.append(f'x{i}·cos(x{j})')
    return np.array(terms, dtype=np.float32), names

def sindy(X, Y, lambda_l1=0.05, max_iter=300):
    """
    رگرسیون تنک با soft-thresholding
    خروجی: ضرایب تنک + لیست عبارت‌های زنده
    """
    n, d_in = X.shape
    Phi = np.array([build_dict(X[k])[0] for k in range(n)], dtype=np.float32)
    names = build_dict(X[0])[1]
    p = Phi.shape[1]
    # نرمال‌سازی
    norms = np.linalg.norm(Phi, axis=0) + 1e-9
    Phi = Phi / norms
    # soft-thresholding iterative
    w = np.zeros(p, dtype=np.float32)
    for it in range(max_iter):
        r = Y - Phi @ w
        grad = -2 * Phi.T @ r
        # soft threshold
        w_new = w - 0.01 * grad  # lr
        w_new = np.sign(w_new) * np.maximum(np.abs(w_new) - lambda_l1*0.01, 0)
        if np.linalg.norm(w_new - w) < 1e-4:
            break
        w = w_new
    # اصلاح نهایی: بدون نرمال‌سازی
    alive = [i for i in range(p) if abs(w[i]) > 1e-3]
    # least squares روی زیرمجموعه
    if len(alive) > 0:
        A = Phi[:, alive]
        w_sub, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        final_w = np.zeros(p, dtype=np.float32)
        final_w[alive] = w_sub
    else:
        final_w = w
    return final_w, names, Phi

def eval_formula(w, names, Phi, Y, top_k=5):
    """خطا با top-K عبارت"""
    order = np.argsort(-np.abs(w))[:top_k]
    A = Phi[:, order]
    w_sub, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ w_sub
    loss = float(np.mean((pred - Y)**2))
    formula = " + ".join([f'{w_sub[k]:.3f}·{names[order[k]]}' for k in range(len(order))])
    return loss, formula, [names[i] for i in order]

print("="*60); print(" lab19: SINDy — کشف خودکار فرمول"); print("="*60)

# ============================================================
# تست A: داده با فرمول شناخته
# ============================================================
print("\n[A/3] داده با فرمول: x₀·x₁ + sin(x₂)")
n=300; d=3
X = np.random.uniform(-1,1,(n,d)).astype(np.float32)
Y = X[:,0]*X[:,1] + np.sin(X[:,2])
print("  ساخت دیکشنری...")
t0=time.time()
w, names, Phi = sindy(X, Y, lambda_l1=0.05)
print("  زمان: %.1fs | تعداد عبارت در دیکشنری: %d" % (time.time()-t0, len(names)))
loss, formula, top = eval_formula(w, names, Phi, Y)
print("  فرمول کشف‌شده: %s" % formula[:120])
print("  loss: %.6f (اگر <0.001 → موفق)" % loss)
print("  عبارات زنده: %s" % [n for n in names if abs(w[names.index(n)])>1e-3][:8])

# ============================================================
# تست B: فرمول پیچیده‌تر
# ============================================================
print("\n[B/3] داده با فرمول: 2x₀ - x₁² + x₂·sin(x₀)")
Y2 = 2*X[:,0] - X[:,1]**2 + X[:,2]*np.sin(X[:,0])
print("  اجرای SINDy...")
w2, _, _ = sindy(X, Y2, lambda_l1=0.05)
loss2, formula2, _ = eval_formula(w2, names, Phi, Y2)
print("  فرمول: %s" % formula2[:120])
print("  loss: %.6f" % loss2)

# ============================================================
# تست C: زبان
# ============================================================
print("\n[C/3] زبان")
nouns=['cat','dog','bird','fish','tree','house']
verbs=['sat','ate','saw','sang','ran','read']
adjs=['big','small','red','blue','old','new']
places=['park','house','tree','sky']
import random
rng=random.Random(42)
def gen():
    return rng.choice([
        "the {n} {v} the {n}","the {a} {n} {v}",
        "a {a} {n} {v} the {n}","the {n} {v} in the {p}",
    ]).format(n=rng.choice(nouns),v=rng.choice(verbs),a=rng.choice(adjs),p=rng.choice(places)).split()
tr=[gen() for _ in range(200)]
aw=sorted(set(w for s in tr for w in s)); w2i={w:i for i,w in enumerate(aw)}; V=len(aw)
Xl=[]; Yl=[]
ctx=2
for s in tr:
    ids=[w2i[w] for w in s]
    for i in range(ctx,len(ids)): Xl.append(ids[i-ctx:i]); Yl.append(ids[i])
Xl=np.array(Xl,dtype=np.float32); Yl=np.array(Yl,dtype=np.float32)
# ورودی رو به ۲ عدد (میانگین جفت‌ها) و خروجی رو به کد کاهش می‌دیم — SINDy روی عدد پیوسته کار می‌کنه
Xl_red = np.array([[a/10.0, b/10.0] for a,b in Xl], dtype=np.float32)
Yl_norm = Yl/float(V)
print("  کاهش به ۲ بعد پیوسته (SINDy فقط روی عدد کار می‌کنه)")
print("  اجرای SINDy روی زبان...")
t0=time.time()
wl, names_l, Phil = sindy(Xl_red, Yl_norm, lambda_l1=0.05)
print("  زمان: %.1fs" % (time.time()-t0))
lossl, formula_l, top_l = eval_formula(wl, names_l, Phil, Yl_norm, top_k=5)
print("  فرمول کشف‌شده: %s" % formula_l[:150])
print("  loss: %.6f" % lossl)
print("  عبارت‌های زنده: %s" % top_l[:8])

# ============================================================
# جمع‌بندی
# ============================================================
print("\n"+"="*60); print(" جمع‌بندی"); print("="*60)
print("  A: loss=%.6f  %s" % (loss, "✓ موفق" if loss<0.01 else "✗"))
print("  B: loss=%.6f  %s" % (loss2, "✓ موفق" if loss2<0.01 else "✗"))
print("  C: loss=%.6f  %s" % (lossl, "✓ فرمول زبانی" if lossl<0.01 else "✗ فرمول عددی پیدا نشد"))
print()
if loss<0.01 and loss2<0.01:
    print("  ✓✓✓ SINDy روی داده‌ی صاف کار می‌کنه → فرضیه‌ی تو تأیید")
    print("       «فرمول از داده کشف می‌شه، بدون دست‌چین کردن»")
else:
    print("  ✗ SINDy حتی روی داده‌ی صاف هم ضعیف بود → دیکشنری/تنظیمات باید بهتر شه")
if lossl<0.05:
    print("  ⚠ زبان یک فرمول عددی ساده داره (بعیده درست باشه، شاید overfit)")
else:
    print("  زبان: فرمول پیوسته نمی‌ده → باید ساختار گرافی/تانسوری امتحان شه")
