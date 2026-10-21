import random, math, sys

train_sentences = [
    "the cat sat on the mat".split(),
    "the cat ate the fish".split(),
    "the cat saw the bird".split(),
    "the dog sat on the log".split(),
    "the dog ate the bone".split(),
    "the dog saw the cat".split(),
    "the bird sang in the tree".split(),
    "the bird flew to the sky".split(),
    "the bird saw the dog".split(),
    "the cat sat on the log".split(),
    "the dog sat on the mat".split(),
]
test_sentences = [
    "the cat ate the bone".split(),
    "the dog saw the bird".split(),
    "the bird sat on the tree".split(),
]

all_words = sorted(set(w for s in train_sentences+test_sentences for w in s))
word2id = {w:i for i,w in enumerate(all_words)}
V = len(all_words)

def make_xy(sentences, ctx=2):
    X, Y = [], []
    for s in sentences:
        ids = [word2id[w] for w in s]
        for i in range(ctx, len(ids)):
            X.append(ids[i-ctx:i]); Y.append(ids[i])
    return X, Y

X_tr, Y_tr = make_xy(train_sentences)
X_te, Y_te = make_xy(test_sentences)

def onehot_ctx(ctx_ids):
    v = [0.0] * (V * len(ctx_ids))
    for c, cid in enumerate(ctx_ids):
        v[c * V + cid] = 1.0
    return v

def render(tag, ep, total, loss, acc, width=15):
    pct=ep/total; n=int(pct*width); bar='#'*n+'-'*(width-n)
    sys.stdout.write('\r  %-24s [%s] %3d%%  loss=%.3f acc=%.0f%%'%(tag[:24],bar,int(pct*100),loss,acc*100))
    sys.stdout.flush()

class MLP:
    def __init__(self, d_in, d_h, d_out, seed=42):
        rng = random.Random(seed)
        self.W1 = [[rng.gauss(0, 1.0/math.sqrt(d_in)) for _ in range(d_h)] for _ in range(d_in)]
        self.b1 = [0.0]*d_h
        self.W2 = [[rng.gauss(0, 1.0/math.sqrt(d_h)) for _ in range(d_out)] for _ in range(d_h)]
        self.b2 = [0.0]*d_out
    def forward(self, x):
        z = [sum(x[i]*self.W1[i][j] for i in range(len(x))) + self.b1[j] for j in range(len(self.b1))]
        h = [v if v > 0 else 0.0 for v in z]
        out = [sum(h[i]*self.W2[i][j] for i in range(len(h))) + self.b2[j] for j in range(len(self.b2))]
        return out, z, h
    def backward(self, x, y, z, h, lr):
        out = [sum(h[i]*self.W2[i][j] for i in range(len(h))) + self.b2[j] for j in range(len(self.b2))]
        m = max(out); e = [math.exp(o-m) for o in out]; s = sum(e)
        p = [ei/s for ei in e]
        dy = p[:]; dy[y] -= 1.0
        dW2 = [[h[i]*dy[j] for j in range(len(dy))] for i in range(len(h))]
        dh = [sum(self.W2[i][j]*dy[j] for j in range(len(dy))) for i in range(len(h))]
        dz = [dh[i] * (1.0 if z[i] > 0 else 0.0) for i in range(len(z))]
        dW1 = [[x[i]*dz[j] for j in range(len(dz))] for i in range(len(x))]
        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])): self.W1[i][j] -= lr * dW1[i][j]
        for j in range(len(self.b1)): self.b1[j] -= lr * dz[j]
        for i in range(len(self.W2)):
            for j in range(len(self.W2[0])): self.W2[i][j] -= lr * dW2[i][j]
        for j in range(len(self.b2)): self.b2[j] -= lr * dy[j]

class FuzzyTSK:
    def __init__(self, d_in, d_out, K, centers_init, seed=42):
        rng = random.Random(seed)
        self.c = [c[:] for c in centers_init]
        self.s = [2.0 for _ in range(K)]
        self.b = [[rng.gauss(0, 0.1) for _ in range(d_out)] for _ in range(K)]
    def forward(self, x):
        mus = []
        for k in range(len(self.c)):
            d2 = sum((x[i]-self.c[k][i])**2 for i in range(len(x)))
            mus.append(math.exp(-d2 / (2*self.s[k]**2 + 1e-9)))
        Z = sum(mus) + 1e-9
        out = [sum(mus[k]*self.b[k][j] for k in range(len(mus)))/Z for j in range(len(self.b[0]))]
        return out, mus, Z
    def backward(self, x, y, mus, Z, lr):
        out = [sum(mus[k]*self.b[k][j] for k in range(len(mus)))/Z for j in range(len(self.b[0]))]
        m = max(out); e = [math.exp(o-m) for o in out]; ss = sum(e)
        p = [ei/ss for ei in e]
        dy = p[:]; dy[y] -= 1.0
        for k in range(len(self.c)):
            for j in range(len(self.b[0])):
                self.b[k][j] -= lr * dy[j] * mus[k] / Z
            dm = sum(dy[j] * (self.b[k][j] - out[j]) for j in range(len(dy))) / Z
            d2 = sum((x[i]-self.c[k][i])**2 for i in range(len(x)))
            for i in range(len(self.c[k])):
                dmu = mus[k] * (x[i]-self.c[k][i]) / (self.s[k]**2 + 1e-9)
                self.c[k][i] -= lr * dm * dmu
            dmu_s = mus[k] * d2 / (self.s[k]**3 + 1e-9)
            self.s[k] = max(0.3, self.s[k] - lr * dm * dmu_s)

def evaluate(model, X, Y, kind):
    correct, total_loss = 0, 0.0
    for xids, y in zip(X, Y):
        x = onehot_ctx(xids)
        out = model.forward(x)[0]
        m = max(out); e = [math.exp(o-m) for o in out]; s = sum(e)
        p = [ei/s for ei in e]
        if p.index(max(p)) == y: correct += 1
        total_loss += -math.log(p[y] + 1e-9)
    return correct/len(X), total_loss/len(X)

def init_centers(X, K, seed=42):
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    centers = [onehot_ctx(X[idx[i % len(X)]]) for i in range(K)]
    for k in range(K):
        for i in range(len(centers[k])):
            centers[k][i] += rng.gauss(0, 0.05)
    return centers

print("="*60); print(" lab14: میدان فازی روی زبان — کورپس کوچیک ولی واقعی"); print("="*60)
print("واژگان: %d کلمه | نمونه‌های آموزش: %d | نمونه‌های تست: %d" % (V, len(X_tr), len(X_te)))
print("ورودی: ۲ کلمه‌ی آخر (%d بعد one-hot)" % (V*2))
print("خروجی: کلمه‌ی بعدی از %d کلاس\n" % V)

d_in = V*2; d_h = 20; d_out = V
mlp_params = d_in*d_h + d_h + d_h*d_out + d_out
K = 20
fuzzy_params = K*(d_in + 1 + d_out)

print("[1/2] MLP baseline (h=%d, params≈%d)" % (d_h, mlp_params))
mlp = MLP(d_in, d_h, d_out, seed=42)
lr_mlp = 0.3
for ep in range(400):
    for xids, y in zip(X_tr, Y_tr):
        x = onehot_ctx(xids)
        out, z, h = mlp.forward(x)
        mlp.backward(x, y, z, h, lr_mlp)
    if (ep+1) % 50 == 0 or ep == 0:
        acc, loss = evaluate(mlp, X_te, Y_te, 'mlp')
        render("MLP h=%d"%d_h, ep+1, 400, loss, acc)
sys.stdout.write('\n'); sys.stdout.flush()
mlp_acc, mlp_loss = evaluate(mlp, X_te, Y_te, 'mlp')

print("\n[2/2] Fuzzy TSK (K=%d, params≈%d، بدون ماتریس)" % (K, fuzzy_params))
centers = init_centers(X_tr, K, seed=42)
fz = FuzzyTSK(d_in, d_out, K, centers, seed=42)
lr_fz = 0.3
for ep in range(400):
    for xids, y in zip(X_tr, Y_tr):
        x = onehot_ctx(xids)
        out, mus, Z = fz.forward(x)
        fz.backward(x, y, mus, Z, lr_fz)
    if (ep+1) % 50 == 0 or ep == 0:
        acc, loss = evaluate(fz, X_te, Y_te, 'fz')
        render("Fuzzy K=%d"%K, ep+1, 400, loss, acc)
sys.stdout.write('\n'); sys.stdout.flush()
fz_acc, fz_loss = evaluate(fz, X_te, Y_te, 'fz')

print("\n"+"="*60); print(" نتیجه"); print("="*60)
print("  MLP:    acc=%.1f%%  loss=%.3f  params=%d" % (mlp_acc*100, mlp_loss, mlp_params))
print("  Fuzzy:  acc=%.1f%%  loss=%.3f  params=%d" % (fz_acc*100, fz_loss, fuzzy_params))
print()
if fz_acc >= mlp_acc - 0.05 and fz_loss <= mlp_loss*1.5:
    print("  ✓✓✓ میدان فازی روی زبان جواب داد!")
    print("       بدون هیچ ماتریس چگال، هم‌رده با MLP")
elif fz_acc >= mlp_acc - 0.15:
    print("  ~ نیمه‌موفق: فازی نزدیکه ولی افت داره")
else:
    print("  ✗ فازی روی زبان شکست خورد")
    print("       → میدان فازی فقط برای توابع صاف جواب می‌ده، نه ساختار زبانی")
open('/data/data/com.termux/files/home/lab14_result.txt','w').write(
    "MLP: acc=%.4f loss=%.4f params=%d\nFuzzy: acc=%.4f loss=%.4f params=%d\n" % (
        mlp_acc, mlp_loss, mlp_params, fz_acc, fz_loss, fuzzy_params))
print("\n✅ lab14_result.txt ذخیره شد")
