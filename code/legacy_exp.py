import numpy as np, time, json

def make_weights(D, L, FF, V, S, seed=42):
    rng = np.random.default_rng(seed)
    W = {
        'emb':  rng.normal(0, 0.02, (V, D)),
        'pos':  rng.normal(0, 0.02, (S, D)),
        'head': rng.normal(0, 0.02, (D, V)),
    }
    for i in range(L):
        W[f'qkv_{i}'] = rng.normal(0, 0.02, (D, 3*D))
        W[f'out_{i}'] = rng.normal(0, 0.02, (D, D))
        W[f'ff1_{i}'] = rng.normal(0, 0.02, (D, FF))
        W[f'ff2_{i}'] = rng.normal(0, 0.02, (FF, D))
    return W

def quantize_binary(W):
    Q = {}
    for k, v in W.items():
        if v.ndim == 2:
            scale = np.abs(v).mean(axis=-1, keepdims=True)
            Q[k] = (np.sign(v) * scale).astype(np.float32)
        else:
            Q[k] = v.astype(np.float32)
    return Q

def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

def forward(W, tokens, L, D):
    B, T = tokens.shape
    x = W['emb'][tokens] + W['pos'][:T][None]
    snaps = [x.copy()]
    for i in range(L):
        qkv = x @ W[f'qkv_{i}']
        q, k, v = np.split(qkv, 3, axis=-1)
        attn = softmax(q @ k.transpose(0,2,1) / np.sqrt(D))
        x = x + (attn @ v) @ W[f'out_{i}']
        x = x + np.maximum(0, x @ W[f'ff1_{i}']) @ W[f'ff2_{i}']
        snaps.append(x.copy())
    logits = x @ W['head']
    return logits, snaps

def cos(a, b):
    a=a.flatten(); b=b.flatten()
    na,nb = np.linalg.norm(a), np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(a@b/(na*nb))

def run_config(name, D, L, FF, V, S=32, seed=42):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, V, (2, S))
    W = make_weights(D, L, FF, V, S, seed)
    W_bin = quantize_binary(W)
    logits_f, h_f = forward(W, tokens, L, D)
    logits_b, h_b = forward(W_bin, tokens, L, D)

    per_layer = [{'layer':i, 'cosine': round(cos(hf,hb),4)}
                 for i,(hf,hb) in enumerate(zip(h_f,h_b))]

    n_params = sum(v.size for v in W.values())
    size_fp32 = n_params * 4
    n_scales = sum(v.shape[-2] if v.ndim==2 else 1 for v in W.values())
    size_bin = n_params/8 + n_scales*4

    t0=time.time()
    for _ in range(10): forward(W, tokens, L, D)
    t_fp32=(time.time()-t0)/10
    t0=time.time()
    for _ in range(10): forward(W_bin, tokens, L, D)
    t_bin=(time.time()-t0)/10

    return {
        'name': name, 'params': n_params,
        'hidden_cosine': round(cos(h_f[-1], h_b[-1]), 4),
        'logits_cosine': round(cos(logits_f, logits_b), 4),
        'per_layer_cosine': per_layer,
        'size_fp32_MB': round(size_fp32/1e6, 3),
        'size_bin_MB': round(size_bin/1e6, 3),
        'compression_x': round(size_fp32/size_bin, 1),
        'time_fp32_ms': round(t_fp32*1000, 1),
        'time_bin_ms': round(t_bin*1000, 1),
        'speedup_x': round(t_fp32/t_bin, 2),
    }

print("="*60)
print(" آزمایش دودویی‌سازی نویز transformer")
print("="*60)

configs = [
    ('tiny',   32,  2,  128,  100),
    ('small',  64,  4,  256,  500),
    ('medium', 128, 4,  512, 1000),
]

results = []
for name, D, L, FF, V in configs:
    print(f"\n[+] {name}: D={D} L={L} FF={FF} V={V}")
    r = run_config(name, D, L, FF, V)
    results.append(r)
    print(f"    params:        {r['params']:,}")
    print(f"    hidden cosine: {r['hidden_cosine']}")
    print(f"    logits cosine: {r['logits_cosine']}")
    print(f"    compression:   {r['compression_x']}x")
    print(f"    speedup:       {r['speedup_x']}x")
    print(f"    per-layer:     {[p['cosine'] for p in r['per_layer_cosine']]}")

print("\n" + "="*60)
print(" خلاصه")
print("="*60)
for r in results:
    ok = r['hidden_cosine'] > 0.85 and r['logits_cosine'] > 0.85
    print(f"  {r['name']:8s}: {'PASS' if ok else 'FAIL'} "
          f"(h={r['hidden_cosine']}, l={r['logits_cosine']})")

if len(results) >= 2:
    trend = results[-1]['hidden_cosine'] - results[0]['hidden_cosine']
    print(f"\n  روند با بزرگ شدن مدل: {trend:+.4f}")
    if trend > 0:
        print("  ➜ فرضیه تایید: مدل بزرگ‌تر، نویز قابل حذف‌تر")
    else:
        print("  ➜ فرضیه رد: مدل بزرگ‌تر، حساس‌تر به باینری")

with open('/data/data/com.termux/files/home/lab1_result.json','w') as f:
    json.dump(results, f, indent=2)

print("\n✅ JSON در lab1_result.json ذخیره شد")
