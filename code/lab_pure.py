import random, math, time, json
def randn(rng,m,n): return [[rng.gauss(0,0.02) for _ in range(n)] for _ in range(m)]
def make_weights(rng,D,L,FF,V,S):
    W={'emb':randn(rng,V,D),'pos':randn(rng,S,D),'head':randn(rng,D,V)}
    for i in range(L):
        W['qkv_%d'%i]=randn(rng,D,3*D); W['out_%d'%i]=randn(rng,D,D)
        W['ff1_%d'%i]=randn(rng,D,FF); W['ff2_%d'%i]=randn(rng,FF,D)
    return W
def qbin(W):
    Q={}
    for k,m in W.items():
        Q[k]=[]
        for row in m:
            s=sum(abs(z) for z in row)/len(row)
            Q[k].append([(1.0 if z>=0 else -1.0)*s for z in row])
    return Q
def mm(A,B):
    m,n,p=len(A),len(A[0]),len(B[0]); C=[[0.0]*p for _ in range(m)]
    for i in range(m):
        Ai,Ci=A[i],C[i]
        for k in range(n):
            a=Ai[k]
            if a==0: continue
            Bk=B[k]
            for j in range(p): Ci[j]+=a*Bk[j]
    return C
def T(A): return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]
def sm(A):
    out=[]
    for row in A:
        mx=max(row); e=[math.exp(x-mx) for x in row]; s=sum(e)
        out.append([x/s for x in e])
    return out
def fwd(W,tok,L,D):
    Tn=len(tok)
    x=[[W['emb'][tok[t]][d]+W['pos'][t][d] for d in range(D)] for t in range(Tn)]
    hs=[[r[:] for r in x]]; sq=1.0/math.sqrt(D)
    for i in range(L):
        qkv=mm(x,W['qkv_%d'%i])
        q=[r[:D] for r in qkv]; k=[r[D:2*D] for r in qkv]; v=[r[2*D:] for r in qkv]
        sc=[[s*sq for s in row] for row in mm(q,T(k))]
        att=sm(sc); o=mm(att,v); pj=mm(o,W['out_%d'%i])
        x=[[x[t][d]+pj[t][d] for d in range(D)] for t in range(Tn)]
        f1=[[z if z>0 else 0.0 for z in row] for row in mm(x,W['ff1_%d'%i])]
        f2=mm(f1,W['ff2_%d'%i])
        x=[[x[t][d]+f2[t][d] for d in range(D)] for t in range(Tn)]
        hs.append([r[:] for r in x])
    return mm(x,W['head']), hs
def fl(M): return [x for r in M for x in r]
def cs(A,B):
    af,bf=fl(A),fl(B); dot=sum(a*b for a,b in zip(af,bf))
    na=math.sqrt(sum(x*x for x in af)); nb=math.sqrt(sum(x*x for x in bf))
    return 0.0 if na==0 or nb==0 else dot/(na*nb)
def run(name,D,L,FF,V,S=16,seed=42):
    rng=random.Random(seed); tok=[rng.randint(0,V-1) for _ in range(S)]
    W=make_weights(rng,D,L,FF,V,S); Wb=qbin(W)
    t0=time.time(); lf,hf=fwd(W,tok,L,D); tfp=time.time()-t0
    t0=time.time(); lb,hb=fwd(Wb,tok,L,D); tbn=time.time()-t0
    pl=[round(cs(a,b),4) for a,b in zip(hf,hb)]
    np_=sum(len(v)*len(v[0]) for v in W.values()); ns=sum(len(v) for v in W.values())
    s32=np_*4; sbn=np_/8+ns*4
    return {'name':name,'params':np_,'hidden_cosine':round(cs(hf[-1],hb[-1]),4),
            'logits_cosine':round(cs(lf,lb),4),'per_layer':pl,
            'compression_x':round(s32/sbn,2),
            'size_fp32_KB':round(s32/1e3,1),'size_bin_KB':round(sbn/1e3,1),
            'time_fp32_ms':round(tfp*1000,1),'time_bin_ms':round(tbn*1000,1)}
print("="*60); print(" آزمایش دودویی‌سازی نویز transformer (pure python)"); print("="*60)
cfgs=[('tiny',16,2,32,50),('small',32,3,64,100),('medium',48,4,96,200)]
res=[]
for c in cfgs:
    print("\n[+] %s: D=%d L=%d FF=%d V=%d"%c); r=run(*c); res.append(r)
    print("    params:        %s"%format(r['params'],','))
    print("    hidden cosine: %s"%r['hidden_cosine'])
    print("    logits cosine: %s"%r['logits_cosine'])
    print("    per-layer:     %s"%r['per_layer'])
    print("    size: fp32=%.1fKB -> bin=%.1fKB (%.1fx)"%(r['size_fp32_KB'],r['size_bin_KB'],r['compression_x']))
    print("    time: fp32=%.0fms bin=%.0fms"%(r['time_fp32_ms'],r['time_bin_ms']))
print("\n"+"="*60); print(" خلاصه"); print("="*60)
for r in res:
    ok=r['hidden_cosine']>0.85 and r['logits_cosine']>0.85
    print("  %-8s: %s (h=%s, l=%s)"%(r['name'],'PASS' if ok else 'FAIL',r['hidden_cosine'],r['logits_cosine']))
if len(res)>=2:
    dh=res[-1]['hidden_cosine']-res[0]['hidden_cosine']; dl=res[-1]['logits_cosine']-res[0]['logits_cosine']
    print("\n  روند: Δh=%+.4f  Δl=%+.4f"%(dh,dl))
    print("  ➜ "+("فرضیه تایید: مدل بزرگ‌تر، نویز قابل حذف‌تر" if dh>0 and dl>0 else
                 "فرضیه رد: مدل بزرگ‌تر، حساس‌تر" if dh<0 and dl<0 else
                 "سیگنال مختلط"))
open('/data/data/com.termux/files/home/lab1_result.json','w').write(json.dumps(res,indent=2))
print("\n✅ lab1_result.json ذخیره شد")
