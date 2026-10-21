import random, math, json
def randn(rng,m,n): return [[rng.gauss(0,0.02) for _ in range(n)] for _ in range(m)]
def mk(rng,D,L,FF,V,S):
    W={'emb':randn(rng,V,D),'pos':randn(rng,S,D),'head':randn(rng,D,V)}
    for i in range(L):
        W['qkv_%d'%i]=randn(rng,D,3*D); W['out_%d'%i]=randn(rng,D,D)
        W['ff1_%d'%i]=randn(rng,D,FF); W['ff2_%d'%i]=randn(rng,FF,D)
    return W
def qrow(row,bits):
    if bits>=32: return row[:]
    mx=max(abs(z) for z in row) or 1e-9
    half=2**(bits-1)
    return [round(z/mx*half)/half*mx for z in row]
def qbits(W,bits): return {k:[qrow(r,bits) for r in m] for k,m in W.items()}
def qctrl(W):
    Q={}
    for k,m in W.items():
        Q[k]=[]
        for row in m:
            s=sum(abs(z) for z in row)/len(row)
            Q[k].append([(1.0 if random.random()>0.5 else -1.0)*s for _ in row])
    return Q
def qsel(W,bits,sel):
    return {k:([qrow(r,bits) for r in m] if sel(k) else [r[:] for r in m]) for k,m in W.items()}
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
    Tn=len(tok); sq=1.0/math.sqrt(D)
    x=[[W['emb'][tok[t]][d]+W['pos'][t][d] for d in range(D)] for t in range(Tn)]
    hs=[x]
    for i in range(L):
        qkv=mm(x,W['qkv_%d'%i])
        q=[r[:D] for r in qkv]; k=[r[D:2*D] for r in qkv]; v=[r[2*D:] for r in qkv]
        sc=[[s*sq for s in row] for row in mm(q,T(k))]
        att=sm(sc); o=mm(att,v); pj=mm(o,W['out_%d'%i])
        x=[[x[t][d]+pj[t][d] for d in range(D)] for t in range(Tn)]
        f1=[[z if z>0 else 0.0 for z in row] for row in mm(x,W['ff1_%d'%i])]
        f2=mm(f1,W['ff2_%d'%i])
        x=[[x[t][d]+f2[t][d] for d in range(D)] for t in range(Tn)]
        hs.append(x)
    return mm(x,W['head']), hs
def fl(M): return [x for r in M for x in r]
def cs(A,B):
    af,bf=fl(A),fl(B); d=sum(a*b for a,b in zip(af,bf))
    na=math.sqrt(sum(x*x for x in af)); nb=math.sqrt(sum(x*x for x in bf))
    return 0.0 if na==0 or nb==0 else d/(na*nb)
def am(A,B):
    m=0
    for ra,rb in zip(A,B): m+=(ra.index(max(ra))==rb.index(max(rb)))
    return m/len(A)
random.seed(1)
rng=random.Random(42); S=16; D,L,FF,V=32,3,64,100
tok=[rng.randint(0,V-1) for _ in range(S)]
W=mk(rng,D,L,FF,V,S)
lf,hf=fwd(W,tok,L,D)
n_params=sum(len(v)*len(v[0]) for v in W.values())
print("="*60); print(" آزمایش ۲: بیت‌سوییپ + حساسیت لایه"); print("="*60)
print("config: D=%d L=%d FF=%d V=%d | params=%d"%(D,L,FF,V,n_params))
print("\n--- ۲a: بیت‌سوییپ ---")
res=[]
for bits in [1,2,3,4,6,8]:
    Q=qbits(W,bits); lq,hq=fwd(Q,tok,L,D)
    r={'bits':bits,'h_cos':round(cs(hf[-1],hq[-1]),4),
       'l_cos':round(cs(lf,lq),4),'argmax':round(am(lf,lq),4)}
    res.append(r)
    print("  %2d-bit: h=%.3f l=%.3f argmax=%.1f%%   (فشرده‌سازی نظری %dx)"%(
        bits,r['h_cos'],r['l_cos'],r['argmax']*100,32//bits))
print("\n--- ۲b: کنترل (sign تصادفی، همان مقیاس) ---")
Qc=qctrl(W); lc,hc=fwd(Qc,tok,L,D)
print("  ctrl : h=%.3f l=%.3f argmax=%.1f%%"%(
    cs(hf[-1],hc[-1]),cs(lf,lc),am(lf,lc)*100))
print("\n--- ۲c: حساسیت لایه‌ها (در ۴ بیت) ---")
for i in range(L):
    Q=qsel(W,4,lambda k,i=i: k.endswith('_%d'%i)); lq,hq=fwd(Q,tok,L,D)
    print("  فقط لایه %d      : h=%.3f l=%.3f argmax=%.1f%%"%(
        i,cs(hf[-1],hq[-1]),cs(lf,lq),am(lf,lq)*100))
    Q=qsel(W,4,lambda k,i=i: not k.endswith('_%d'%i)); lq,hq=fwd(Q,tok,L,D)
    print("  همه جز لایه %d   : h=%.3f l=%.3f argmax=%.1f%%"%(
        i,cs(hf[-1],hq[-1]),cs(lf,lq),am(lf,lq)*100))
Q=qsel(W,4,lambda k: k in ('emb','pos','head')); lq,hq=fwd(Q,tok,L,D)
print("  فقط emb+pos+head : h=%.3f l=%.3f argmax=%.1f%%"%(
    cs(hf[-1],hq[-1]),cs(lf,lq),am(lf,lq)*100))
Q=qsel(W,4,lambda k: k not in ('emb','pos','head')); lq,hq=fwd(Q,tok,L,D)
print("  همه جز emb+head  : h=%.3f l=%.3f argmax=%.1f%%"%(
    cs(hf[-1],hq[-1]),cs(lf,lq),am(lf,lq)*100))
open('/data/data/com.termux/files/home/lab2_result.json','w').write(
    json.dumps({'bit_sweep':res,'config':{'D':D,'L':L,'FF':FF,'V':V}},indent=2))
print("\n✅ lab2_result.json ذخیره شد")
