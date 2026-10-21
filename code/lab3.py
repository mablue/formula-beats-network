import random, math, json
def randn(rng,m,n): return [[rng.gauss(0,0.02) for _ in range(n)] for _ in range(m)]
def mk(rng,D,L,FF,V,S):
    W={'emb':randn(rng,V,D),'pos':randn(rng,S,D),'head':randn(rng,D,V)}
    for i in range(L):
        W['qkv_%d'%i]=randn(rng,D,3*D); W['out_%d'%i]=randn(rng,D,D)
        W['ff1_%d'%i]=randn(rng,D,FF); W['ff2_%d'%i]=randn(rng,FF,D)
    return W
def qrow(row,bits):
    mx=max(abs(z) for z in row) or 1e-9
    half=2**(bits-1)
    return [round(z/mx*half)/half*mx for z in row]
def quant(W,sel,bits):
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
def fl(M): return [x for r in M for x in r]
def cs(A,B):
    af,bf=fl(A),fl(B); d=sum(a*b for a,b in zip(af,bf))
    na=math.sqrt(sum(x*x for x in af)); nb=math.sqrt(sum(x*x for x in bf))
    return 0.0 if na==0 or nb==0 else d/(na*nb)
def am(A,B):
    return sum((ra.index(max(ra))==rb.index(max(rb))) for ra,rb in zip(A,B))/len(A)
def fwd(W,tok,L,D,skip=()):
    Tn=len(tok); sq=1.0/math.sqrt(D)
    x=[[W['emb'][tok[t]][d]+W['pos'][t][d] for d in range(D)] for t in range(Tn)]
    hs=[x]; contrib=[]
    for i in range(L):
        xin=x
        if i in skip:
            contrib.append(1.0); hs.append(x); continue
        qkv=mm(x,W['qkv_%d'%i])
        q=[r[:D] for r in qkv]; k=[r[D:2*D] for r in qkv]; v=[r[2*D:] for r in qkv]
        sc=[[s*sq for s in row] for row in mm(q,T(k))]
        att=sm(sc); o=mm(att,v); pj=mm(o,W['out_%d'%i])
        x=[[x[t][d]+pj[t][d] for d in range(D)] for t in range(Tn)]
        f1=[[z if z>0 else 0.0 for z in row] for row in mm(x,W['ff1_%d'%i])]
        f2=mm(f1,W['ff2_%d'%i])
        x=[[x[t][d]+f2[t][d] for d in range(D)] for t in range(Tn)]
        contrib.append(round(cs(xin,x),4)); hs.append(x)
    return mm(x,W['head']), hs, contrib
random.seed(7)
rng=random.Random(42); S=48; D,L,FF,V=32,4,64,100
tok=[rng.randint(0,V-1) for _ in range(S)]
W=mk(rng,D,L,FF,V,S)
lf,hf,cf=fwd(W,tok,L,D)
R={'config':{'D':D,'L':L,'FF':FF,'V':V,'S':S},'layer_contribution':cf}
def score(label,Q,skip=()):
    lq,hq,_=fwd(Q,tok,L,D,skip)
    return {'label':label,'h_cos':round(cs(hf[-1],hq[-1]),4),
            'l_cos':round(cs(lf,lq),4),'argmax':round(am(lf,lq),4)}
print("="*60); print(" آزمایش ۳: سیگنال کجاست؟"); print("="*60)
print("config: D=%d L=%d FF=%d V=%d | %d پیش‌بینی"%(D,L,FF,V,S))
print("\nمشارکت هر لایه (cos input→output، کوچک = قابل حذف):")
for i,c in enumerate(cf): print("  layer %d: %.4f"%(i,c))
print("\n--- فقط emb+pos+head در ۴ بیت ---")
R['edge_only']=score('edge@4b',quant(W,lambda k:k in('emb','pos','head'),4))
print("  h=%.3f l=%.3f argmax=%.1f%%"%(R['edge_only']['h_cos'],R['edge_only']['l_cos'],R['edge_only']['argmax']*100))
print("\n--- هر لایه تنها در ۴ بیت ---")
R['per_layer_quant']=[]
for i in range(L):
    r=score('layer_%d@4b'%i,quant(W,lambda k,i=i:k.endswith('_%d'%i),4))
    R['per_layer_quant'].append(r)
    print("  layer %d: h=%.3f l=%.3f argmax=%.1f%%"%(i,r['h_cos'],r['l_cos'],r['argmax']*100))
print("\n--- همه لایه‌ها در ۴ بیت ---")
R['all_4b']=score('all@4b',quant(W,lambda k:True,4))
print("  h=%.3f l=%.3f argmax=%.1f%%"%(R['all_4b']['h_cos'],R['all_4b']['l_cos'],R['all_4b']['argmax']*100))
print("\n--- همه جز emb+head ---")
R['no_edge_4b']=score('no-edge@4b',quant(W,lambda k:k not in('emb','pos','head'),4))
print("  h=%.3f l=%.3f argmax=%.1f%%"%(R['no_edge_4b']['h_cos'],R['no_edge_4b']['l_cos'],R['no_edge_4b']['argmax']*100))
print("\n--- حذف کامل هر لایه (identity) ---")
R['skip']=[]
for i in range(L):
    r=score('skip_%d'%i,W,skip=(i,))
    R['skip'].append(r)
    print("  skip layer %d: h=%.3f l=%.3f argmax=%.1f%%"%(i,r['h_cos'],r['l_cos'],r['argmax']*100))
print("\n"+"="*60); print(" نتیجه‌گیری"); print("="*60)
lowest=cf.index(min(cf)); highest=cf.index(max(cf))
print("  کم‌مشارکت‌ترین لایه: layer %d (%.4f)"%(lowest,cf[lowest]))
print("  پر‌مشارکت‌ترین لایه: layer %d (%.4f)"%(highest,cf[highest]))
lc_l=[r['l_cos'] for r in R['per_layer_quant']]
print("  حساس‌ترین لایه به کوانت (کمترین l_cos): layer %d"%lc_l.index(min(lc_l)))
print("  حساس‌ترین لایه به skip (کمترین l_cos):  layer %d"%[r['l_cos'] for r in R['skip']].index(min([r['l_cos'] for r in R['skip']])))
open('/data/data/com.termux/files/home/lab3_result.json','w').write(json.dumps(R,indent=2))
print("\n✅ lab3_result.json ذخیره شد")
