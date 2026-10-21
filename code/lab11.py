import random, math, json
def mse(A,B):
    t=0.0; n=0
    for ra,rb in zip(A,B):
        for a,b in zip(ra,rb): t+=(a-b)**2; n+=1
    return t/n
def make_data(rng,n):
    X=[]; Y=[]
    for _ in range(n):
        x=[rng.uniform(-1,1) for _ in range(4)]
        X.append(x); Y.append([math.sin(x[0]*x[1]),x[2]*x[2]+x[3]])
    return X,Y
def init_emitter(rng,d_in):
    return {'pos':[rng.uniform(-1,1) for _ in range(d_in)],
            'dir':[rng.gauss(0,1/math.sqrt(d_in)) for _ in range(d_in)],
            'I':rng.gauss(0,0.3), 'r':rng.uniform(0.4,1.2)}
def emit(x,e):
    diff=[xi-pi for xi,pi in zip(x,e['pos'])]
    dist2=sum(d*d for d in diff); r2=e['r']*e['r']
    radial=math.exp(-dist2/(2*r2))
    directional=sum(di*di_ for di,di_ in zip(diff,e['dir']))
    tanh_d=math.tanh(directional)
    act=e['I']*radial*tanh_d
    return act,(diff,dist2,r2,radial,tanh_d)
def emit_back(x,e,cache,d_act):
    diff,dist2,r2,radial,tanh_d=cache
    dI=radial*tanh_d*d_act
    dr=e['I']*tanh_d*d_act*radial*dist2/(r2*e['r']+1e-9)
    d_dir=[e['I']*radial*(1-tanh_d**2)*diff[j]*d_act for j in range(len(diff))]
    d_pos=[]; d_x=[]
    for j in range(len(diff)):
        dp=e['I']*d_act*((radial*diff[j]/r2)*tanh_d - radial*(1-tanh_d**2)*e['dir'][j])
        d_pos.append(dp); d_x.append(-dp)
    return {'pos':d_pos,'dir':d_dir,'I':dI,'r':dr},d_x
def forward(x,L1,L2,W):
    h1=[]; c1=[]
    for e in L1:
        a,c=emit(x,e); h1.append(a); c1.append(c)
    h2=[]; c2=[]
    for e in L2:
        a,c=emit(h1,e); h2.append(a); c2.append(c)
    y=[sum(h2[k]*W[k][j] for k in range(len(h2))) for j in range(2)]
    return y,(h1,c1,h2,c2)
def backward(x,Y,L1,L2,W,cache,dy):
    h1,c1,h2,c2=cache
    dW=[[h2[k]*dy[j] for j in range(2)] for k in range(len(h2))]
    d_h2=[sum(W[k][j]*dy[j] for j in range(2)) for k in range(len(h2))]
    gL2=[]; d_h1=[0.0]*len(h1)
    for e,c,da in zip(L2,c2,d_h2):
        g,dx=emit_back(h1,e,c,da); gL2.append(g)
        for j in range(len(d_h1)): d_h1[j]+=dx[j]
    gL1=[]; d_x=[0.0]*len(x)
    for e,c,da in zip(L1,c1,d_h1):
        g,dx=emit_back(x,e,c,da); gL1.append(g)
        for j in range(len(d_x)): d_x[j]+=dx[j]
    return gL1,gL2,dW
def train(X_tr,Y_tr,X_te,Y_te,K,epochs=400,lr=0.02,
          train_L1=True,train_L2=True,train_W=True,seed=42):
    rng=random.Random(seed)
    L1=[init_emitter(rng,4) for _ in range(K)]
    L2=[init_emitter(rng,K) for _ in range(K)]
    W=[[rng.gauss(0,0.1) for _ in range(2)] for _ in range(K)]
    n=len(X_tr)
    for ep in range(epochs):
        accL1=[{'pos':[0]*4,'dir':[0]*4,'I':0,'r':0} for _ in range(K)]
        accL2=[{'pos':[0]*K,'dir':[0]*K,'I':0,'r':0} for _ in range(K)]
        accW=[[0.0]*2 for _ in range(K)]
        for xi,yi in zip(X_tr,Y_tr):
            y,cache=forward(xi,L1,L2,W)
            dy=[2*(y[j]-yi[j])/n for j in range(2)]
            g1,g2,gw=backward(xi,yi,L1,L2,W,cache,dy)
            for e_acc,ge in zip(accL1,g1):
                for j in range(4):
                    e_acc['pos'][j]+=ge['pos'][j]; e_acc['dir'][j]+=ge['dir'][j]
                e_acc['I']+=ge['I']; e_acc['r']+=ge['r']
            for e_acc,ge in zip(accL2,g2):
                for j in range(K):
                    e_acc['pos'][j]+=ge['pos'][j]; e_acc['dir'][j]+=ge['dir'][j]
                e_acc['I']+=ge['I']; e_acc['r']+=ge['r']
            for k in range(K):
                for j in range(2): accW[k][j]+=gw[k][j]
        if train_L1:
            for e,ge in zip(L1,accL1):
                for j in range(4):
                    e['pos'][j]-=lr*ge['pos'][j]; e['dir'][j]-=lr*ge['dir'][j]
                e['I']-=lr*ge['I']; e['r']=max(0.15,e['r']-lr*ge['r'])
        if train_L2:
            for e,ge in zip(L2,accL2):
                for j in range(K):
                    e['pos'][j]-=lr*ge['pos'][j]; e['dir'][j]-=lr*ge['dir'][j]
                e['I']-=lr*ge['I']; e['r']=max(0.15,e['r']-lr*ge['r'])
        if train_W:
            for k in range(K):
                for j in range(2): W[k][j]-=lr*accW[k][j]
    loss=0.0
    for xi,yi in zip(X_te,Y_te):
        y,_=forward(xi,L1,L2,W)
        loss+=sum((y[j]-yi[j])**2 for j in range(2))
    loss/=len(X_te)
    n_params=K*10 + K*(2*K+2) + K*2
    return loss,n_params
random.seed(1); rng=random.Random(42)
X_tr,Y_tr=make_data(rng,80); X_te,Y_te=make_data(rng,40)
print("="*60); print(" lab11: Field Emitters vs MLP"); print("="*60)
print("Baseline (4-layer MLP): loss=0.0226  params=2144\n")
R={'baseline':{'loss':0.0226,'params':2144}}
print("%-35s %-10s %-10s %-8s"%("config","loss","params","ratio"))
configs=[
 ('K=8 full-train',8,True,True,True),
 ('K=16 full-train',16,True,True,True),
 ('K=32 full-train',32,True,True,True),
 ('K=8 ELM (readout only)',8,False,False,True),
 ('K=16 ELM',16,False,False,True),
 ('K=32 ELM',32,False,False,True),
 ('K=16 layer1-only-train',16,True,False,True),
]
for name,K,t1,t2,tw in configs:
    loss,nparams=train(X_tr,Y_tr,X_te,Y_te,K,train_L1=t1,train_L2=t2,train_W=tw)
    ratio=loss/0.0226
    print("%-35s %-10.4f %-10d %-8.2fx"%(name,loss,nparams,ratio))
    R[name]={'loss':loss,'params':nparams,'ratio':ratio}
print("\n"+"="*60); print(" جمع‌بندی"); print("="*60)
keys=[k for k in R if k!='baseline']
best=min(keys,key=lambda k:R[k]['ratio'])
print("  بهترین: %s"%(best))
print("  loss=%.4f   ratio=%.2fx   params=%d"%(R[best]['loss'],R[best]['ratio'],R[best]['params']))
if R[best]['ratio']<1.5:
    print("  ✓✓✓ ایده‌ی میدان جواب داد! بدون ماتریس، با پارامتر کمتر")
elif R[best]['ratio']<2.5:
    print("  ~ نیمه‌موفق: میدان کار می‌کنه ولی افت داره")
else:
    print("  ✗ این نسخه جواب نداد — باید مدل میدان رو پیچیده‌تر کنیم")
open('/data/data/com.termux/files/home/lab11_result.json','w').write(json.dumps(R,indent=2))
print("\n✅ lab11_result.json ذخیره شد")
