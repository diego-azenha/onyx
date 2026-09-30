#!/usr/bin/env python
"""Premissa 4 (knowledge/premissas/04-gerador-das-quebras.md): censo SEM MODELO do que muda nas quebras,
contra um nulo de negativos com pseudo-tau sorteado da distribuicao real de tau. Diagnostico com rotulos,
nunca usado para treino."""
import pandas as pd, numpy as np
from scipy import stats
X=pd.read_parquet('data/X_train.parquet'); yi=pd.read_parquet('data/y_train_index.parquet')
ids=X.index.get_level_values(0).to_numpy(); per=X.period.to_numpy(); v=X.value.to_numpy(np.float64)
cut=np.flatnonzero(np.diff(ids))+1; ini=np.r_[0,cut]; fim=np.r_[cut,len(ids)]
rng=np.random.default_rng(0)
taus=yi.tau_index[yi.tau_index>=0].to_numpy()
def acf(x,k):
    x=x-x.mean(); d=np.dot(x,x); return np.dot(x[:-k],x[k:])/d if d>0 else 0
def stats_seg(h,post):
    out={'dlogvar':np.log(np.var(post)/np.var(h)), 'dmean_sd':(post.mean()-h.mean())/h.std()}
    for k in (1,2,5,10): out[f'dacf{k}']=acf(post,k)-acf(h,k)
    out['dacf_abs1']=acf(np.abs(post-post.mean()),1)-acf(np.abs(h-h.mean()),1)
    out['dkurt']=stats.kurtosis(post)-stats.kurtosis(h)
    out['dskew']=stats.skew(post)-stats.skew(h)
    dh,dp=np.diff(h),np.diff(post); out['dlogvar_diff']=np.log(np.var(dp)/np.var(dh))
    return out
rows=[]
for a,b in zip(ini,fim):
    sid=ids[a]; tau=int(yi.loc[sid,'tau_index'])
    h=v[a:b][per[a:b]==1]; on=v[a:b][per[a:b]==2]
    if tau<0:
        tau=int(rng.choice(taus))  # pseudo-tau para o nulo
        if tau>=len(on): continue
        tipo='negativo'
    else: tipo='quebra'
    post=on[tau:]
    if len(post)<100: continue
    post=post[:300]  # mesmo comprimento máximo nos dois grupos
    r=stats_seg(h,post); r['tipo']=tipo; r['n']=len(post); rows.append(r)
d=pd.DataFrame(rows)
d.loc[(d.tipo=='quebra')&(d.dlogvar.abs()<=0.3),'tipo']='var_fixa'
d.loc[(d.tipo=='quebra'),'tipo']='var_muda'
neg=d[d.tipo=='negativo']
eixos=[c for c in d.columns if c not in ('tipo','n')]
print(d.tipo.value_counts().to_dict())
print('fração que excede o quantil 95% bilateral do nulo (negativos com pseudo-tau):')
res={}
for tp in ('var_fixa','var_muda'):
    g=d[d.tipo==tp]; res[tp]={}
    for e in eixos:
        lo,hi=neg[e].quantile([0.025,0.975]); res[tp][e]=((g[e]<lo)|(g[e]>hi)).mean()
print(pd.DataFrame(res).round(3).to_string())
g=d[d.tipo=='var_fixa']; lo=neg[eixos].quantile(0.025); hi=neg[eixos].quantile(0.975)
exc=((g[eixos]<lo)|(g[eixos]>hi))
print('var_fixa sem NENHUM eixo fora do nulo: %.3f'%(~exc.any(axis=1)).mean(), ' (esperado por acaso com %d eixos ~%.2f)'%(len(eixos),0.95**len(eixos)))
