from __future__ import annotations

import numpy as np
import torch

LATENT_DIMS=(4,8,12,16,24,32)
BETAS=(.0001,.0003,.001,.003,.01)
SPACE=tuple((d,b) for d in LATENT_DIMS for b in BETAS)
BASELINE=SPACE.index((16,.001))


def normalized_loss(output,target,latent_dim,beta):
    mse=(output["prediction"]-target).square().mean()
    per_dimension_kl=output["kl"].mean()/latent_dim
    return mse+beta*per_dimension_kl,mse,per_dimension_kl


def initial_ids(seed):
    candidates=np.delete(np.arange(len(SPACE)),BASELINE)
    return [BASELINE]+np.random.default_rng(seed).choice(candidates,3,replace=False).tolist()


def ranking(record):
    return record["selection_score"],record["parameter_count"],record["config_id"]


def search(method,seed,lookup):
    initial=initial_ids(seed)
    records=[];visited=set()
    def query(index,origin):
        assert index not in visited
        result=lookup(int(index))
        record={"query":len(records)+1,"config_id":int(index),"selection_score":result["selection_score"],
                "parameter_count":result["parameter_count"],"fit_id":result.get("fit_id"),"origin":origin}
        records.append(record);visited.add(int(index))
        return record
    population=[query(i,"shared_initial") for i in initial]
    fallbacks=0;proposal_attempts=0
    if method=="random_search":
        remaining=np.array([i for i in range(30) if i not in visited])
        for index in np.random.default_rng(seed+10000).choice(remaining,12,replace=False):query(index,"uniform_without_replacement")
    elif method=="genetic_search":
        rng=np.random.default_rng(seed+20000)
        for generation in range(1,4):
            new=[]
            for _ in range(4):
                attempts=0
                while True:
                    attempts+=1;proposal_attempts+=1
                    if attempts>200:
                        child=int(rng.choice([i for i in range(30) if i not in visited]));fallbacks+=1
                        origin=f"generation_{generation}_fallback"
                    else:
                        def parent():
                            choices=rng.choice(4,2,replace=False)
                            return min([population[int(i)] for i in choices],key=ranking)["config_id"]
                        a,b=parent(),parent()
                        first=np.array(divmod(a,5));second=np.array(divmod(b,5))
                        genes=np.where(rng.random(2)<.5,first,second) if rng.random()<.8 else first.copy()
                        for gene,size in ((0,6),(1,5)):
                            if rng.random()<.3:
                                genes[gene]=rng.choice([i for i in range(size) if i!=genes[gene]])
                        child=int(genes[0]*5+genes[1]);origin=f"generation_{generation}"
                    if child not in visited:break
                new.append(query(child,origin))
            population=sorted(population+new,key=ranking)[:4]
    else:raise ValueError(method)
    assert len(records)==len(visited)==16
    best=min(records,key=ranking)
    running=[]
    for i in range(1,17):running.append(min(records[:i],key=ranking)["selection_score"])
    return {"method":method,"search_seed":seed,"records":records,"best":best,
            "best_score_by_query":running,"proposal_attempts":proposal_attempts,"uniform_fallbacks":fallbacks}
