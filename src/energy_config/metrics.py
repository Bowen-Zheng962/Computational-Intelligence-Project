import hashlib
from collections import Counter
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def energy(power):
    return np.asarray(power,dtype=float).cumsum(axis=-1)[...,[1,3,5]]*(5/3600)


def flight_weights(ids):
    counts=Counter(int(v) for v in ids)
    return np.array([1/(len(counts)*counts[int(v)]) for v in ids])


def evaluate(power,target,ids,scale):
    power=np.asarray(power,dtype=float);target=np.asarray(target,dtype=float)
    ee=energy(power)-energy(target);pe=power-target
    rows=[]
    for flight in sorted(set(ids.tolist())):
        mask=ids==flight
        r={"flight":int(flight),"windows":int(mask.sum()),"energy_score":float((np.abs(ee[mask])/scale).mean()),
           "power_mae_w":float(np.abs(pe[mask]).mean()),"power_mse_w2":float((pe[mask]**2).mean()),"power_bias_w":float(pe[mask].mean())}
        for j,h in enumerate((10,20,30)):
            r[f"energy_mae_{h}s_wh"]=float(np.abs(ee[mask,j]).mean());r[f"energy_bias_{h}s_wh"]=float(ee[mask,j].mean())
        for j in range(6):r[f"power_mae_step{j+1}_w"]=float(np.abs(pe[mask,j]).mean())
        rows.append(r)
    result={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("flight","windows")}
    result["power_rmse_w"]=float(np.sqrt(result.pop("power_mse_w2")))
    return result,rows
