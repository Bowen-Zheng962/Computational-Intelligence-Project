"""Flight-disjoint preprocessing. Test windows are constructed only in test stage."""
import csv
import json
from pathlib import Path

import numpy as np
import world_model.data as original
from .metrics import sha256,energy


def split_ids(data_dir):
    with (data_dir/"wm_split.csv").open(encoding="utf-8-sig") as f:
        return {int(float(r["flight"])):r["split"] for r in csv.DictReader(f)}


def raw_windows(data_dir,allowed):
    grouped={int(f):[] for f in allowed}
    with (data_dir/"wm_segments.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            flight=int(float(row["flight"]))
            if flight in grouped:grouped[flight].append(row)
    result={k:[] for k in ("history","future","target","flights","segments")}
    for flight,rows in sorted(grouped.items()):
        prepared=original._prepare_flight_rows(sorted(rows,key=lambda r:int(r["segment_index"])))
        usable=[r for r in prepared if original._truthy(r.get("usable_segment"))]
        for run in original._contiguous_runs(usable):
            for origin in range(12,len(run)-6+1):
                history=run[origin-12:origin];future=run[origin:origin+6]
                result["history"].append([[original._history_value(r,n) for n in original.HISTORY_FEATURES] for r in history])
                result["future"].append(original._future_features(history[-1],future))
                result["target"].append([float(r["mean_power_W"]) for r in future])
                result["flights"].append(flight);result["segments"].append([int(r["segment_index"]) for r in future])
    return {k:np.asarray(v,dtype=int if k in ("flights","segments") else float) for k,v in result.items()}


def fit_preprocessing(train):
    state={}
    for name in ("history","future"):
        flat=train[name].reshape(-1,train[name].shape[-1])
        med=original._finite_medians(flat);clean=original._impute(flat,med)
        mean,std=original._mean_std(clean)
        state.update({name+"_medians":med.tolist(),name+"_mean":mean.tolist(),name+"_std":std.tolist()})
    state["target_mean"]=float(train["target"].mean());state["target_std"]=float(train["target"].std())
    return state


def transform(raw,state):
    result={"flights":raw["flights"],"segments":raw["segments"],"target":raw["target"].astype(np.float32)}
    for name in ("history","future"):
        a=np.where(np.isfinite(raw[name]),raw[name],state[name+"_medians"])
        result[name]=((a-state[name+"_mean"])/state[name+"_std"]).astype(np.float32)
    result["target_norm"]=((raw["target"]-state["target_mean"])/state["target_std"]).astype(np.float32)
    return result


def development(data_dir,output):
    ids=split_ids(data_dir)
    raw_train=raw_windows(data_dir,[f for f,s in ids.items() if s=="train"])
    raw_val=raw_windows(data_dir,[f for f,s in ids.items() if s=="val"])
    state=fit_preprocessing(raw_train)
    datasets={"train":transform(raw_train,state),"val":transform(raw_val,state)}
    scale=energy(datasets["train"]["target"]).std(axis=0)
    train_ids=set(datasets["train"]["flights"].tolist());val_ids=set(datasets["val"]["flights"].tolist())
    test_ids={f for f,s in ids.items() if s=="test"}
    assert train_ids.isdisjoint(val_ids) and (train_ids|val_ids).isdisjoint(test_ids)
    np.savez_compressed(output/"development_inputs.npz",**{f"{s}_{k}":v for s,d in datasets.items() for k,v in d.items()})
    return datasets,{"preprocessing_state":state,"target_mean":state["target_mean"],"target_std":state["target_std"],
                     "energy_scale":scale.tolist(),"training_flights":sorted(train_ids),"validation_flights":sorted(val_ids),
                     "test_flights":sorted(test_ids),"training_windows":len(raw_train["flights"]),"validation_windows":len(raw_val["flights"]),
                     "source_data_sha256":{name:sha256(data_dir/name) for name in ("wm_segments.csv","wm_split.csv","preprocess_config.json")},
                     "history_features":original.HISTORY_FEATURES,"future_features":original.FUTURE_FEATURES,
                     "test_windows_constructed_during_fit":False,"normalization_fitted_only_on_training":True}


def test_data(data_dir,state,allowed):
    ids=split_ids(data_dir)
    assert set(allowed)=={f for f,s in ids.items() if s=="test"}
    return transform(raw_windows(data_dir,allowed),state)
