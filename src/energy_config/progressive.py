"""Resume exact training trajectories, charging incremental epochs for promotions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .core import SPACE,normalized_loss
from .training import Trainer


class ProgressiveTrainer(Trainer):
    def fit(self,index,seed,epochs,patience=0,label="search"):
        assert patience==0 and epochs in (10,20)
        key=self.key(index,seed,epochs,0)
        if key in self.cache:return self.cache[key],True
        # Every 20-epoch request preserves the exact 10-epoch rung for later
        # lower-fidelity queries. Total actual work remains 10+10=20 epochs.
        if epochs==20 and self.key(index,seed,10,0) not in self.cache:
            self.fit(index,seed,10,0,label+"_midpoint")
        previous=self.cache.get(self.key(index,seed,10,0)) if epochs==20 else None
        latent_dim,beta=SPACE[index]
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if self.device.type=="cuda":torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model=self.model(latent_dim);optimizer=torch.optim.Adam(model.parameters(),lr=.001,foreach=False)
        rng=np.random.default_rng(seed+1000)
        if previous:
            cp=torch.load(self.output/previous["checkpoint"],map_location="cpu",weights_only=True)
            model.load_state_dict(cp["last_model_state_dict"]);optimizer.load_state_dict(cp["optimizer_state_dict"])
            rng.bit_generator.state=cp["order_rng_state"]
            torch.set_rng_state(cp["torch_rng_state"])
            if self.device.type=="cuda":torch.cuda.set_rng_state_all(cp["cuda_rng_states"])
            best=previous["selection_score"];best_epoch=previous["best_epoch"]
            best_state=cp["model_state_dict"];initial_digest=cp["initial_state_sha256"]
            history=cp["history"];start_epoch=10
        else:
            hasher=hashlib.sha256()
            for name,value in model.state_dict().items():hasher.update(name.encode());hasher.update(value.detach().cpu().numpy().tobytes())
            initial_digest=hasher.hexdigest();best=float("inf");best_epoch=0;best_state=None;history=[];start_epoch=0
        began=time.perf_counter()
        for epoch in range(start_epoch+1,epochs+1):
            model.train();order=rng.permutation(len(self.train["history"]));totals=np.zeros(3)
            for start in range(0,len(order),128):
                ix=torch.as_tensor(order[start:start+128],device=self.device)
                batch={k:v[ix] for k,v in self.train.items()}
                optimizer.zero_grad(set_to_none=True)
                out=model(batch["history"],batch["future"],target=batch["target_norm"],sample=True)
                loss,mse,kl=normalized_loss(out,batch["target_norm"],latent_dim,beta)
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite progressive loss")
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
                totals+=np.array([v.detach().item() for v in (loss,mse,kl)])*len(ix)
            score=self.score(model)
            history.append({"epoch":epoch,"training_loss":float(totals[0]/len(order)),"power_mse":float(totals[1]/len(order)),
                            "kl_per_latent_dimension":float(totals[2]/len(order)),"selection_score":score})
            if score<best:
                best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        folder=self.output/"fits"/f"fit_{len(self.all_fits):03d}";folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/"model.pt"
        state={"model_state_dict":best_state,"last_model_state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
               "optimizer_state_dict":optimizer.state_dict(),"order_rng_state":rng.bit_generator.state,
               "torch_rng_state":torch.get_rng_state(),"cuda_rng_states":torch.cuda.get_rng_state_all() if self.device.type=="cuda" else [],
               "history":history,"initial_state_sha256":initial_digest,"model":"rssm","latent_dim":latent_dim,"beta":beta,
               "config_id":index,"seed":seed,"epoch_budget":epochs,"best_epoch":best_epoch,"selection_score":best,
               "preprocessing_state":self.meta["preprocessing_state"]}
        torch.save(state,checkpoint)
        with (folder/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        elapsed=time.perf_counter()-began
        record={"fit_id":len(self.all_fits),"config_id":index,"seed":seed,"epoch_budget":epochs,"patience":0,
                "epochs_ran":epochs,"incremental_epochs":epochs-start_epoch,"resumed_from_fit_id":previous["fit_id"] if previous else None,
                "latent_dim":latent_dim,"beta":beta,"selection_score":best,"best_epoch":best_epoch,
                "parameter_count":sum(p.numel() for p in model.parameters()),"initial_state_sha256":initial_digest,
                "elapsed_seconds":elapsed,"cumulative_seconds":elapsed+(previous["cumulative_seconds"] if previous else 0),
                "checkpoint":checkpoint.relative_to(self.output).as_posix(),"label":label}
        self.all_fits.append(record);self.cache[key]=record
        (self.output/"fits.json").write_text(json.dumps(self.all_fits,indent=2)+"\n",encoding="utf-8")
        print(f"search z={latent_dim:2d} beta={beta:g} seed={seed} epoch={epochs} (+{epochs-start_epoch}) J={best:.5f}",flush=True)
        return record,False
