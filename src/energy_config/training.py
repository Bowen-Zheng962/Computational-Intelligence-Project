from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import random
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from .metrics import energy,flight_weights
from .core import SPACE,normalized_loss

class Trainer:
    def __init__(self,output,datasets,selection_mask,meta,build_model):
        self.output=output;self.datasets=datasets;self.meta=meta;self.build_model=build_model
        self.device=torch.device(meta["device"])
        def gpu(data):return {k:torch.from_numpy(v).to(self.device) for k,v in data.items() if k!="flights"}
        self.train=gpu(datasets["train"])
        self.validation=gpu(datasets["val"])
        self.selection={k:v[selection_mask] for k,v in self.validation.items()}
        self.y_energy=energy(datasets["val"]["target"][selection_mask])
        self.weights=flight_weights(datasets["val"]["flights"][selection_mask])
        self.scale=np.asarray(meta["energy_scale"])
        self.all_fits=[];self.cache={}
        if (output/"fits.json").exists():
            self.all_fits=json.loads((output/"fits.json").read_text(encoding="utf-8"))
            for record in self.all_fits:self.cache[self.key(record["config_id"],record["seed"],record["epoch_budget"],record["patience"])]=record

    @staticmethod
    def key(index,seed,epochs,patience):return f"{index}_{seed}_{epochs}_{patience}"

    def model(self,latent_dim):
        return self.build_model("rssm",history_dim=self.train["history"].shape[-1],future_dim=self.train["future"].shape[-1],
                                hidden_dim=64,latent_dim=latent_dim,step_seconds=5).to(self.device)

    def forecast(self,model,split):
        model.eval()
        with torch.inference_mode():
            prediction=model(split["history"],split["future"],target=None,sample=False)["prediction"]
            power=(prediction*self.meta["target_std"]+self.meta["target_mean"]).cpu().numpy()
        assert np.isfinite(power).all()
        return power

    def score(self,model):
        power=self.forecast(model,self.selection)
        return float(self.weights@(np.abs(energy(power)-self.y_energy)/self.scale).mean(axis=1))

    def fit(self,index,seed,epochs,patience,label):
        key=self.key(index,seed,epochs,patience)
        if key in self.cache:return self.cache[key],True
        latent_dim,beta=SPACE[index]
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if self.device.type=="cuda":torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model=self.model(latent_dim)
        initial_hasher=__import__("hashlib").sha256()
        for name,value in model.state_dict().items():
            initial_hasher.update(name.encode());initial_hasher.update(value.detach().cpu().numpy().tobytes())
        initial_digest=initial_hasher.hexdigest()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,foreach=False)
        rng=np.random.default_rng(seed+1000)
        best=float("inf");progress_best=float("inf");best_epoch=0;best_state=None;stale=0;history=[];began=time.perf_counter()
        for epoch in range(1,epochs+1):
            model.train();order=rng.permutation(len(self.train["history"]));totals=np.zeros(3)
            for start in range(0,len(order),128):
                ix=torch.as_tensor(order[start:start+128],device=self.device)
                batch={k:v[ix] for k,v in self.train.items()}
                optimizer.zero_grad(set_to_none=True)
                output=model(batch["history"],batch["future"],target=batch["target_norm"],sample=True)
                loss,mse,kl=normalized_loss(output,batch["target_norm"],latent_dim,beta)
                if not torch.isfinite(loss):raise RuntimeError(f"Nonfinite loss for {index}, {seed}")
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
                totals+=np.array([v.detach().item() for v in (loss,mse,kl)])*len(ix)
            score=self.score(model)
            history.append({"epoch":epoch,"training_loss":totals[0]/len(order),"power_mse":totals[1]/len(order),
                            "kl_per_latent_dimension":totals[2]/len(order),"selection_score":score})
            if score<best:
                best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if score<progress_best-1e-4:progress_best=score;stale=0
            else:stale+=1
            if patience and stale>=patience:break
        folder=self.output/"fits"/f"fit_{len(self.all_fits):03d}";folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/"model.pt"
        torch.save({"model_state_dict":best_state,"model":"rssm","latent_dim":latent_dim,"beta":beta,
                    "kl_divided_by_latent_dim":True,"config_id":index,"seed":seed,"best_epoch":best_epoch,
                    "initial_state_sha256":initial_digest,"selection_score":best,"hidden_dim":64,
                    "history_dim":self.train["history"].shape[-1],"future_dim":self.train["future"].shape[-1],
                    "preprocessing_state":self.meta["preprocessing_state"]},checkpoint)
        with (folder/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        result={"fit_id":len(self.all_fits),"config_id":index,"latent_dim":latent_dim,"beta":beta,"label":label,
                "seed":seed,"epoch_budget":epochs,"patience":patience,"epochs_ran":len(history),"best_epoch":best_epoch,
                "selection_score":best,"parameter_count":sum(p.numel() for p in model.parameters()),
                "initial_state_sha256":initial_digest,"checkpoint":checkpoint.relative_to(self.output).as_posix(),"elapsed_seconds":time.perf_counter()-began}
        self.all_fits.append(result);self.cache[key]=result
        (self.output/"fits.json").write_text(json.dumps(self.all_fits,indent=2)+"\n",encoding="utf-8")
        print(f"{label} id={index:02d} z={latent_dim:2d} beta={beta:g} seed={seed} epochs={len(history)} best_J={best:.5f} seconds={result['elapsed_seconds']:.1f}",flush=True)
        return result,False

    def predictions(self,result,split="val"):
        checkpoint=torch.load(self.output/result["checkpoint"],map_location="cpu",weights_only=True)
        model=self.model(checkpoint["latent_dim"]);model.load_state_dict(checkpoint["model_state_dict"])
        return self.forecast(model,self.validation if split=="val" else self.train)
