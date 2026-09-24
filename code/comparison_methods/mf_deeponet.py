from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .common import load_bundle



import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class PointDataset(Dataset):
    """Lazy (sample, wavelength) point view; avoids materializing N*K repeats."""
    def __init__(self, x, y, wavelengths):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.wavelengths = np.asarray(wavelengths, dtype=np.float32)
        if self.y.shape != (len(self.x), len(self.wavelengths)):
            raise ValueError(f"point data shape mismatch: x={self.x.shape}, y={self.y.shape}, wl={self.wavelengths.shape}")
    def __len__(self): return self.x.shape[0] * self.wavelengths.size
    def __getitem__(self, j):
        i, k = divmod(int(j), self.wavelengths.size)
        return self.x[i], self.wavelengths[k:k+1], self.y[i, k:k+1]


class DeepONet(nn.Module):
    def __init__(self, branch_dim, branch_width=128, branch_depth=3,
                 trunk_width=128, trunk_depth=3, latent_width=64, trunk_dim=1):
        super().__init__()
        self.branch = self._mlp(branch_dim, branch_width, branch_depth, latent_width)
        self.trunk = self._mlp(trunk_dim, trunk_width, trunk_depth, latent_width)
        self.bias = nn.Parameter(torch.zeros(1))
    @staticmethod
    def _mlp(inp, width, depth, out):
        layers = [nn.Linear(inp, width), nn.Tanh()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out))
        return nn.Sequential(*layers)
    def forward(self, branch_input, wavelength):
        return (self.branch(branch_input) * self.trunk(wavelength)).sum(-1, keepdim=True) + self.bias


def predict_grid(model, x, wavelengths, device, batch_size=16384):
    model.eval(); out = np.empty((len(x), len(wavelengths)), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(x)):
            xb = torch.as_tensor(np.repeat(x[i:i+1], len(wavelengths), axis=0), device=device)
            wb = torch.as_tensor(wavelengths[:, None], dtype=torch.float32, device=device)
            vals = []
            for j in range(0, len(wavelengths), batch_size):
                vals.append(model(xb[j:j+batch_size], wb[j:j+batch_size]).squeeze(-1).cpu().numpy())
            out[i] = np.concatenate(vals)
    return out
def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def fit_model(model, ds, epochs, lr, batch_size, patience, weight_decay, device, val_ds=None):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, stale = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        for xb, wb, yb in loader:
            xb, wb, yb = xb.to(device), wb.to(device), yb.to(device)
            opt.zero_grad(); loss = ((model(xb, wb) - yb) ** 2).mean(); loss.backward(); opt.step()
        if val_ds is not None:
            model.eval(); total = 0.; n = 0
            vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for xb, wb, yb in vl:
                    total += float(((model(xb.to(device), wb.to(device))-yb.to(device))**2).sum()); n += yb.numel()
            score = total / max(1,n)
            if score < best:
                best, best_state, stale = score, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
            else: stale += 1
            if stale >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model

def metrics(y, p):
    e = np.asarray(p)-np.asarray(y); rmse=float(np.sqrt(np.mean(e*e)))
    den=float(np.sum((y-np.mean(y))**2)); r2=float(1-np.sum(e*e)/den) if den else float('nan')
    return {'rmse':rmse, 'r2':r2}

def main(a):
    seed_all(a.seed); device=torch.device(a.device)
    data=load_bundle(a.data_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    hf, lp, lu=data.hf, data.lf_paired, data.lf_unpaired
    # Fit every transformer on training data only. x is shared by both stages.
    x_scaler=StandardScaler().fit(np.concatenate([lp['train']['x'],lu['train']['x']],0))
    w_scaler=StandardScaler().fit(data.wavelengths[:,None])
    y_scaler=StandardScaler().fit(np.concatenate([lp['train']['y'].ravel(),lu['train']['y'].ravel()])[:,None])
    def sx(z): return x_scaler.transform(z).astype(np.float32)
    sw=data.wavelengths[:,None].astype(np.float32); sw=w_scaler.transform(sw).astype(np.float32).ravel()
    def sy(z): return y_scaler.transform(z.reshape(-1,1)).reshape(z.shape).astype(np.float32)
    lf_x=np.concatenate([sx(lp['train']['x']),sx(lu['train']['x'])]); lf_y=np.concatenate([sy(lp['train']['y']),sy(lu['train']['y'])])
    lf_val=PointDataset(sx(lp['val']['x']),sy(lp['val']['y']),sw)
    t0=time.perf_counter(); lf=DeepONet(lf_x.shape[1],a.branch_width,a.branch_depth,a.trunk_width,a.trunk_depth,a.latent_width).to(device)
    fit_model(lf,PointDataset(lf_x,lf_y,sw),a.epochs,a.lr,a.batch_size,a.patience,a.weight_decay,device,lf_val); stage1=time.perf_counter()-t0
    def lf_pred(x): return predict_grid(lf,sx(x),sw,device)
    # Match the reference protocol: paired LF truth is used for HF training; validation/test use LF predictions.
    lfh=sy(lp['train']['y']); hfy=sy(hf['train']['y']); resid=hfy-lfh
    r_scaler=StandardScaler().fit(resid.reshape(-1,1)); resid_s=r_scaler.transform(resid.reshape(-1,1)).reshape(resid.shape).astype(np.float32)
    lfv=lf_pred(hf['val']['x']); hfv=sy(hf['val']['y']); rv=hfv-lfv; rv_s=r_scaler.transform(rv.reshape(-1,1)).reshape(rv.shape).astype(np.float32)
    # Match the reference: stack the LF response with wavelength in the HF trunk.
    t0=time.perf_counter(); corr=DeepONet(hf['train']['x'].shape[1],a.branch_width,a.branch_depth,a.trunk_width,a.trunk_depth,a.latent_width,trunk_dim=2).to(device)
    # custom correction dataset keeps one LF value per sample/wavelength lazily.
    class Corr(PointDataset):
        def __init__(self,x,lf,y,w): self.x,self.lf,self.y,self.w=np.asarray(x,np.float32),np.asarray(lf,np.float32),np.asarray(y,np.float32),np.asarray(w,np.float32)
        def __len__(self): return len(self.x)*len(self.w)
        def __getitem__(self,j):
            i,k=divmod(int(j),len(self.w)); return self.x[i],np.r_[self.w[k],self.lf[i,k]],self.y[i,k:k+1]
    cds=Corr(sx(hf['train']['x']),lfh,resid_s,sw); cv=Corr(sx(hf['val']['x']),lfv,rv_s,sw)
    fit_model(corr,cds,a.epochs,a.lr,a.batch_size,a.patience,a.weight_decay,device,cv); stage2=time.perf_counter()-t0
    def corr_pred(x,lpred):
        corr.eval(); arr=np.empty_like(lpred)
        with torch.no_grad():
            for i in range(len(x)):
                xb=np.repeat(sx(x[i:i+1]),len(sw),0); wb=np.c_[sw,lpred[i]]; vals=[]
                for j in range(0,len(sw),16384): vals.append(corr(torch.as_tensor(xb[j:j+16384],device=device),torch.as_tensor(wb[j:j+16384],device=device)).squeeze(-1).cpu().numpy())
                arr[i]=np.concatenate(vals)
        return r_scaler.inverse_transform(arr.reshape(-1,1)).reshape(arr.shape)
    t0=time.perf_counter(); plv=lf_pred(hf['val']['x']); plt_=lf_pred(hf['test']['x']); pv=y_scaler.inverse_transform((plv+0).reshape(-1,1)).reshape(plv.shape)+corr_pred(hf['val']['x'],plv)*y_scaler.scale_[0] ; pt=y_scaler.inverse_transform(plt_.reshape(-1,1)).reshape(plt_.shape)+corr_pred(hf['test']['x'],plt_)*y_scaler.scale_[0]; inference=time.perf_counter()-t0
    pred=out/'pred_arrays'; pred.mkdir(exist_ok=True)
    for n,v in [('val_mean',pv),('test_mean',pt),('val_indices',hf['val']['idx']),('test_indices',hf['test']['idx']),('wavelengths',data.wavelengths),('idx_wavelength',data.idx_wavelength)]: np.save(pred/f'{n}.npy',v)
    report={'method':'MF-DeepONet','dataset':a.dataset,'setting':data.root.name,'device':str(device),'out_dir':str(out.resolve()),'run_kind':a.run_kind,'scientific_result':a.run_kind=='scientific','seed':a.seed,'architecture':vars(a),'scaling':{'x':'StandardScaler fit on LF paired+unpaired train','wavelength':'StandardScaler fit on wavelength grid','target':'StandardScaler fit on LF train targets','residual':'StandardScaler fit on HF train residuals'},'metrics':{'val':metrics(hf['val']['y'],pv),'test':metrics(hf['test']['y'],pt)},'uq':{'status':'N/A deterministic DeepONet'}}
    timing={'method':'MF-DeepONet','dataset':a.dataset,'setting':data.root.name,'seed':a.seed,'device':str(device),'out_dir':str(out.resolve()),'stage_i_lf_training_seconds':stage1,'stage_ii_hf_correction_training_seconds':stage2,'inference_seconds':inference,'total_model_seconds':stage1+stage2+inference}
    (out/'report.json').write_text(json.dumps(report,indent=2)); (out/'timing.json').write_text(json.dumps(timing,indent=2))
    print(json.dumps({'test':report['metrics']['test'],'timing':timing,'shapes':{'val':pv.shape,'test':pt.shape}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description='MF-DeepONet multi-fidelity baseline (smoke or scientific).')
    p.add_argument('--data_dir',required=True); p.add_argument('--dataset',choices=['TM','AB','MTM'],default='TM'); p.add_argument('--out_dir',required=True); p.add_argument('--seed',type=int,default=200); p.add_argument('--run_kind',choices=['smoke','scientific'],default='scientific'); p.add_argument('--device',default='cpu'); p.add_argument('--epochs',type=int,default=200); p.add_argument('--patience',type=int,default=25); p.add_argument('--batch_size',type=int,default=4096); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--weight_decay',type=float,default=0.0); p.add_argument('--branch_width',type=int,default=128); p.add_argument('--branch_depth',type=int,default=3); p.add_argument('--trunk_width',type=int,default=128); p.add_argument('--trunk_depth',type=int,default=3); p.add_argument('--latent_width',type=int,default=64)
    main(p.parse_args())
