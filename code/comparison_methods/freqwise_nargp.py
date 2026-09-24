from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from mf_train_baseline.mf_utils import rmse, r2_score, gaussian_nll, gaussian_nlpd, ci_coverage_y, ci_width_y, set_seed
from .common import load_bundle
from .nargp_core import train_nargp, predict_nargp

def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    try: tmp.write_text(json.dumps(obj, indent=2)); os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()
def atomic_npz(path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        with tmp.open('wb') as f: np.savez_compressed(f, **kw)
        os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()
def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)
def indices(args, n):
    if args.freq_indices:
        out=[int(v.strip()) for v in args.freq_indices.split(',') if v.strip()]
        if not out or len(set(out))!=len(out) or any(v<0 or v>=n for v in out): raise ValueError('invalid --freq_indices')
        return out
    a,b=args.freq_start,(n if args.freq_end is None else args.freq_end)
    if a<0 or b>n or a>=b: raise ValueError(f'invalid range [{a},{b})')
    return list(range(a,b))
def _hash_array(h, name, value):
    array=np.ascontiguousarray(value); h.update(name.encode()); h.update(str(array.dtype).encode()); h.update(np.asarray(array.shape,dtype=np.int64).tobytes()); h.update(array.tobytes())
def signature(args,d):
    h=hashlib.sha256(); _hash_array(h,'wavelengths',d.wavelengths); _hash_array(h,'idx_wavelength',d.idx_wavelength)
    for block_name,block in (('hf',d.hf),('lf_paired',d.lf_paired),('lf_unpaired',d.lf_unpaired)):
        for s in ('train','val','test'):
            for q in ('x','y','t','idx'): _hash_array(h,f'{block_name}/{s}/{q}',block[s][q])
    identity={'data_root':str(d.root.resolve()),'data_hash':h.hexdigest(),'seed':args.seed,'steps':args.steps,'inducing':args.inducing,'lr':args.lr,'kernel':args.kernel,'mc_samples':args.mc_samples}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
def cp(out,k): return out/'wavelength_checkpoints'/f'wavelength_{k:04d}.npz'
def valid_checkpoints(out,sig,n):
    found=[]; seen=set()
    for p in sorted((out/'wavelength_checkpoints').glob('wavelength_*.npz')) if (out/'wavelength_checkpoints').exists() else []:
        k=int(p.stem.split('_')[-1])
        if k in seen or k<0 or k>=n: raise ValueError(f'duplicate/out-of-range checkpoint {p}')
        with np.load(p,allow_pickle=False) as z:
            if str(z['signature'].item())!=sig or int(z['k'].item())!=k: raise ValueError(f'checkpoint signature/index mismatch: {p}')
        seen.add(k); found.append(k)
    return found
def chunk_identifier(requested):
    encoded=np.asarray(requested,dtype=np.int64).tobytes(); digest=hashlib.sha256(encoded).hexdigest()[:12]
    return f'{min(requested):04d}_{max(requested):04d}_{len(requested):04d}_{digest}'
def progress(out,args,d,sig,requested):
    done=valid_checkpoints(out,sig,len(d.wavelengths)); done_set=set(done); completed=[k for k in requested if k in done_set]
    atomic_json(out/'progress'/f'chunk_{chunk_identifier(requested)}.json',{'method':'Frequency-wise NARGP','run_kind':args.run_kind,'run_signature':sig,'requested_indices':requested,'completed_requested_indices':completed,'completed_requested_count':len(completed),'completed_global_count':len(done),'total_wavelengths':len(d.wavelengths)})
def canonical_progress(out,args,d,sig,requested):
    done=valid_checkpoints(out,sig,len(d.wavelengths)); done_set=set(done); completed=[k for k in requested if k in done_set]
    atomic_json(out/'progress.json',{'method':'Frequency-wise NARGP','run_kind':args.run_kind,'run_signature':sig,'aggregated_indices':requested,'completed_aggregated_indices':completed,'completed_aggregated_count':len(completed),'completed_global_indices':done,'completed_global_count':len(done),'total_wavelengths':len(d.wavelengths)})
def metrics(y,m,v,args):
    s=np.sqrt(np.maximum(v,0)); return {'rmse':rmse(y,m),'r2':r2_score(y,m),'gaussian_nll':gaussian_nll(y,m,var=v),'gaussian_nlpd':gaussian_nlpd(y,m,var=v),'coverage_raw':ci_coverage_y(y,m,s,args.ci_level),'interval_width_raw':ci_width_y(s,args.ci_level)}
def train_chunk(args):
    set_seed(args.seed); d=load_bundle(args.data_dir); dev=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); out=Path(args.out_dir).resolve(); req=indices(args,len(d.wavelengths)); sig=signature(args,d); done=set(valid_checkpoints(out,sig,len(d.wavelengths)))
    if not args.resume and done.intersection(req): raise FileExistsError('completed requested wavelengths exist; pass --resume')
    xlf=np.concatenate([d.lf_paired['train']['x'],d.lf_unpaired['train']['x']]).astype(np.float32); xhf=d.hf['train']['x'].astype(np.float32); wall=time.perf_counter()
    for k in req:
        if k in done: continue
        raw=np.concatenate([d.lf_paired['train']['y'][:,k],d.lf_unpaired['train']['y'][:,k]]).astype(np.float32)[:,None]; sc=StandardScaler().fit(raw)
        ylf=sc.transform(raw).astype(np.float32); ylp=sc.transform(d.lf_paired['train']['y'][:,k:k+1]).astype(np.float32); yhf=sc.transform(d.hf['train']['y'][:,k:k+1]).astype(np.float32)
        model=train_nargp(xlf,ylf,xhf,ylp,yhf,device=dev,inducing=args.inducing,steps=args.steps,lr=args.lr,kernel_name=args.kernel,seed=args.seed); sync(dev); t=time.perf_counter()
        vm,vv=predict_nargp(model,d.hf['val']['x'],args.mc_samples,dev,args.seed+100000+k); tm,tv=predict_nargp(model,d.hf['test']['x'],args.mc_samples,dev,args.seed+200000+k); sync(dev); inf=time.perf_counter()-t
        scale=float(sc.scale_[0]); mean=float(sc.mean_[0]); vm=vm*scale+mean; tm=tm*scale+mean; vv=vv*scale**2; tv=tv*scale**2
        if not all(np.isfinite(a).all() for a in (vm,vv,tm,tv)) or np.any(vv<0) or np.any(tv<0): raise FloatingPointError(f'invalid predictions at {k}')
        atomic_npz(cp(out,k),signature=np.asarray(sig),k=np.asarray(k),wavelength=np.asarray(d.wavelengths[k]),idx_wavelength=np.asarray(d.idx_wavelength[k]),val_mean=vm[:,0],val_variance=vv[:,0],test_mean=tm[:,0],test_variance=tv[:,0],stage1=np.asarray(model.stage1_train_s),stage2=np.asarray(model.stage2_train_s),inference=np.asarray(inf),target_mean=np.asarray(mean),target_scale=np.asarray(scale)); done.add(k); progress(out,args,d,sig,req); print(f'completed {k}',flush=True)
    progress(out,args,d,sig,req); print(json.dumps({'requested_indices':req,'completed':len(req),'wall_s':time.perf_counter()-wall},indent=2))
def aggregate(args):
    d=load_bundle(args.data_dir); out=Path(args.out_dir).resolve(); req=sorted(indices(args,len(d.wavelengths))); sig=signature(args,d); done=valid_checkpoints(out,sig,len(d.wavelengths));
    if set(done)!=set(done): raise AssertionError
    missing=[k for k in req if k not in done]
    if missing or len(done)!=len(set(done)): raise ValueError(f'missing requested checkpoints: {missing}')
    rec=[]
    for k in req:
        with np.load(cp(out,k),allow_pickle=False) as z:
            if float(z['wavelength'])!=float(d.wavelengths[k]) or int(z['idx_wavelength'])!=int(d.idx_wavelength[k]): raise ValueError(f'axis mismatch at {k}')
            rec.append({q:z[q].copy() for q in z.files})
    vm=np.column_stack([r['val_mean'] for r in rec]); vv=np.column_stack([r['val_variance'] for r in rec]); tm=np.column_stack([r['test_mean'] for r in rec]); tv=np.column_stack([r['test_variance'] for r in rec]); pred=out/'pred_arrays'; pred.mkdir(parents=True,exist_ok=True)
    for s,m,v in (('val',vm,vv),('test',tm,tv)): np.save(pred/f'{s}_mean.npy',m.astype(np.float32)); np.save(pred/f'{s}_variance.npy',v.astype(np.float32)); np.save(pred/f'{s}_indices.npy',d.hf[s]['idx'])
    np.save(pred/'wavelengths.npy',d.wavelengths[req]); np.save(pred/'idx_wavelength.npy',d.idx_wavelength[req]); yv=d.hf['val']['y'][:,req]; yt=d.hf['test']['y'][:,req]; full=req==list(range(len(d.wavelengths))); timing={'method':'Frequency-wise NARGP','dataset':args.dataset,'setting':d.root.name,'seed':args.seed,'device':args.device,'stage1_train_s':float(sum(r['stage1'] for r in rec)),'stage2_train_s':float(sum(r['stage2'] for r in rec)),'inference_val_test_s':float(sum(r['inference'] for r in rec)),'completed_wavelengths':len(req),'total_model_wall_s':float(sum(r['stage1']+r['stage2']+r['inference'] for r in rec)),'excludes_em_simulation_time':True}
    report={'method':'Frequency-wise NARGP','run_kind':args.run_kind,'scientific_result':bool(args.run_kind=='scientific' and full),'complete_spectral_axis':bool(full),'preliminary_subset_diagnostic':bool(args.run_kind=='preliminary_subset' or not full),'dataset':args.dataset,'setting':d.root.name,'seed':args.seed,'wavelength_count':len(req),'total_wavelength_count':len(d.wavelengths),'requested_indices':req,'nargp':{'implementation':'shared nargp_core train_nargp/predict_nargp','kernel':'k_rho(z,z\') * k_x(x,x\') + k_delta(x,x\')','inducing':args.inducing,'steps':args.steps,'lr':args.lr,'mc_samples':args.mc_samples,'ard':True,'likelihood':'GaussianLikelihood','likelihood_noise_init':'0.01 * Var(training target)','x_scaler_fit':'combined LF train only per wavelength','target_scaler_fit':'LF train only per wavelength','uncertainty_propagation':'MC LF posterior; E[var]+Var[mean]'},'metrics':{'val':metrics(yv,vm,vv,args),'test':metrics(yt,tm,tv,args)},'timing_file':'timing.json','prediction_dir':'pred_arrays'}; atomic_json(out/'timing.json',timing); atomic_json(out/'report.json',report); canonical_progress(out,args,d,sig,req); print(json.dumps(report,indent=2))
def parse_args():
    p=argparse.ArgumentParser(description='Independent wavelength-wise NARGP diagnostic'); p.add_argument('--data_dir',required=True); p.add_argument('--out_dir',required=True); p.add_argument('--seed',type=int,default=200); p.add_argument('--dataset',choices=('TM','AB','MTM'),default='TM'); p.add_argument('--device',choices=('cpu','cuda'),default='cpu'); p.add_argument('--freq_start',type=int,default=0); p.add_argument('--freq_end',type=int); p.add_argument('--freq_indices',default=''); p.add_argument('--resume',action='store_true'); p.add_argument('--checkpoint_every',type=int,default=1); p.add_argument('--aggregate',action='store_true'); p.add_argument('--inducing',type=int,default=64); p.add_argument('--steps',type=int,default=2000); p.add_argument('--lr',type=float,default=0.005); p.add_argument('--kernel',choices=('rbf','matern'),default='rbf'); p.add_argument('--mc_samples',type=int,default=512); p.add_argument('--ci_level',type=float,default=0.95); p.add_argument('--run_kind',choices=('smoke','preliminary_subset','scientific'),default='scientific'); return p.parse_args()
if __name__=='__main__':
    a=parse_args(); aggregate(a) if a.aggregate else train_chunk(a)
