#!/usr/bin/env python3
"""Serial, resumable isolated timing protocol."""
import argparse,csv,datetime as dt,json,os,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; CODE=ROOT/'code'; OUT=ROOT/'result_out/final_timing'; DATA=ROOT/'data/mf_sweep_datasets_nano_tm'
JOBS=[(m,f'hf{h}_lfx10') for m in ('fpca_nargp','mf_deeponet') for h in (50,100,500)]+[('direct_latent','hf100_lfx10'),('freqwise_nargp','hf100_lfx10')]
FIELDS=['job_id','method','setting','seed','status','start_utc','end_utc','wall_clock_seconds','returncode','command','stage_timing_json','device','environment_json']
def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def gpu_check():
 try:
  import torch
  if not torch.cuda.is_available(): return False,'CUDA unavailable'
  if torch.cuda.device_count()<1: return False,'GPU 0 unavailable'
  return True,torch.cuda.get_device_name(0)
 except Exception as e: return False,str(e)
def warn_active():
 try:
  o=subprocess.check_output(['ps','-eo','pid,args'],text=True); h=[x.strip() for x in o.splitlines() if any(k in x.lower() for k in ('train','nargp','deeponet')) and 'run_final_timing.py' not in x]
  if h: print('WARNING: possible active project training process:',h[:3])
 except Exception: pass
def base_cmd(method,setting,out):
 mod={'fpca_nargp':'comparison_methods.fpca_nargp','mf_deeponet':'comparison_methods.mf_deeponet','direct_latent':'comparison_methods.direct_latent','freqwise_nargp':'comparison_methods.freqwise_nargp'}[method]
 c=[sys.executable,'-m',mod,'--dataset','TM','--data_dir',str(DATA/setting),'--out_dir',str(out),'--seed','42','--device','cuda','--run_kind','scientific']
 if method=='fpca_nargp': c += ['--inducing','64','--steps','2000','--lr','0.005','--fpca_var_ratio','0.999','--fpca_max_dim','64','--fpca_dim','0','--fpca_ridge','1e-8','--mc_samples','1000','--kernel','rbf','--wl_low','380.0','--wl_high','750.0','--ci_level','0.95']
 elif method=='mf_deeponet': c += ['--epochs','200','--patience','25','--batch_size','4096','--lr','0.001','--weight_decay','0.0','--branch_width','128','--branch_depth','3','--trunk_width','128','--trunk_depth','3','--latent_width','64']
 elif method=='direct_latent': c += ['--fpca_var_ratio','0.999','--fpca_max_dim','64','--fpca_dim','0','--fpca_ridge','1e-8','--hidden','256','256','256','--feat_dim','32','--act','relu','--feat_act','leakyrelu','--feat_leaky_slope','0.01','--dropout','0.0','--student_lr','0.0003','--student_bs','256','--student_wd','0.0001','--student_epochs','2000','--student_patience','100','--student_min_delta','0.0001','--svgp_M','64','--svgp_steps','2000','--svgp_lr','0.005','--kernel','matern','--matern_nu','2.5','--rho_ridge','1e-6']
 return c
def commands(method,setting,out):
 c=base_cmd(method,setting,out)
 return [c+['--freq_start','0','--freq_end','500'],c+['--aggregate']] if method=='freqwise_nargp' else [c]
def read_records(p):
 d={}
 if p.exists():
  with p.open() as f:
   for r in csv.DictReader(f):
    if r.get('job_id'): d[r['job_id']]=r
 return d
def write_records(p,d):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(d[k] for k in sorted(d))
def main():
 ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(required=True); g.add_argument('--dry-run',action='store_true'); g.add_argument('--run',action='store_true'); ap.add_argument('--gpu',type=int,default=0); a=ap.parse_args()
 if a.gpu!=0: print('Frozen timing protocol requires --gpu 0',file=sys.stderr); return 2
 ok,device=gpu_check(); print(f'GPU check: {"OK" if ok else "FAIL"} ({device})')
 if not ok: return 2
 warn_active(); OUT.mkdir(parents=True,exist_ok=True); (OUT/'logs').mkdir(exist_ok=True); rec=read_records(OUT/'timing_runs.csv'); print(f'Frozen timing jobs: {len(JOBS)}')
 for i,(method,setting) in enumerate(JOBS,1):
  jid=f'{method}_{setting}_seed42'; out=OUT/jid; cs=commands(method,setting,out); print(f'{i}/8 {jid}')
  if rec.get(jid,{}).get('status')=='success': print('  already validated; skipping'); continue
  for c in cs: print('  '+' '.join(c))
  if a.dry_run: continue
  start=utc(); t0=time.perf_counter(); rc=0; out.mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env.update(PYTHONPATH=str(CODE),CUDA_VISIBLE_DEVICES='0'); log=OUT/'logs'/(jid+'.log')
  with log.open('w') as lf:
   for c in cs:
    rc=subprocess.run(c,cwd=ROOT,env=env,stdout=lf,stderr=subprocess.STDOUT).returncode
    if rc: break
  dur=time.perf_counter()-t0; status='success' if rc==0 else 'failed'; report=out/'report.json'; stage=''
  if status=='success' and method=='freqwise_nargp':
   try:
    d=json.loads(report.read_text()); status='success' if d.get('complete_spectral_axis') and d.get('wavelength_count')==500 and d.get('total_wavelength_count')==500 else 'failed'
   except Exception: status='failed'
  if report.exists():
   try: stage=json.dumps(json.loads(report.read_text()).get('timing',''))
   except Exception: pass
  if not stage and (out/'timing.json').exists():
   try: stage=json.dumps(json.loads((out/'timing.json').read_text()))
   except Exception: pass
  rec[jid]={'job_id':jid,'method':method,'setting':setting,'seed':42,'status':status,'start_utc':start,'end_utc':utc(),'wall_clock_seconds':f'{dur:.6f}','returncode':rc,'command':' && '.join(' '.join(c) for c in cs),'stage_timing_json':stage,'device':device,'environment_json':json.dumps({'python':sys.executable,'platform':platform.platform(),'hostname':platform.node(),'gpu':0},sort_keys=True)}; write_records(OUT/'timing_runs.csv',rec)
  if status!='success': print(f'  FAILED; see {log}')
 if a.dry_run:
  (OUT/'timing_protocol.md').write_text('# Frozen isolated timing protocol\n\nTM only, seed 42, GPU 0, eight jobs, strictly serial. Frequency-wise is one timed job containing fresh 0:500 training followed by aggregation; no resume/checkpoint reuse is permitted.\n'); print('TIMING_RUNNER_READY'); return 0
 valid=[]
 for method,setting in JOBS:
  jid=f'{method}_{setting}_seed42'; r=rec.get(jid)
  if not r or r.get('status')!='success': raise SystemExit('Timing incomplete: '+jid)
  try: assert float(r['wall_clock_seconds'])>0
  except Exception: raise SystemExit('Invalid duration: '+jid)
  valid.append(r)
 with (OUT/'timing_summary.csv').open('w',newline='') as f:
  fs=['job_id','method','setting','seed','status','wall_clock_seconds','stage_timing_json','device']; w=csv.DictWriter(f,fieldnames=fs); w.writeheader(); w.writerows({k:r.get(k,'') for k in fs} for r in valid)
 print('ISOLATED_TIMING_READY'); return 0
if __name__=='__main__': raise SystemExit(main())
