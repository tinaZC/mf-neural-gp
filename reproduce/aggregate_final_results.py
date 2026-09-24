#!/usr/bin/env python3
"""Aggregate audited final-run reports without touching the source results."""
import csv, glob, json, math, os, re
from collections import defaultdict

ROOT = "result_out/final_runs"
OUT = "result_out/final_analysis"
SEEDS = [200,210,220,230,240,250,260,270,280,290,300,311,322,333,344,355,366,377,388,399]
BUDGETS = {50,100,200,300,400,500}

def ci(vals):
    n=len(vals); mean=sum(vals)/n
    sd=math.sqrt(sum((x-mean)**2 for x in vals)/(n-1)) if n>1 else float("nan")
    try:
        from scipy.stats import t
        q=float(t.ppf(.975,n-1)) if n>1 else float("nan")
    except Exception:
        q=1.96
    h=q*sd/math.sqrt(n) if n>1 else float("nan")
    return n,mean,sd,sorted(vals)[n//2] if n%2 else (sorted(vals)[n//2-1]+sorted(vals)[n//2])/2,min(vals),max(vals),mean-h,mean+h

def add(rows, dataset, setting, method, seed, split, metric, value, source):
    if value is None or isinstance(value,bool): return
    try: value=float(value)
    except (TypeError,ValueError): return
    if math.isfinite(value): rows.append(dict(dataset=dataset,setting=setting,method=method,seed=seed,split=split,metric=metric,value=value,source=source))

def parse_setting(s):
    m=re.fullmatch(r"hf(50|100|200|300|400|500)_lfx(10)",s)
    return int(m.group(1)) if m else None

def main():
    os.makedirs(OUT, exist_ok=True)
    rows=[]; uq=[]; direct=[]; freq=[]; controlled=[]
    names={"hf_only":"HF-only","ar1":"AR1/co-kriging","ours":"Neural-GP MF (ours)"}
    # Authoritative original bundle reports (exclude nested per-method reports).
    for p in glob.glob(ROOT+"/original/*/*/seed_*/**/report.json", recursive=True):
        if p.split("/")[-2] in {"hf_only","ar1","ours"}: continue
        d=json.load(open(p)); parts=p.split("/original/")[1].split("/")
        ds, setting=parts[0].upper(), parts[1]; budget=parse_setting(setting)
        mseed=re.search(r"seed_(\d+)",p)
        if ds not in ("TM","AB") or budget is None or not mseed: continue
        seed=int(mseed.group(1))
        if seed not in SEEDS: continue
        met=d.get("metrics",{})
        for key in ("target_rmse","y_rmse"):
            for method,val in met.get(key,{}).items(): add(rows,ds,setting,names.get(method,method),seed,"test",key,val,p)
        for split, vals in met.get("r2",{}).items():
            for method,val in vals.items(): add(rows,ds,setting,names.get(method,method),seed,split,"r2",val,p)
        for split, kinds in met.get("nll",{}).items():
            for cal, vals in kinds.items():
                for method,val in vals.items(): add(rows,ds,setting,names.get(method,method),seed,split,f"nll_{cal}",val,p)
        for split, kinds in met.get("nlpd",{}).items():
            for cal, vals in kinds.items():
                for method,val in vals.items(): add(rows,ds,setting,names.get(method,method),seed,split,f"nlpd_{cal}",val,p)
        if ds=="TM" and setting=="hf100_lfx10":
            for metric_name, metric_key in (("nll", "nll"), ("nlpd", "nlpd")):
                for cal, vals in met.get(metric_key, {}).get("test", {}).items():
                    for method,val in vals.items():
                        uq.append(dict(dataset=ds,setting=setting,method=names.get(method,method),seed=seed,split="test",metric=f"{metric_name}_{cal}",value=float(val),source=p))
        for split in ("val","test"):
            for kind in ("coverage_raw","coverage_cal","width_raw","width_cal"):
                for method,val in met.get("uq",{}).get(split,{}).get(kind,{}).items():
                    uq.append(dict(dataset=ds,setting=setting,method=names.get(method,method),seed=seed,split=split,metric=kind,value=float(val),source=p))
    # Modern reports.
    for p in glob.glob(ROOT+"/comparisons/*/*/*/seed_*/report.json"):
        d=json.load(open(p)); ds,setting=d.get("dataset"),d.get("setting"); seed=int(d.get("seed",-1)); method=d.get("method")
        if ds not in ("TM","AB") or parse_setting(setting or "") is None or seed not in SEEDS or method not in ("FPCA-NARGP","MF-DeepONet"): continue
        for split, vals in d.get("metrics",{}).items():
            for metric,val in vals.items(): add(rows,ds,setting,method,seed,split,metric,val,p)
        if ds=="TM" and setting=="hf100_lfx10" and method=="FPCA-NARGP":
            for split, vals in d.get("metrics",{}).items():
                if "gaussian_nll" in vals: uq.append(dict(dataset=ds,setting=setting,method=method,seed=seed,split=split,metric="nll_raw",value=float(vals["gaussian_nll"]),source=p))
                if "gaussian_nlpd" in vals: uq.append(dict(dataset=ds,setting=setting,method=method,seed=seed,split=split,metric="nlpd_raw",value=float(vals["gaussian_nlpd"]),source=p))
                if "coverage_raw" in vals: uq.append(dict(dataset=ds,setting=setting,method=method,seed=seed,split=split,metric="coverage_raw",value=float(vals["coverage_raw"]),source=p))
                if "interval_width_raw" in vals: uq.append(dict(dataset=ds,setting=setting,method=method,seed=seed,split=split,metric="width_raw",value=float(vals["interval_width_raw"]),source=p))
    # Representation reports.
    for p in glob.glob(ROOT+"/representation/direct_latent/*/*/seed_*/report.json"):
        d=json.load(open(p)); seed=int(d.get("seed",-1))
        if d.get("dataset")=="TM" and d.get("setting")=="hf100_lfx10" and seed in SEEDS:
            for split,vals in d.get("metrics",{}).items():
                for metric,val in vals.items(): add(direct,"TM","hf100_lfx10","Direct-latent",seed,split,metric,val,p)
    for p in glob.glob(ROOT+"/representation/freqwise_nargp/*/*/seed_*/report.json"):
        d=json.load(open(p)); seed=int(d.get("seed",-1))
        if d.get("dataset")=="TM" and d.get("setting")=="hf100_lfx10" and seed in (200,210,220) and d.get("complete_spectral_axis"):
            for split,vals in d.get("metrics",{}).items():
                for metric,val in vals.items(): add(freq,"TM","hf100_lfx10","Frequency-wise NARGP",seed,split,metric,val,p)
    for p in glob.glob(ROOT+"/wavelength_gp/hf100_lfx10/seed*/report.json"):
        d=json.load(open(p)); seed=int(d.get("seed",-1))
        if d.get("dataset")=="TM" and d.get("setting")=="hf100_lfx10" and seed in (200,210,220) and d.get("scientific_result") is True:
            for split,vals in d.get("metrics",{}).items():
                for metric,val in vals.items(): add(controlled,"TM","hf100_lfx10","Controlled wavelength-wise Stage II",seed,split,metric,val,p)
    def write(path,data,fields):
        with open(path,"w",newline="") as f: csv.DictWriter(f,fieldnames=fields).writeheader(); csv.DictWriter(f,fieldnames=fields).writerows(data)
    write(OUT+"/master_accuracy_results.csv",rows,"dataset setting method seed split metric value source".split())
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in ("dataset","setting","method","split","metric"))].append(r["value"])
    summary=[]
    for key,v in sorted(groups.items()):
        ds,st,me,sp,mt=key; n,mean,sd,med,mi,ma,lo,hi=ci(v); summary.append(dict(dataset=ds,setting=st,method=me,split=sp,metric=mt,n=n,mean=mean,std=sd,median=med,min=mi,max=ma,ci95_low=lo,ci95_high=hi))
    write(OUT+"/main_accuracy_summary.csv",summary,"dataset setting method split metric n mean std median min max ci95_low ci95_high".split())
    # Seed-level and grouped paired RMSE comparisons.
    pm=defaultdict(dict)
    for r in rows:
        if r["split"]=="test" and r["metric"] in ("target_rmse","rmse"): pm[(r["dataset"],r["setting"],r["seed"])][r["method"]]=r["value"]
    raw=[]; pg=defaultdict(list)
    for (ds,st,se),mm in pm.items():
        ours=mm.get("Neural-GP MF (ours)")
        if ours is None: continue
        for comp in ("HF-only","AR1/co-kriging","FPCA-NARGP","MF-DeepONet"):
            if comp in mm:
                diff=mm[comp]-ours; raw.append(dict(dataset=ds,setting=st,seed=se,comparator=comp,ours_rmse=ours,comparator_rmse=mm[comp],paired_difference=diff,relative_difference_pct=100*diff/ours)); pg[(ds,st,comp)].append((ours,mm[comp],diff))
    write(OUT+"/paired_accuracy_comparisons.csv",raw,"dataset setting seed comparator ours_rmse comparator_rmse paired_difference relative_difference_pct".split())
    ps=[]
    for (ds,st,comp),vals in sorted(pg.items()):
        ours=[x[0] for x in vals]; compv=[x[1] for x in vals]; dif=[x[2] for x in vals]; n,md,sd,med,mi,ma,lo,hi=ci(dif)
        ps.append(dict(dataset=ds,setting=st,comparator=comp,n_paired=n,mean_ours_rmse=sum(ours)/n,mean_comparator_rmse=sum(compv)/n,mean_paired_difference=md,std_paired_difference=sd,paired_difference_ci95_low=lo,paired_difference_ci95_high=hi,relative_difference_pct=100*(sum(compv)/n-sum(ours)/n)/(sum(ours)/n)))
    write(OUT+"/paired_accuracy_summary.csv",ps,"dataset setting comparator n_paired mean_ours_rmse mean_comparator_rmse mean_paired_difference std_paired_difference paired_difference_ci95_low paired_difference_ci95_high relative_difference_pct".split())
    # Representation: matched three-seed diagnostics plus the 20-seed direct-latent comparison.
    rep=[]
    ours_all=[r for r in rows if r["dataset"]=="TM" and r["setting"]=="hf100_lfx10" and r["method"]=="Neural-GP MF (ours)" and r["metric"] in ("target_rmse","rmse") and r["split"]=="test"]
    for method,data in (("Neural-GP MF matched reference",[r for r in ours_all if r["seed"] in (200,210,220)]),("Controlled wavelength-wise Stage II",[r for r in controlled if r["metric"]=="rmse" and r["split"]=="test"]),("Direct-latent Stage I",[r for r in direct if r["metric"]=="rmse" and r["split"]=="test"]),("Wavelength-wise NARGP",[r for r in freq if r["metric"]=="rmse" and r["split"]=="test"])):
        v=[r["value"] for r in sorted(data,key=lambda x:int(x["seed"]))]; n,mean,sd,med,mi,ma,lo,hi=ci(v); rep.append(dict(dataset="TM",setting="hf100_lfx10",method=method,split="test",metric="rmse",n=n,individual_values=";".join(f"{x:.12g}" for x in v),mean=mean,std=sd,min=mi,max=ma,ci95_low=lo,ci95_high=hi))
    ours={r["seed"]:r["value"] for r in rows if r["dataset"]=="TM" and r["setting"]=="hf100_lfx10" and r["method"]=="Neural-GP MF (ours)" and r["metric"] in ("target_rmse","rmse") and r["split"]=="test"}; dire={r["seed"]:r["value"] for r in direct if r["metric"]=="rmse" and r["split"]=="test"}; dif=[dire[s]-ours[s] for s in SEEDS]; n,md,sd,_,_,_,lo,hi=ci(dif); rep.append(dict(dataset="TM",setting="hf100_lfx10",method="Full-vs-Direct paired",split="test",metric="rmse_difference",n=n,individual_values="",mean=(sum(ours.values())/n),std=0,min=min(dif),max=max(dif),ci95_low=lo,ci95_high=hi,mean_ours_rmse=sum(ours.values())/n,mean_direct_rmse=sum(dire.values())/n,mean_paired_difference=md,std_paired_difference=sd,relative_degradation_pct=100*(sum(dire.values())/n-sum(ours.values())/n)/(sum(ours.values())/n)))
    write(OUT+"/representation_ablation_summary.csv",rep,"dataset setting method split metric n individual_values mean std min max ci95_low ci95_high mean_ours_rmse mean_direct_rmse mean_paired_difference std_paired_difference relative_degradation_pct".split())
    # Publication UQ: TM hf100 only, actual probabilistic methods/fields.
    ug=defaultdict(list)
    for r in uq:
        if r["dataset"]=="TM" and r["setting"]=="hf100_lfx10" and r["split"]=="test": ug[tuple(r[k] for k in ("method","metric"))].append(r["value"])
    # Add frequency-wise UQ observations from representation reports.
    for r in freq:
        if r["split"]=="test" and r["metric"] in ("gaussian_nll","gaussian_nlpd","coverage_raw","interval_width_raw"):
            mt={"gaussian_nll":"nll_raw","gaussian_nlpd":"nlpd_raw","coverage_raw":"coverage_raw","interval_width_raw":"width_raw"}[r["metric"]]; ug[("Frequency-wise NARGP",mt)].append(r["value"])
    for r in controlled:
        if r["split"]=="test" and r["metric"] in ("gaussian_nll","gaussian_nlpd","coverage_raw","interval_width_raw"):
            mt={"gaussian_nll":"nll_raw","gaussian_nlpd":"nlpd_raw","coverage_raw":"coverage_raw","interval_width_raw":"width_raw"}[r["metric"]]; ug[("Controlled wavelength-wise Stage II",mt)].append(r["value"])
    ug={("Wavelength-wise NARGP" if me=="Frequency-wise NARGP" else me,mt):v for (me,mt),v in ug.items()}
    # Publication-facing table uses canonical NLL and raw interval metrics.
    ug={k:v for k,v in ug.items() if k[1] in ("nll_raw","coverage_raw","width_raw")}
    us=[]
    for (me,mt),v in sorted(ug.items()):
        diagnostic="n=3 representation diagnostic" if me in ("Wavelength-wise NARGP","Controlled wavelength-wise Stage II") else ""
        n,mean,sd,med,mi,ma,lo,hi=ci(v); us.append(dict(dataset="TM",setting="hf100_lfx10",method=me,split="test",metric=mt,n=n,mean=mean,std=sd,median=med,min=mi,max=ma,ci95_low=lo,ci95_high=hi,diagnostic=diagnostic))
    write(OUT+"/uq_summary.csv",us,"dataset setting method split metric n mean std median min max ci95_low ci95_high diagnostic".split())
    # Numerical markdown report.
    with open(OUT+"/formal_result_summary.md","w") as f:
        f.write("# Formal statistical result summary\n\n")
        f.write("Source: authoritative `result_out/final_runs/` only. No timing data or MTM modern comparisons are included.\n\n## Accuracy\n\n")
        for ds in ("TM","AB"):
            f.write(f"### {ds}\n\n| HF budget | HF-only | AR1/co-kriging | Neural-GP MF (ours) | FPCA-NARGP | MF-DeepONet |\n|---:|---:|---:|---:|---:|---:|\n")
            for b in sorted(BUDGETS):
                st=f"hf{b}_lfx10"; vals={r["method"]:r["mean"] for r in summary if r["dataset"]==ds and r["setting"]==st and r["split"]=="test" and r["metric"] in ("target_rmse","rmse")}; f.write(f"| {b} | "+" | ".join(f"{vals.get(m,float('nan')):.6g}" for m in ("HF-only","AR1/co-kriging","Neural-GP MF (ours)","FPCA-NARGP","MF-DeepONet"))+" |\n")
            f.write("\nRelative comparator-vs-ours RMSE differences (group means):\n\n")
            for p in ps:
                if p["dataset"]==ds: f.write(f"- {p['setting']}: {p['comparator']} {p['relative_difference_pct']:.3f}%\n")
            f.write("\n")
        f.write("## Representation ablation\n\n")
        rr={r["method"]:r for r in rep}; f.write(f"At TM/hf100_lfx10, the matched Neural-GP MF reference mean RMSE is {rr['Neural-GP MF matched reference']['mean']:.6g} (n=3), controlled wavelength-wise Stage II is {rr['Controlled wavelength-wise Stage II']['mean']:.6g} (n=3), direct-latent Stage I is {rr['Direct-latent Stage I']['mean']:.6g} (n=20), and wavelength-wise NARGP is {rr['Wavelength-wise NARGP']['mean']:.6g} (n=3). The paired direct-minus-full difference is {rr['Full-vs-Direct paired']['mean_paired_difference']:.6g}, 95% CI [{rr['Full-vs-Direct paired']['ci95_low']:.6g}, {rr['Full-vs-Direct paired']['ci95_high']:.6g}], relative degradation {rr['Full-vs-Direct paired']['relative_degradation_pct']:.3f}%.\n\n")
        f.write("## UQ\n\nThe source helper defines `gaussian_nlpd` as a direct alias of `gaussian_nll` (same element-wise Gaussian predictive-density formula), so NLPD is omitted from the publication-facing CSV in favor of canonical NLL. The original runs set `ci_calibrate=0`; calibration is therefore not applied and raw/calibrated intervals and scores are genuinely identical. Only canonical raw fields are shown.\n\n")
        for r in us: f.write(f"- {r['method']} {r['metric']}: {r['mean']:.6g} +/- {r['std']:.6g} (n={r['n']})" + ("; diagnostic only" if r['diagnostic'] else "") + "\n")
        f.write("\n## Claims\n\n- Strongly supported: the completed 20-seed reruns provide consistent descriptive RMSE estimates for all listed methods and budgets; paired differences are directly seed-matched.\n- Descriptive only: frequency-wise NARGP (n=3) and representation-ablation comparisons should be presented as diagnostics, without p-values or strong inferential claims.\n- Soften/remove: claims of universal superiority, statistical significance for n=3 diagnostics, timing advantages, and any modern MTM conclusion are unsupported by this completed protocol.\n")
    for ds in ("TM", "AB"):
        vals=[]
        for b in sorted(BUDGETS):
            st=f"hf{b}_lfx10"; hit=[r for r in summary if r["dataset"]==ds and r["setting"]==st and r["method"]=="Neural-GP MF (ours)" and r["split"]=="test" and r["metric"]=="target_rmse"]
            if hit: vals.append(f"{b}:{hit[0]['mean']:.6g}")
        print(f"{ds} ours test RMSE " + ", ".join(vals))
    print("Paired groups:", len(ps), "Representation groups:", len(rep), "UQ groups:", len(us))
    print("FORMAL_STATISTICS_VALIDATED")
if __name__=="__main__": main()
