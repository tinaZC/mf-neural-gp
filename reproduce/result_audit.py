import csv, json, math
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1] / "result_out"
SCHED = ROOT / "_scheduler"
MANIFEST = json.loads((SCHED / "task_manifest.json").read_text())
SUMMARY = json.loads((SCHED / "scheduler_summary.json").read_text())
issues = []
inventory = []

def finite_tree(x, path=""):
    bad = []
    if isinstance(x, dict):
        for k, v in x.items(): bad += finite_tree(v, f"{path}.{k}" if path else k)
    elif isinstance(x, list):
        for i, v in enumerate(x): bad += finite_tree(v, f"{path}[{i}]")
    elif isinstance(x, float) and not math.isfinite(x): bad.append(path)
    return bad

def report_paths(out, kind):
    if kind == "original_bundle": return sorted(out.glob("*/report.json"))
    return [out / "report.json"] if (out / "report.json").exists() else []

for t in MANIFEST:
    out = Path(t["out_dir"]); errs = []
    if t.get("status") not in {"SUCCEEDED", "COMPLETE"}: errs.append(f"scheduler_status={t.get('status')}")
    rp = report_paths(out, t["kind"])
    if t["kind"] == "freqwise_chunk":
        s, e = t["freq_start"], t["freq_end"]
        missing = [f"wavelength_{i:04d}.npz" for i in range(s,e) if not (out/"wavelength_checkpoints"/f"wavelength_{i:04d}.npz").exists()]
        if missing: errs.append("missing_chunk_checkpoints="+','.join(missing))
        rp = []
    elif not rp: errs.append("missing_report.json")
    for p in rp:
        try:
            bad = finite_tree(json.loads(p.read_text()))
            if bad: errs.append(f"nonfinite_metrics:{p.name}:{','.join(bad[:5])}")
        except Exception as ex: errs.append(f"unparseable_report:{p.name}:{type(ex).__name__}")
    if t["kind"] == "direct_latent" and not (out/"stage2_results.csv").exists(): errs.append("missing_stage2_results.csv")
    if t["kind"] == "freqwise_aggregate" and not (out/"progress.json").exists(): errs.append("missing_progress.json")
    inventory.append({"job_id":t["job_id"],"kind":t["kind"],"dataset":t.get("dataset"),"setting":t.get("setting"),"method":t.get("method"),"seed":t.get("seed"),"status":t.get("status"),"report_paths":";".join(str(p) for p in rp),"valid_reports":not errs,"issues":";".join(errs)})
    issues += [(t["job_id"], e) for e in errs]

def expect(label, pred, n):
    got=sum(1 for t in MANIFEST if pred(t))
    if got != n: issues.append((label,f"count={got}, expected={n}"))
expect("original TM drivers",lambda t:t["kind"]=="original_bundle" and t["dataset"]=="TM",360)
expect("original AB drivers",lambda t:t["kind"]=="original_bundle" and t["dataset"]=="AB",360)
expect("original MTM drivers",lambda t:t["kind"]=="original_mtm_bundle",20)
expect("modern TM",lambda t:t["kind"]=="comparison" and t["dataset"]=="TM",240)
expect("modern AB",lambda t:t["kind"]=="comparison" and t["dataset"]=="AB",240)
expect("direct latent",lambda t:t["kind"]=="direct_latent",20)
expect("freq chunks",lambda t:t["kind"]=="freqwise_chunk",60)
expect("freq aggregates",lambda t:t["kind"]=="freqwise_aggregate",3)

for t in MANIFEST:
    if t["kind"]=="comparison" and (t.get("method") not in {"FPCA-NARGP","MF-DeepONet"} or t.get("dataset") not in {"TM","AB"}): issues.append((t["job_id"],"modern_scope_mismatch"))
    if t["kind"]=="direct_latent" and not (t.get("dataset")=="TM" and t.get("setting")=="hf100_lfx10"): issues.append((t["job_id"],"direct_protocol_mismatch"))
    if t["kind"].startswith("freqwise") and not (t.get("dataset")=="TM" and t.get("setting")=="hf100_lfx10" and t.get("seed") in {200,210,220}): issues.append((t["job_id"],"freq_protocol_mismatch"))

for seed in (200,210,220):
    chunks=[t for t in MANIFEST if t["kind"]=="freqwise_chunk" and t.get("seed")==seed]; spans=sorted((t["freq_start"],t["freq_end"]) for t in chunks)
    if len(chunks)!=20: issues.append((f"freq_seed_{seed}",f"chunks={len(chunks)}, expected=20"))
    if any(e-s!=25 for s,e in spans): issues.append((f"freq_seed_{seed}","chunk_size_not_25"))
    if [i for s,e in spans for i in range(s,e)] != list(range(500)): issues.append((f"freq_seed_{seed}","coverage_not_0_500"))
    agg=next((t for t in MANIFEST if t["kind"]=="freqwise_aggregate" and t.get("seed")==seed),None)
    if agg:
        try:
            d=json.loads((Path(agg["out_dir"])/"report.json").read_text())
            if d.get("wavelength_count")!=500 or d.get("complete_spectral_axis") is not True: issues.append((f"freq_seed_{seed}","aggregate_metadata_mismatch"))
        except Exception: pass

for method in ("FPCA-NARGP","MF-DeepONet"):
    for ds in ("TM","AB"):
        for hf in (50,100,200,300,400,500):
            n=sum(1 for t in MANIFEST if t["kind"]=="comparison" and t.get("method")==method and t.get("dataset")==ds and t.get("setting","").startswith(f"hf{hf}_"))
            if n!=20: issues.append((f"{method}_{ds}_hf{hf}",f"seeds={n}, expected=20"))

bad=[t for t in MANIFEST if t.get("status") not in {"SUCCEEDED","COMPLETE"}]
if bad: issues.append(("scheduler",f"forbidden_statuses={Counter(t.get('status') for t in bad)}"))
if SUMMARY.get("manifest_task_count")!=1303: issues.append(("scheduler","manifest_task_count_mismatch"))

fields=["job_id","kind","dataset","setting","method","seed","status","report_paths","valid_reports","issues"]
with (ROOT/"final_analysis"/"result_inventory.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(inventory)
with (ROOT/"final_analysis"/"missing_or_invalid_results.csv").open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["job_id","issue"]); w.writerows(issues)
outcome="RESULT_AUDIT_READY" if not issues else "RESULT_AUDIT_NOT_READY"
(ROOT/"final_analysis"/"result_completeness_audit.md").write_text("\n".join(["# Phase-1 Result Completeness Audit","",f"- Scheduler manifest tasks: **{len(MANIFEST)}** (summary: {SUMMARY.get('manifest_task_count')})",f"- Scheduler statuses: `{SUMMARY.get('statuses')}`",f"- Total audit issues: **{len(issues)}**","","All checks are read-only and use `result_out/final_runs/` as authoritative.","",f"**{outcome}**","","See `result_inventory.csv` and `missing_or_invalid_results.csv` for task-level detail."])+'\n')
print(outcome, "issues", len(issues))
for x in issues[:50]: print(x)
