"""Verify and package the canonical local submission; never upload it."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from datetime import datetime
import argparse,ast,csv,hashlib,json,re

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(pdf_reviewed=False):
    if not pdf_reviewed:raise RuntimeError('Render and inspect the final PDF before recording visual review')
    registry=json.loads((ROOT/'outputs/run_registry.json').read_text())
    final=json.loads((ROOT/'outputs/canonical_selection.json').read_text())
    assert final['run']==registry['reference']=='v3'
    assert json.loads((ROOT/'outputs/final_selection.json').read_text())==final
    notebook=ROOT/'notebooks/Mars_HiRISE_Final.ipynb'
    nb=json.loads(notebook.read_text(encoding='utf-8'))
    cells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert all(c.get('execution_count') is not None for c in cells)
    assert not any(o.get('output_type')=='error' for c in cells for o in c['outputs'])
    specs=[]
    for cell in cells:
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['MAIN_RUNS','ROBUSTNESS_RUNS'] for t in node.targets):
                specs+=ast.literal_eval(node.value)
    assert {s['name']:{k:v for k,v in s.items() if k!='name'} for s in specs}==registry['runs']
    test=(ROOT/'outputs/test_results_final.txt').read_text(encoding='utf-8-sig')
    match=re.search(r'(\d+) passed(?:, \d+ warnings)? in ([\d.]+)s',test)
    assert match and int(match[1])>=16
    assert f'Executed {len(cells)} code cells with no errors' in (ROOT/'outputs/final_notebook_execution.txt').read_text(encoding='utf-8-sig')
    evidence={}
    for name,cfg in registry['runs'].items():
        run=ROOT/'outputs'/name
        status=json.loads((run/'training_status.json').read_text());assert status['status']=='complete'
        with (run/'history.csv').open(newline='') as f:history=list(csv.DictReader(f))
        assert len(history)==15
        with ZipFile(run/'best.pt') as z:assert z.testzip() is None
        folder=run/registry['analysis']
        threshold=json.loads((folder/'calibration.json').read_text())['threshold']
        with (folder/'novelty_scores.csv').open(newline='') as f:scores=list(csv.DictReader(f))
        flagged=sorted([r for r in scores if float(r['novelty_score'])>threshold],key=lambda r:(-float(r['novelty_score']),r['filename']))
        with (folder/'selected_for_interpretation.csv').open(newline='') as f:selected=list(csv.DictReader(f))
        assert [r['filename'] for r in selected]==[r['filename'] for r in flagged[:5]]
        assert all((r['flagged'].lower()=='true')==(float(r['novelty_score'])>threshold) for r in scores)
        assert json.loads((folder/'diagnostics.json').read_text())['flagged_total']==len(flagged)
        assert all((folder/f"heatmap_{Path(r['filename']).stem}.png").exists() for r in selected)
        evidence[name]={'epochs':len(history),'flags':len(flagged),'best_checkpoint_sha256':sha(run/'best.pt'),'config_sha256':sha(run/'config.json')}
    report=ROOT/'output/pdf/Mars_HiRISE_Final_Report.pdf'
    pdfcheck=json.loads((ROOT/'outputs/final_pdf_review.json').read_text())
    assert pdfcheck['sha256']==sha(report) and pdfcheck['all_pages_visually_reviewed']
    result={'recorded_at':datetime.now().astimezone().isoformat(timespec='seconds'),'reference':'v3','tests_passed':int(match[1]),'test_seconds':float(match[2]),'notebook_code_cells':len(cells),'notebook_errors':0,'notebook_sha256':sha(notebook),'pdf_sha256':sha(report),'pdf_pages':pdfcheck['pages'],'pdf_visual_review_passed':True,'runs':evidence,'cloud_execution_verified':False,'fresh_training_check':'Two-epoch, one-batch CPU smoke with legacy resume; full saved experiments reused during notebook execution. Full cloud retraining not claimed.','external_submission_performed':False}
    (ROOT/'outputs/final_verification.json').write_text(json.dumps(result,indent=2))
    return registry

def package(registry):
    files=set()
    files.update(ROOT.glob('*.docx'))
    for folder in ['mars_anomaly','tests']:
        files.update((ROOT/folder).glob('*.py'))
    for name in ['build_final_notebook.py','build_final_report.py','package_final_submission.py','execute_notebook.py','summarize_experiments.py','project_latents.py','measure_photometric_sensitivity.py','metadata_fusion.py','evaluate_border_experiment.py','audit_inputs.py','inspect_selected_context.py','update_progress.py','forest_sensitivity.py']:
        files.add(ROOT/'scripts'/name)
    for name in ['README.md','ENGINEERING_CHANGELOG.md','DECISION_LOG.md','CURRENT_PROGRESS.md','DEFENSE_GUIDE.md','RESULTS.md','SUBMISSION_CHECKLIST.md','RESEARCH_AND_APPROACH.md','THRESHOLD_DECISION.md','IMPROVEMENT_PROTOCOL.md','BORDER_EXPERIMENT_PROTOCOL.md','V7_GRADIENT_EXPERIMENT_PROTOCOL.md','REPAIR_SUMMARY.md','requirements.txt','requirements-lock-local.txt']:
        files.add(ROOT/name)
    for name in ['run_registry.json','canonical_selection.json','final_selection.json','geological_hypotheses.json','final_verification.json','final_pdf_review.json','test_results_final.txt','final_notebook_execution.txt']:
        files.add(ROOT/'outputs'/name)
    files.update([ROOT/'notebooks/Mars_HiRISE_Final.ipynb',ROOT/'output/pdf/Mars_HiRISE_Final_Report.pdf'])
    for folder in ['outputs/comparison','outputs/border_experiment']:
        files.update(p for p in (ROOT/folder).glob('*') if p.is_file())
    for name in ['audit.json','image_statistics.csv','random_batch.png']:
        files.add(ROOT/'outputs/audit'/name)
    for name in registry['runs']:
        run=ROOT/'outputs'/name
        files.add(run/'best.pt')
        files.update(p for p in run.glob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.png'])
        files.update(p for p in (run/registry['analysis']).glob('*') if p.is_file())
    files.update(p for p in (ROOT/'outputs/v8/fusion_controlled').glob('*') if p.is_file())
    files=sorted(files)
    assert all(p.is_file() for p in files)
    records=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    destination=ROOT/'output/submission';destination.mkdir(exist_ok=True)
    manifest=destination/'final_manifest.json';manifest.write_text(json.dumps(records,indent=2))
    target=destination/'NSSC_2026_Mars_Final.zip'
    with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.write(manifest,'submission_manifest.json')
    with ZipFile(target) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'] for r in records)
        for name in registry['runs']:
            assert f'outputs/{name}/best.pt' in z.namelist()
            assert f"outputs/{name}/{registry['analysis']}/novelty_scores.csv" in z.namelist()
        assert 'notebooks/Mars_HiRISE_Final.ipynb' in z.namelist()
        assert not any('history_before_repair' in n or n.startswith('data/') for n in z.namelist())
    result={'files':len(records),'bytes':target.stat().st_size,'sha256':sha(target),'crc_passed':True,'all_manifest_hashes_passed':True,'all_14_run_checkpoints_included':True,'target':target.name}
    (destination/'final_package_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pdf-reviewed',action='store_true');args=parser.parse_args()
    package(verify(args.pdf_reviewed))
