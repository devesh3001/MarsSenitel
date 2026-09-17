"""Package local deliverables only. Does not upload or alter access permissions."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,hashlib

ROOT=Path(__file__).resolve().parents[1]


def main():
    final=json.loads((ROOT/'outputs/final_selection.json').read_text())
    if final.get('status')!='Complete':raise RuntimeError('Final selection is still provisional')
    required=['README.md','DECISION_LOG.md','CURRENT_PROGRESS.md','DEFENSE_GUIDE.md','RESULTS.md','ENGINEERING_CHANGELOG.md','SUBMISSION_CHECKLIST.md','output/pdf/Mars_HiRISE_Analysis_Report.pdf','outputs/verification.json']
    for name in required:
        if not (ROOT/name).is_file():raise FileNotFoundError(name)
    verification=json.loads((ROOT/'outputs/verification.json').read_text())
    if not verification.get('pdf_visual_review_passed'):raise RuntimeError('PDF visual review is not recorded')
    # Primary evidence runs — canonical selection is v3; all other runs supply
    # Phase 4 design-journal evidence and robustness comparisons.
    primary_runs = ['v1', 'v2', 'v3', 'robust_seed', 'robust_split']
    journal_runs = ['v4', 'v5', 'v6', 'v7', 'v8',
                    'v7_robust_seed', 'v7_robust_split',
                    'improved_seed', 'improved_split']
    for run in primary_runs:
        folder = ROOT / 'outputs' / run
        if json.loads((folder/'training_status.json').read_text())['status'] != 'complete':
            raise RuntimeError(f'{run} training is incomplete')
        if not (folder/'mixture_three_sigma_trees2000/selected_for_interpretation.csv').exists():
            raise RuntimeError(f'{run} evaluation is incomplete')
    for run in primary_runs + journal_runs:
        folder = ROOT / 'outputs' / run
        status_file = folder / 'training_status.json'
        if status_file.exists():
            status = json.loads(status_file.read_text()).get('status', '')
            if status not in ('complete', 'smoke_complete'):
                print(f'Warning: {run} status={status!r}, including partial artifacts')
    notebook = json.loads((ROOT/'notebooks/Mars_HiRISE_Submission.ipynb').read_text(encoding='utf-8'))
    codes=[c for c in notebook['cells'] if c['cell_type']=='code']
    if any(c['execution_count'] is None for c in codes):raise RuntimeError('Notebook has unexecuted code cells')
    if any(o.get('output_type')=='error' for c in codes for o in c['outputs']):raise RuntimeError('Notebook has error outputs')
    files=[]
    for folder in ['mars_anomaly','scripts','tests']:
        files.extend(p for p in (ROOT/folder).glob('*.py'))
    for name in ['README.md','RESEARCH_AND_APPROACH.md','THRESHOLD_DECISION.md','ENGINEERING_CHANGELOG.md','DECISION_LOG.md','CURRENT_PROGRESS.md','DEFENSE_GUIDE.md','SUBMISSION_CHECKLIST.md','requirements.txt','requirements-lock-local.txt','RESULTS.md']:
        files.append(ROOT/name)
    files.extend([ROOT/'output/pdf/Mars_HiRISE_Analysis_Report.pdf',ROOT/'notebooks/Mars_HiRISE_Submission.ipynb',ROOT/'outputs/final_selection.json',ROOT/'outputs/geological_hypotheses.json',ROOT/'outputs/verification.json'])
    files.extend((ROOT/'outputs/comparison').glob('*'))
    for name in ['audit.json','random_batch.png','image_statistics.csv']:
        files.append(ROOT/'outputs/audit'/name)
    all_runs = primary_runs + journal_runs
    for run in all_runs:
        folder = ROOT / 'outputs' / run
        if not folder.exists():
            continue
        files.extend(p for p in folder.rglob('*')
                     if p.is_file()
                     and p.suffix in ['.json', '.jsonl', '.csv', '.png', '.npy', '.joblib']
                     and 'source_snapshot' not in p.parts)
        best = folder / 'best.pt'
        if best.exists():
            files.append(best)
    files=sorted(set(p for p in files if p.is_file()))
    records=[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    destination=ROOT/'output/submission';destination.mkdir(exist_ok=True,parents=True)
    (destination/'manifest.json').write_text(json.dumps(records,indent=2))
    target=destination/'NSSC_2026_Mars_Analysis.zip'
    with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
        z.write(destination/'manifest.json','submission_manifest.json')
    with ZipFile(target) as z:
        assert z.testzip() is None
    print(len(records),'files;',round(target.stat().st_size/1e6,1),'MB;',target)

if __name__=='__main__':main()
