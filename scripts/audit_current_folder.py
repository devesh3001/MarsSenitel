"""Read-only project checks; write evidence only under outputs/folder_audit."""
from pathlib import Path
import ast, csv, hashlib, io, json, os
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/folder_audit'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    result = {'recorded_at': datetime.now().astimezone().isoformat(), 'checks': {}}
    inventory = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.venv', '.git', '__pycache__', '.pytest_cache'}]
        for name in files:
            path = Path(base)/name
            if OUT in path.parents: continue
            inventory.append({'path':path.relative_to(ROOT).as_posix(), 'bytes':path.stat().st_size})
    pd.DataFrame(inventory).to_csv(OUT/'inventory.csv', index=False)
    result['inventoried_files_excluding_runtime_caches'] = len(inventory)
    syntax = []
    for p in [*ROOT.glob('*.py'), *ROOT.glob('mars_anomaly/*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('tests/*.py')]:
        text = p.read_text(encoding='utf-8-sig')
        try:
            tree = ast.parse(text)
            error = None
            definitions = [n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        except SyntaxError as e:
            error, definitions = str(e), []
        syntax.append({'path':p.relative_to(ROOT).as_posix(),'lines':len(text.splitlines()),'sha256':digest(p.read_bytes()),'syntax_error':error,'definitions':definitions})
    result['python_sources'] = syntax
    original = json.loads((ROOT/'outputs/audit/audit.json').read_text())
    result['original_input_hashes_match'] = {r['file']:digest((ROOT/r['file']).read_bytes())==r['sha256'] for r in original['manifest']}
    with ZipFile(next(ROOT.glob('DATASETS-*.zip'))) as outer:
        result['metadata_bytes_match_archive'] = {n:(ROOT/'data'/n).read_bytes()==outer.read('DATASETS/'+n) for n in ['crop_metadata_index.csv','source_image_metadata.csv']}
        with ZipFile(io.BytesIO(outer.read('DATASETS/images.zip'))) as inner:
            differences, invalid, hashes = [], [], Counter()
            for e in inner.infolist():
                if e.is_dir(): continue
                name=Path(e.filename).name
                raw=(ROOT/'data/images'/name).read_bytes()
                if raw!=inner.read(e):differences.append(name)
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    if im.mode!='L' or im.size!=(227,227):invalid.append(name)
                    hashes[digest(np.asarray(im).tobytes())]+=1
            result['data_recheck']={'decoded_images':sum(hashes.values()),'changed_images':differences,'invalid_contract':invalid,'exact_duplicate_groups':sum(v>1 for v in hashes.values())}
    stats=pd.read_csv(ROOT/'outputs/audit/image_statistics.csv').set_index('filename')
    stats['brightness_percentile']=stats['mean'].rank(pct=True)*100
    runs, frames = [], {}
    for folder in sorted((ROOT/'outputs').iterdir()):
        if not (folder/'config.json').exists():continue
        config=json.loads((folder/'config.json').read_text())
        history=pd.read_csv(folder/'history.csv')
        best=history.loc[history.validation_loss.idxmin()]
        split=pd.read_csv(folder/'split_manifest.csv')
        row={'run':folder.name,'config':config,'epochs':len(history),'best_epoch':int(best.epoch),'validation_mse':best.validation_mse,'validation_ssim':best.validation_ssim,'validation_gradient':best.validation_gradient,'source_leakage':bool((split.groupby('source_image_id').split.nunique()>1).any())}
        analysis=folder/'mixture_three_sigma_trees2000'
        if (analysis/'diagnostics.json').exists():
            d=json.loads((analysis/'diagnostics.json').read_text())
            cal=json.loads((analysis/'calibration.json').read_text())
            scores=pd.read_csv(analysis/'novelty_scores.csv').set_index('filename')
            selected=pd.read_csv(analysis/'selected_for_interpretation.csv')
            actual=scores[scores.novelty_score>cal['threshold']].sort_values('novelty_score',ascending=False)
            names=np.load(analysis/'latent_filenames.npy',allow_pickle=False)
            latents=np.load(analysis/'latents.npy',mmap_mode='r')
            row.update(flags=len(actual),threshold=cal['threshold'],selected=selected.filename.tolist(),selected_matches_threshold_top5=selected.filename.tolist()==actual.head(5).index.tolist(),flags_match_csv=bool(np.array_equal(scores.flagged,scores.novelty_score>cal['threshold'])),flags_match_diagnostics=len(actual)==d['flagged_total'],latent_shape=list(latents.shape),latent_finite=bool(np.isfinite(latents).all()),latent_filename_alignment=bool(np.array_equal(names,scores.index)),top5_brightness_median=float(stats.loc[selected.filename,'brightness_percentile'].median()) if len(selected) else None,missing_heatmaps=[n for n in selected.filename if not (analysis/f'heatmap_{Path(n).stem}.png').exists()],forest_min_jaccard=min((r['flag_jaccard'] for r in d['forest_seed_stability'] if r['flag_jaccard'] is not None),default=None))
            frames[folder.name]=scores
        runs.append(row)
    result['runs']=runs
    stability=[]
    for base, others in [('v3',['robust_seed','robust_split']),('v7',['v7_robust_seed','v7_robust_split'])]:
        sets=[]
        for name in [base,*others]:sets.append(set(frames[name].index[frames[name].flagged]))
        for name, flagset in zip(others,sets[1:]):
            stability.append({'left':base,'right':name,'jaccard':len(sets[0]&flagset)/len(sets[0]|flagset) if sets[0]|flagset else None})
        result[base+'_three_way_flag_intersection']=sorted(set.intersection(*sets))
    result['independent_stability']=stability
    result['notebooks']=[]
    for p in [*ROOT.glob('notebooks/*.ipynb'),*ROOT.glob('output/submission/*.ipynb')]:
        nb=json.loads(p.read_text(encoding='utf-8')); codes=[c for c in nb['cells'] if c['cell_type']=='code']
        sources='\n\n'.join(''.join(c['source']) for c in nb['cells'])
        (OUT/(p.parent.name+'_'+p.stem+'_source.txt')).write_text(sources,encoding='utf-8')
        result['notebooks'].append({'path':p.relative_to(ROOT).as_posix(),'code_cells':len(codes),'executed_cells':sum(c.get('execution_count') is not None for c in codes),'error_outputs':sum(o.get('output_type')=='error' for c in codes for o in c.get('outputs',[])),'sha256':digest(p.read_bytes())})
    verify=json.loads((ROOT/'outputs/improved_verification.json').read_text())
    result['improved_verified_hash_matches']={key:digest((ROOT/path).read_bytes())==verify[key] for key,path in [('notebook_sha256','notebooks/Mars_HiRISE_Colab.ipynb'),('pdf_sha256','output/pdf/Mars_HiRISE_Improved_Report.pdf')]}
    result['packages']=[]
    for p in (ROOT/'output/submission').glob('*.zip'):
        with ZipFile(p) as z:
            records=json.loads(z.read('submission_manifest.json'))
            mismatches=[r['path'] for r in records if digest(z.read(r['path']))!=r['sha256']]
            current_changes=[r['path'] for r in records if not (ROOT/r['path']).is_file() or digest((ROOT/r['path']).read_bytes())!=r['sha256']]
            result['packages'].append({'path':p.name,'manifest_files':len(records),'crc_failure':z.testzip(),'manifest_hash_failures':mismatches,'workspace_differs_from_archive':current_changes,'has_v7_checkpoint':'outputs/v7/best.pt' in z.namelist(),'has_v8_checkpoint':'outputs/v8/best.pt' in z.namelist()})
    result['fusion']=[]
    for p in (ROOT/'outputs').glob('*/fusion/fused_novelty_scores.csv'):
        data=pd.read_csv(p);calpath=p.parent/'calibration.json'
        record={'path':p.relative_to(ROOT).as_posix(),'rows':len(data),'calibration_exists':calpath.exists()}
        if calpath.exists():
            cal=json.loads(calpath.read_text());record['threshold']=cal['threshold'];record['flags']=int((data.novelty_score>cal['threshold']).sum());record['top5']=data[data.novelty_score>cal['threshold']].nlargest(5,'novelty_score').filename.tolist()
        result['fusion'].append(record)
    (OUT/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['python_sources','runs','notebooks','packages']},indent=2))
    for r in runs: print(r['run'],r['epochs'],r.get('flags'),r.get('top5_brightness_median'),r.get('selected_matches_threshold_top5'))
    for r in result['packages']:print(json.dumps(r))
    for r in result['notebooks']:print(json.dumps(r))

if __name__=='__main__':main()
