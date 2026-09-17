"""Package the verified second-stage deliverables without uploading them."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    final = json.loads((ROOT / 'outputs/improved_selection.json').read_text())
    verification = json.loads((ROOT / 'outputs/improved_verification.json').read_text())
    if final['status'] != 'Complete' or not verification['pdf_visual_review_passed']:
        raise RuntimeError('Selection or visual verification is incomplete')
    notebook = ROOT / 'notebooks/Mars_HiRISE_Colab.ipynb'
    report = ROOT / 'output/pdf/Mars_HiRISE_Improved_Report.pdf'
    for path, key in [(notebook, 'notebook_sha256'), (report, 'pdf_sha256')]:
        if sha256(path) != verification[key]:
            raise RuntimeError(f'Artifact changed after verification: {path}')
    cells = [c for c in json.loads(notebook.read_text(encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
    if any(c['execution_count'] is None or any(o.get('output_type') == 'error' for o in c['outputs']) for c in cells):
        raise RuntimeError('Notebook has missing execution or error outputs')
    files = [notebook, report]
    files.append(ROOT / 'outputs/test_results_improved.txt')
    for folder in ['mars_anomaly', 'scripts', 'tests']:
        files.extend((ROOT / folder).glob('*.py'))
    files.extend(ROOT.glob('*.md'))
    files.extend(ROOT.glob('requirements*.txt'))
    for name in ['improved_selection.json', 'improved_hypotheses.json', 'improved_verification.json', 'improvement_status.json', 'final_selection.json', 'geological_hypotheses.json']:
        files.append(ROOT / 'outputs' / name)
    files.extend(p for p in (ROOT / 'outputs/comparison').glob('*') if p.is_file())
    for name in ['audit.json', 'random_batch.png', 'image_statistics.csv']:
        files.append(ROOT / 'outputs/audit' / name)
    for name in ['v1', 'v2', 'v3', 'robust_seed', 'robust_split', 'v4', 'v5', 'improved_seed', 'improved_split']:
        folder = ROOT / 'outputs' / name
        if json.loads((folder / 'training_status.json').read_text())['status'] != 'complete':
            raise RuntimeError(f'{name} training incomplete')
        if not (folder / 'mixture_three_sigma_trees2000/selected_for_interpretation.csv').is_file():
            raise RuntimeError(f'{name} evaluation incomplete')
        files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.json', '.jsonl', '.csv', '.png', '.npy', '.joblib'] and 'source_snapshot' not in p.parts)
        files.append(folder / 'best.pt')
    files = sorted(set(files))
    records = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files]
    destination = ROOT / 'output/submission'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'improved_manifest.json'
    manifest.write_text(json.dumps(records, indent=2), encoding='utf-8')
    target = destination / 'NSSC_2026_Mars_Improved.zip'
    with ZipFile(target, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.write(manifest, 'submission_manifest.json')
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP CRC check failed')
        for record in records:
            if hashlib.sha256(archive.read(record['path'])).hexdigest() != record['sha256']:
                raise RuntimeError('Packaged SHA-256 mismatch: ' + record['path'])
    print(f'{len(records)} files; {target.stat().st_size / 1e6:.1f} MB; CRC and all SHA-256 hashes passed; {target}')


if __name__ == '__main__':
    main()
