"""Check artifact consistency; record visual review only when explicitly supplied."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import csv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf-reviewed', action='store_true')
    parser.add_argument('--tests-passed', type=int, required=True)
    parser.add_argument('--test-seconds', type=float, required=True)
    args = parser.parse_args()
    if not args.pdf_reviewed:
        raise RuntimeError('Render and inspect the final PDF before recording visual review')
    selection = json.loads((ROOT/'outputs/improved_selection.json').read_text())
    assert selection['status'] == 'Complete'
    from pypdf import PdfReader
    notebook_path = ROOT/'notebooks/Mars_HiRISE_Colab.ipynb'
    # execute_notebook.py already performs full nbformat schema validation.
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    codes = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert all(c['execution_count'] is not None for c in codes)
    assert not any(o['output_type'] == 'error' for c in codes for o in c['outputs'])
    test_output = (ROOT/'outputs/test_results_improved.txt').read_text(encoding='utf-8-sig')
    assert f'{args.tests_passed} passed in {args.test_seconds:.2f}s' in test_output
    report = ROOT/'output/pdf/Mars_HiRISE_Improved_Report.pdf'
    pages = PdfReader(report).pages
    assert all(len(p.extract_text().strip()) > 100 for p in pages), 'Unexpected near-empty PDF page'
    runs = ['v1','v2','v3','robust_seed','robust_split','v4','v5','improved_seed','improved_split']
    for name in runs:
        folder = ROOT/'outputs'/name
        assert json.loads((folder/'training_status.json').read_text())['status'] == 'complete'
        analysis = folder/'mixture_three_sigma_trees2000'
        threshold = json.loads((analysis/'calibration.json').read_text())['threshold']
        with (analysis/'selected_for_interpretation.csv').open(newline='') as stream:
            selected = list(csv.DictReader(stream))
        assert len(selected) <= 5
        assert all(float(row['novelty_score']) > threshold for row in selected)
        assert all((analysis/f"heatmap_{Path(row['filename']).stem}.png").is_file() for row in selected)
    result = dict(recorded_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                  tests_passed=args.tests_passed, test_observed_seconds=args.test_seconds,
                  test_command='.venv/Scripts/python.exe -m pytest -q',
                  completed_runs=runs, notebook_code_cells=len(codes), notebook_errors=0,
                  notebook_execution='Executed locally, reusing completed training and detector artifacts; visible definitions and sensitivity calculations executed in the notebook',
                  notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                  pdf_pages=len(pages), pdf_visual_review_passed=True,
                  pdf_review='All final rendered pages inspected after the measured results and journal were inserted; captions, tables, images and page boundaries reviewed.',
                  pdf_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                  cloud_execution_verified=False, external_submission_performed=False)
    (ROOT/'outputs/improved_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
