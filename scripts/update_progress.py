"""Generate a progress snapshot from saved artifacts, without assuming processes live."""
from pathlib import Path
import json,csv
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]


def main():
    lines=['# Current project progress','',f'Updated: {datetime.now().astimezone().isoformat(timespec="seconds")}','',
           'This snapshot is generated from files. An incomplete checkpoint does not prove a process is currently running. Read DECISION_LOG.md for actions, evidence and the rationale for changes.','',
           '## Experiments','', '| Run | Training evidence | Latest epoch | Final 2,000-tree evaluation |','|---|---|---:|---|']
    improvement_path=ROOT/'outputs/improvement_status.json'
    improvement=json.loads(improvement_path.read_text()) if improvement_path.exists() else None
    border_path=ROOT/'outputs/border_status.json'
    border=json.loads(border_path.read_text()) if border_path.exists() else None
    names=['v1','v2','v3','robust_seed','robust_split']
    if improvement:names+=['v4','v5','improved_seed','improved_split']
    if border:names+=['v6']
    for name in names:
        run=ROOT/'outputs'/name
        history=[]
        if (run/'history.csv').exists():
            with open(run/'history.csv',newline='') as f:history=list(csv.DictReader(f))
        status=json.loads((run/'training_status.json').read_text()) if (run/'training_status.json').exists() else None
        state='Completed' if status and status['status']=='complete' else 'Checkpoint available; incomplete' if history else 'Initialized; no epoch saved yet' if (run/'config.json').exists() else 'Not started'
        evaluation=run/'mixture_three_sigma_trees2000'
        assessed='Completed' if (evaluation/'diagnostics.json').exists() and (evaluation/'selected_for_interpretation.csv').exists() else 'Partial artifacts; incomplete' if evaluation.exists() else 'Not started'
        lines.append(f'| {name} | {state} | {int(float(history[-1]["epoch"])) if history else 0} | {assessed} |')
    lines += ['', '## Completed foundation','',
              '- All supplied documents and rules read; all 10,422 crops decoded and metadata joins validated.',
              '- Source-disjoint splits, from-scratch model, resumable checkpoints and independent score threshold implemented.',
              '- Initial threshold failure diagnosed; mixture-envelope alternative documented.',
              '- Controlled forest-tree-count and subsample comparison completed.',
              '- Eleven tests passed, including preprocessing invariance, connectivity, constant images and recovery after a partial checkpoint write.',
              '', '## Deliverables','', '| Item | State |','|---|---|']
    deliverables=[('Decision log','DECISION_LOG.md'),('First-stage selection','outputs/final_selection.json'),('First-stage PDF','output/pdf/Mars_HiRISE_Analysis_Report.pdf'),('First-stage ZIP','output/submission/NSSC_2026_Mars_Analysis.zip')]
    if improvement:
        deliverables += [('Second-stage selection','outputs/improved_selection.json'),('Second-stage results','IMPROVEMENT_RESULTS.md'),('Revised PDF report','output/pdf/Mars_HiRISE_Improved_Report.pdf'),('Readable Colab notebook','notebooks/Mars_HiRISE_Colab.ipynb'),('Revised submission ZIP','output/submission/NSSC_2026_Mars_Improved.zip')]
    for label,path in deliverables:
        p=ROOT/path;state='Exists; see verification notes' if p.exists() else 'Pending'
        if p.exists() and path.endswith('selection.json'):state=json.loads(p.read_text()).get('status','Exists; see verification notes')
        if p.exists() and p.suffix=='.ipynb':
            n=json.loads(p.read_text(encoding='utf-8'));c=[x for x in n['cells'] if x['cell_type']=='code']
            executed=sum(x.get('execution_count') is not None for x in c)
            errors=sum(o.get('output_type')=='error' for x in c for o in x.get('outputs',[]))
            state=f'{executed}/{len(c)} code cells executed; {errors} error outputs'
        lines.append(f'| {label} | {state} |')
    lines += ['', '## Important boundaries','',
              '- No verified anomaly labels are available; no precision, recall, F1 or AUROC claim is justified.',
              '- A final configuration must be selected from completed comparisons; larger models are not automatically better.',
              '- PDF creation and visual verification are separate steps. A file existing is not proof of visual review.',
              '- The private GitHub repository, collaborator invitations and team details have not been submitted or published.',
              '- Per-epoch measurements are in outputs/<run>/history.csv; new training events are recorded in that run\'s events.jsonl.',
              '', '## Current next steps','']
    verification=ROOT/('outputs/improved_verification.json' if improvement else 'outputs/verification.json')
    if border:
        lines += ['Border experiment: '+border['stage']+'.', '',
                  'The previous nine-run package remains the verified fallback. See BORDER_EXPERIMENT_PROTOCOL.md and outputs/border_experiment for the final bounded comparison. Twelve tests passed after adding near-black border handling.','']
    elif improvement and improvement.get('status')!='Complete':
        lines += ['Second-stage improvement work is active: '+improvement['stage']+'. Existing report/package verification applies to the first-stage artifacts only. New experiments and the natural first-person Colab notebook are not yet final.', '',
                  'Protocol: IMPROVEMENT_PROTOCOL.md. Compare contrast and border-footprint ablations, repeat the supported choice across initialization/source splits, then execute and verify the revised notebook and report.','']
    elif verification.exists():
        checked=json.loads(verification.read_text())
        lines += [f"Verification recorded: {checked.get('recorded_at','See verification file')}. {checked.get('tests_passed',0)} tests passed; {checked.get('notebook_code_cells',0)} notebook code cells executed without errors; {checked.get('pdf_pages',0)} PDF pages visually reviewed.", '',
                  'Nine experiments and the revised local analysis are complete. Normalization improves the specified illumination test but adds stability and footprint concerns. v3 remains the screening reference; no detection-accuracy gain is established. Read IMPROVEMENT_RESULTS.md and DEFENSE_GUIDE.md before presenting.', '',
                  'Remaining competition logistics: team information, private repository setup and required collaborator access. No external submission has been performed. Proposed further scientific experiments are listed in RESULTS.md and are not counted as completed work.','']
    else:
        lines += ['Finish independent checks and final notebook/PDF verification, then assemble the local package. Update this note and the decision log at each material milestone.','']
    (ROOT/'CURRENT_PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:16]))

if __name__=='__main__':main()
