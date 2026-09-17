"""Append the final bounded experiment; preserve the verified previous notebook."""
from pathlib import Path
import ast,json

ROOT=Path(__file__).resolve().parents[1]


class InlineSource(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        return None if node.level or (node.module or '').startswith('mars_anomaly') else node

    def visit_Name(self,node):
        if node.id=='ANALYSIS':node.id='BORDER_ANALYSIS_NAME'
        return node

    def visit_FunctionDef(self,node):
        if node.name=='main':node.name='run_border_response_checks'
        return self.generic_visit(node)

    def visit_If(self,node):
        if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__':return None
        return self.generic_visit(node)


def main():
    evidence=ROOT/'outputs/border_experiment'
    decision=json.loads((evidence/'decision.json').read_text())
    assert decision['visual_review']!='pending'
    assert not decision['promoted'], 'A promoted model requires updating the primary interpretation too'
    notebook=json.loads((ROOT/'notebooks/Mars_HiRISE_Colab.ipynb').read_text(encoding='utf-8'))
    def md(text):notebook['cells'].append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
    def code(text):notebook['cells'].append(dict(cell_type='code',metadata={},source=text.splitlines(True),execution_count=None,outputs=[]))
    for cell in notebook['cells']:
        if cell['cell_type']=='code' and 'def border_zero_mask' in ''.join(cell['source']):
            cell['source']=(ROOT/'mars_anomaly/preprocessing.py').read_text().splitlines(True)
    notebook['cells'][0]['source'].append('\n\n**Final bounded check:** I tested one additional border policy in the appendix. It did not pass all predeclared promotion checks, so v3 remains my reference.\n')
    md('## Final experiment: border handling under a fixed budget\n\n'+
       'I kept the 256-dimensional model, loss, seeds and 15-epoch budget unchanged. I replaced the exact-zero/neutral-fill policy with border-connected values at most 4/255, two pixels of mask expansion and nearest-valid-pixel extension. I still standardized outside the inferred mask. v5 is the matched preprocessing comparator; v3 is the submission reference.\n\n'+
       'The mask is a heuristic. It can remove border-connected shadows, and extension can stretch texture into stripes. I fixed the rules before training rather than adjusting them to obtain attractive candidates.')
    code("""border_run = ROOT/'outputs/v6'
border_config = TrainConfig(version='v6',latent_dim=256,structural_weight=0.1,
                            preprocessing='border_fill',seed=2026,split_seed=2026,epochs=15)
if not (border_run/'training_status.json').exists():
    run_training(DATA,border_run,border_config,resume=(border_run/'last.pt').exists())
assert json.loads((border_run/'training_status.json').read_text())['status']=='complete'
border_analysis = border_run/'mixture_three_sigma_trees2000'
if not (border_analysis/'selected_for_interpretation.csv').exists():
    evaluate(DATA,border_run,bootstrap=100,projection=False,method='mixture_three_sigma',trees=2000)
display(pd.read_csv(border_run/'history.csv').tail())
display(json.loads((border_analysis/'calibration.json').read_text()))
""")
    md('### Fixed response checks\n\n'+
       'On 256 validation crops sampled with seed 73, I compare an eight-column near-black border followed by JPEG compression against a JPEG-only control. The border test removes some edge content, so it is not a perfectly content-preserving nuisance.\n\n'+
       'I separately add a bright square, a dark square and a checkerboard at the center. These are diagnostic perturbations only: I never train the encoder, fit the forest or calibrate a cutoff on them. They do not reveal real-anomaly accuracy. A higher score after a defect is a response check, not proof that a hidden competition anomaly will be detected.')
    tree=InlineSource().visit(ast.parse((ROOT/'scripts/evaluate_border_experiment.py').read_text()))
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROOT' for t in n.targets))
               and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='insert')]
    code(ast.unparse(ast.fix_missing_locations(tree)))
    code("""border_evidence = ROOT/'outputs/border_experiment'
RERUN_BORDER_CHECKS = False
if RERUN_BORDER_CHECKS or not (border_evidence/'response_summary.csv').exists():
    run_border_response_checks()
display(pd.read_csv(border_evidence/'response_summary.csv'))
display(json.loads((border_evidence/'mask_summary.json').read_text()))
display(json.loads((border_evidence/'decision.json').read_text()))
if (border_evidence/'border_examples.png').exists():
    display(DisplayImage(filename=str(border_evidence/'border_examples.png')))
""")
    md('### My decision\n\n'+decision['conclusion']+'\n\n'+
       'I required a 25% reduction in the border score-change percentile against both comparators; at least 70% positive score responses for every defect type, within five percentage points of v3; forest-only flag overlap at least as high as v3; and a satisfactory visual/mask review. I did not relax those checks after seeing the result.\n\n'+
       'A single v6 training run cannot establish initialization or split robustness. These results complete the agreed experiment, not a claim of a perfect detector. A new cloud reproduction needs fresh visual review if candidates change.\n\n'+
       '[Nearest-valid-pixel feature transform documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.ndimage.distance_transform_edt.html)')
    for i,cell in enumerate(notebook['cells']):
        cell['id']=f'mars-final-{i:03d}'
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),f'cell {i}','exec')
            cell['execution_count']=None;cell['outputs']=[]
    notebook['metadata']['colab']['name']='Mars_HiRISE_Final.ipynb'
    target=ROOT/'notebooks/Mars_HiRISE_Final.ipynb'
    target.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    print(target)


if __name__=='__main__':main()
