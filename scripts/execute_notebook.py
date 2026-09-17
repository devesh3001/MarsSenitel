"""Execute the notebook with this project's interpreter and retain real outputs."""
from pathlib import Path
import os,json,sys
import argparse
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
kernel=ROOT/'tmp/jupyter/kernels/mars-project'
kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Mars project Python','language':'python'}))
os.environ['JUPYTER_PATH']=str(ROOT/'tmp/jupyter')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'tmp/jupyter/runtime')
os.environ['IPYTHONDIR']=str(ROOT/'tmp/ipython')
parser=argparse.ArgumentParser()
parser.add_argument('--notebook',default='notebooks/Mars_HiRISE_Submission.ipynb')
target=ROOT/parser.parse_args().notebook
notebook=nbformat.read(target,as_version=4)
nbformat.validate(notebook)
client=NotebookClient(notebook,timeout=7200,kernel_name='mars-project',resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
try:
    client.execute()
finally:
    nbformat.write(notebook,target)
errors=[o for c in notebook.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
if errors:raise RuntimeError(f'{len(errors)} notebook errors')
print('Executed',sum(c.cell_type=='code' for c in notebook.cells),'code cells with no errors:',target)
