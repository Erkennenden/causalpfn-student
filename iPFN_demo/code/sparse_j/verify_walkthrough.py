"""Execute the original walkthrough unchanged, saving a separate evidence copy."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=Path('../../notebooks/current_ipfn_student_walkthrough.ipynb'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = args.notebook.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = original.read_bytes()
    # Kernel registration is task-local; do not alter the user's Jupyter setup.
    runtime = output/'runtime'
    spec = runtime/'kernels'/'e12-evidence'
    spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='E1.2 evidence Python',language='python')))
    os.environ['JUPYTER_PATH'] = str(runtime)+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime/'connections')
    os.environ['IPYTHONDIR'] = str(runtime/'ipython')
    os.environ['MPLCONFIGDIR'] = str(runtime/'matplotlib')
    os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
    os.environ['PYTHONUTF8'] = '1'
    from nbclient import NotebookClient
    import nbformat
    notebook = nbformat.reads(raw.decode('utf-8'),as_version=4)
    sources = [cell.source for cell in notebook.cells]
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    count = sum(c.cell_type=='code' for c in notebook.cells)
    began = time.perf_counter()
    print(f'Executing {count} code cells from the unchanged original notebook',flush=True)
    client = NotebookClient(notebook, timeout=300, kernel_name='e12-evidence',
                            resources={'metadata':{'path':str(original.parent)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook,output/'walkthrough_executed.ipynb')
    assert [cell.source for cell in notebook.cells] == sources
    assert original.read_bytes() == raw
    errors = [o for c in notebook.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
    assert not errors and all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
    text_outputs = []
    images = 0
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        for item in cell.outputs:
            if item.output_type=='stream':
                text_outputs.append(f'CELL {i}\n{item.text}')
            elif 'data' in item:
                if 'text/plain' in item.data:
                    text_outputs.append(f'CELL {i}\n{item.data["text/plain"]}')
                images += int('image/png' in item.data)
    (output/'execution_text.txt').write_text('\n'.join(text_outputs),encoding='utf-8')
    record = dict(original_sha256=hashlib.sha256(raw).hexdigest(),
        original_notebook=str(original), unchanged_cell_sources=True,
        executed_code_cells=count, errors=0, rendered_png_outputs=images,
        elapsed_seconds=time.perf_counter()-began,
        python=sys.version, executable=sys.executable,
        versions={m:importlib.metadata.version(m) for m in
            ('numpy','pandas','torch','scipy','networkx','ipykernel','nbclient','nbformat')},
        checks='Original assertions executed, including DGP formula/truth, ITE/ATE contrasts, baseline J agreement, small-graph ensemble, and two CLI comparisons.')
    (output/'provenance.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)


if __name__ == '__main__':
    main()
