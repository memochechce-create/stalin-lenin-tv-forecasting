"""Execute every notebook code cell with IPython, preserving rich outputs.

In-process execution also works in restricted environments without localhost
kernel connections. The delivered notebook remains ordinary Jupyter format.
"""
from pathlib import Path
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'output/joint_forecast/final_project.ipynb'

if __name__ == '__main__':
    os.chdir(ROOT)
    notebook = nbformat.read(path, as_version=4)
    config = Config()
    config.HistoryManager.hist_file = ':memory:'
    config.HistoryManager.enabled = False
    shell = InteractiveShell.instance(config=config)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        print(f'Executing cell {count}...', flush=True)
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=False)
        error = result.error_before_exec or result.error_in_exec
        if error:
            raise RuntimeError(f'Notebook cell {count} failed: {error}') from error
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for output in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
        cell.outputs = outputs
        cell.execution_count = count
    notebook.metadata['execution_engine'] = 'IPython in-process; all code cells executed without errors'
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f'Executed all {count} notebook code cells successfully.')
