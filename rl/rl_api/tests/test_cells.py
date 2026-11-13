"""Execute cell blocks without __file__, with notebook argv, and no real training."""
import re
import sys

from rl.rl_api import Experiment, CouplingModel, EvaluationBatch
from rl.rl_api import experiments
from rl.rl_api._compat import PACKAGE_DIR, REPOSITORY_DIR
from rl.rl_api.tests.test_workflow import make_model


def read_cells(name):
    text=(PACKAGE_DIR/'experiments'/name).read_text()
    return [cell for cell in re.split(r'(?m)^#\s*%%[^\n]*\n',text) if cell.strip()]


def test_training_cells_preview_then_explicit_start(tmp_path,monkeypatch):
    monkeypatch.chdir(REPOSITORY_DIR/'rl')
    monkeypatch.setattr(experiments,'running_cells',lambda:True)
    monkeypatch.setattr(sys,'argv',['ipykernel_launcher.py','-f','connection.json'])
    calls=[]
    completed=object()
    def train(self,**options):
        calls.append((self,options))
        return completed
    monkeypatch.setattr(Experiment,'train',train)
    cells=read_cells('train_current.py')
    scope={'__name__':'__main__','__package__':None}
    for cell in cells:
        exec(compile(cell,'<training-cell>','exec'),scope)
        if 'CONFIG' in scope:
            scope['OUTPUT_DIRECTORY']=tmp_path/'cell-run'
    assert calls==[]
    assert 'experiment' in scope and '__file__' not in scope
    assert not (tmp_path/'cell-run').exists()
    scope['START_TRAINING']=True
    scope['RUN_NAME']='edited-after-preview'
    start_cell=next(cell for cell in cells if 'result = experiment.train(' in cell)
    exec(compile(start_cell,'<start-cell>','exec'),scope)
    assert scope['result'] is completed
    assert len(calls)==1 and calls[0][0].name=='edited-after-preview'
    assert calls[0][0].config.optimization.workers==14


def test_inspection_is_one_cell_with_editable_settings(tmp_path,monkeypatch,fake_physics):
    model=make_model(tmp_path)
    monkeypatch.chdir(REPOSITORY_DIR/'rl')
    monkeypatch.setattr(experiments,'running_cells',lambda:True)
    monkeypatch.setattr(sys,'argv',['ipykernel_launcher.py','-f','connection.json'])
    monkeypatch.setattr(CouplingModel,'load',lambda *args,**kwargs:model)
    plots=[]
    monkeypatch.setattr(EvaluationBatch,'plot',lambda self,**kwargs:plots.append(kwargs))
    cells=read_cells('inspect_current.py')
    assert len(cells)==1
    scope={'__name__':'__main__','__package__':None}
    for cell in cells:
        cell=cell.replace('NUMBER_OF_CANDIDATES = 100','NUMBER_OF_CANDIDATES = 4')
        # The inspector is user-editable. Supply this test's own four-laser
        # context without editing the user's current inspection settings.
        for name,value in {'N_LASERS':'4', 'N_CLUSTERS':'1',
                'MANUAL_TARGET_PHASES_PI':'None', 'DETUNING_MODE':'"equal"',
                'TOPOLOGY_MATRIX_SOURCE_RECEIVER':'None'}.items():
            cell=re.sub(rf'(?m)^{name}\s*=.*$', f'{name} = {value}', cell)
        exec(compile(cell,'<inspection-cell>','exec'),scope)
    assert '__file__' not in scope
    assert scope['kappa_per_ns'].shape==(4,4,4)
    assert len(fake_physics)==1 and len(plots)==1
    scope['results'].plot(select='min_budget',show=False,options=scope['plot_options']())
    assert len(fake_physics)==1 and len(plots)==2
