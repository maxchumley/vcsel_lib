"""Verify the thin adapter calls VCSEL's parameter and simulation methods."""
import numpy as np
import pytest

from rl.rl_api import Experiment, prepare_vcsel
from rl.rl_api import simulation
from rl.rl_api._compat import core


def test_vcsel_owns_scaling_ramp_history_and_integration(tmp_path,monkeypatch):
    events=[]
    class VCSEL:
        def __init__(self,physical):self.phys=physical
        def scale_params(self):
            events.append('scale')
            self.nd={'steps':5,'phi_p':self.phys['phi_p_mat']}
            return self.nd
        @staticmethod
        def cosine_ramp(times,**options):
            events.append(('ramp',options))
            return np.ones(len(times))
        def generate_history(self,nd,shape,n_cases):
            events.append(('history',shape,n_cases))
            return np.zeros((n_cases,9,4)),np.zeros((n_cases,3)),None,None
        def integrate(self,history,nd,**options):
            events.append(('integrate',options))
            return np.arange(5),np.zeros((len(history),9,5)),np.zeros((len(history),3,5))
    monkeypatch.setattr(core,'_load_vcsel_class',lambda:VCSEL)
    config=Experiment('physics',output_directory=tmp_path).backend_config
    kappa=np.arange(18,dtype=float).reshape(2,3,3)
    phi=np.zeros_like(kappa)
    detuning=np.array([-.5,0.,.5])
    vcsel,history,initial=prepare_vcsel(kappa,phi,config,detunings_ghz=detuning)
    np.testing.assert_array_equal(vcsel.phys['kappa_c_mat'],kappa*1e9)
    np.testing.assert_array_equal(vcsel.phys['delta'],detuning*2*np.pi*1e9)
    np.testing.assert_array_equal(vcsel.nd['kappa'],kappa*1e9*config.photon_lifetime_seconds)
    assert vcsel.nd['kappa_case_dependent'] is True
    assert vcsel.nd['phi_p'] is phi
    assert history.shape==(2,9,4)
    assert events[0]=='scale' and events[2]==('history','FR',2)
    result=simulation.simulate(kappa,phi,config,detunings_ghz=detuning,
                               smooth_frequencies=False,progress=True)
    assert len(result)==4
    assert events[-1]==('integrate',{'progress':True,'max_iter':1,'smooth_freqs':False})


def test_prepare_vcsel_checks_case_axes(tmp_path):
    config=Experiment('physics',output_directory=tmp_path).backend_config
    with pytest.raises(ValueError,match='cases,M,M'):
        prepare_vcsel(np.zeros((3,3)),np.zeros((3,3)),config,detunings_ghz=[0.,0.,0.])
