import numpy as np
import tensorflow as tf
import generator
from parameter_handler import ParameterHandler

class _BlockBase():
    def __init__(self):
        self.param_names = []
        self.param_values = np.array([])
        self.param_dict = {}
        
        self.input = []
        self.output = []
    
    def __call__(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)
        
    def update_parameter_from_handler(self, par_handler):
        check, missing = par_handler.check_parameter_exist(self.param_names, return_not_exist=True)
        assert check, "%s not found in par_handler!"%missing
        
        self.param_values = par_handler.get_parameter(self.param_names)
        self.param_dict = {key : val for key, val in zip(self.param_names, self.param_values)}
        
        for key, val in zip(self.param_names, self.param_values):
            self.__setattr__(key, val)
    
    def simulate(self, *args, **kwargs):
        pass


class EnergySpec(_BlockBase):
    def __init__(self, par_handler : ParameterHandler, lower, upper):
        super().__init__()
        self.lower = lower
        self.upper = upper
        
        self.input = ['sim_size']
        self.output = ['energy']
        
    def simulate(self, sim_size):
        return generator.uniform(self.lower, self.upper, shape=[sim_size])


class QuenchingFano(_BlockBase):
    def __init__(self, par_handler : ParameterHandler):
        super().__init__()
        self.param_names = ['w', 'lindhard', 'fano']
        self.update_parameter_from_handler(par_handler)
        
        self.input = ['energy']
        self.output = ['Nq']
        
    def simulate(self, energy):
        Nq_avg = energy/self.w
        Nq = generator.normal(Nq_avg, tf.sqrt(Nq_avg*self.fano))
        Nq = tf.math.round(Nq)
        return generator.binomial(Nq, self.lindhard)
    
    
class Ionization(_BlockBase):
    def __init__(self, par_handler : ParameterHandler):
        super().__init__()
        self.param_names = ['ex_ion_ratio']
        self.update_parameter_from_handler(par_handler)
        
        self.input = ['Nq']
        self.output = ['Ni']
        
    def simulate(self, Nq):
        return generator.binomial(Nq, 1./(1.+self.ex_ion_ratio))
        

class mTI(_BlockBase):
    def __init__(self, par_handler : ParameterHandler):
        super().__init__()
        self.param_names = ['w', 'ex_ion_ratio', 'gamma', 'omega', 'delta', 'field', 'q0', 'q1', 'q2', 'q3']
        self.update_parameter_from_handler(par_handler)
        
        self.input = ['energy']
        self.output = ['recomb']
    
    def get_mean_recomb(self, energy):
        ni = energy / self.w / (1+self.ex_ion_ratio)
        ti = ni * self.gamma * tf.exp(-energy/self.omega) * self.field**(-self.delta) / 4.
        fd = 1. / (1. + tf.exp(-(energy-self.q0)/self.q1))
        return (1. - tf.math.log(1. + ti) / ti) * fd
    
    def get_std_recomb(self, energy):
        return self.q2 * (1 - tf.exp(-energy/self.q3))
    
    def simulate(self, energy):
        mean_recomb = self.get_mean_recomb(energy)
        std_recomb = self.get_std_recomb(energy)
        return generator.truncated_normal(mean_recomb, std_recomb, vmin=0., vmax=1.0)

    
class Recomb(_BlockBase):
    def __init__(self, par_handler : ParameterHandler):
        super().__init__()
        
        self.input = ['Nq', 'Ni', 'recomb']
        self.output = ['Nph', 'Ne']
        
    def simulate(self, Nq, Ni, recomb):
        Ne = generator.binomial(Ni, 1.-recomb)
        return Nq-Ne, Ne        
        
        
        
        
        
        
        
        