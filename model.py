from parameter_handler import ParameterHandler
from block import *
from utils import timeit

def combine_sequential_blocks():
    pass

class Model():
    def __init__(self, par_handler : ParameterHandler, active_blocks=[]):
        self.blocks = {}
        for block, args, kwargs in active_blocks:
            self.blocks[block.__name__] = block(par_handler, *args, **kwargs)
    
    @timeit
    def __call__(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)
    
    def get_network_edges(self):
        edges = []
        for edge_name in self.blocks:
            _input = self.blocks[edge_name].input
            _output = self.blocks[edge_name].output
            for i in _input:
                for o in _output:
                    edges.append([i, o, edge_name])
        return edges
    
    def update_parameter_from_handler(self, par_handler : ParameterHandler):
        for key in self.blocks:
            self.blocks[key].update_parameter_from_handler(par_handler)
    
    def simulate(self, *args, **kwargs):
        pass
    
    
class XENON1T_ERmTIModel(Model):
    def __init__(self, par_handler : ParameterHandler, lower_e, upper_e):
        active_blocks = [
            (EnergySpec,    (), dict(lower=lower_e, upper=upper_e)),
            (QuenchingFano, (), dict()),
            (Ionization,    (), dict()),
            (mTI,           (), dict()),
            (Recomb,        (), dict()),
        ]
        super().__init__(par_handler, active_blocks)
        
    def simulate(self, sim_size):
        energy  = self.blocks['EnergySpec'](sim_size)
        Nq      = self.blocks['QuenchingFano'](energy)
        Ni      = self.blocks['Ionization'](Nq)
        recomb  = self.blocks['mTI'](energy)
        Nph, Ne = self.blocks['Recomb'](Nq, Ni, recomb)
        return Nph, Ne