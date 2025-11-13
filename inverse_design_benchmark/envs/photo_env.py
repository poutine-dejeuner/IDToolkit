from copy import deepcopy
import numpy as np

from .base import EnvBase
from ..parameter_space.combine import CombineSpace
from ..parameter_space.uniform import UniformSpace

# from .tpv_model import simulate, resol, load_target
from nanophoto.meep_compute_fom import compute_FOM as simulate


class PhotoEnv(EnvBase):

    def __init__(self, seed=0, save_model=False, substitute_model_name='', ensemble=False):
        super().__init__("photo", seed, save_model, substitute_model_name, ensemble)
        self.target = 0.5
        self.imshape = (101, 91)

    def env_forward(self, param, force_numerical=False):
        self.parameter_space.check(param)
        h, w = self.imshape
        param = self.process_param(param)
        param = self.parameter_space.to_numpy(param).reshape(-1, h, w)

        if not force_numerical and self.substitute_model_name:
            result = self.env_forward_by_models(param)
        else:
            result = simulate(param)
        return result

    def process_param(self, param):
        new_param = deepcopy(param)
        for k in new_param.keys():
            new_param[k] = np.clip(new_param[k], 0, 1)
        return new_param

    def score(self, value):
        _score = -np.mean((value - self.target)**2)
        return _score

    @property
    def parameter_space(self):
        # la parametrisation des design: pixels, somme de gaussiennes ...
        if not hasattr(self, "_parameter_space"):
            spaces = {(i, j): UniformSpace(0, 1) for i in range(self.imageh) for
                     j in range(self.imagew)}
            self._parameter_space = CombineSpace(space_dict=spaces)
        return self._parameter_space
    
    @property
    def get_input_dim(self):
        return self.imageh * self.imagew
    
    @property
    def get_output_dim(self):
        return self.imageh * self.imagew
