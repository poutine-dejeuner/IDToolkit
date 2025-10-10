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
        # self.target, self.wavelength = load_target()
        self.target = 0.5
        self.imageh = 101
        self.imagew = 101

    def env_forward(self, param, force_numerical=False):
        self.parameter_space.check(param)

        if not force_numerical and self.substitute_model_name:
            result = self.env_forward_by_models(self.parameter_space.to_numpy(param))
        else:
            param = self.process_param(param)
            params = [param["p"], param["ts"], param["tm"]] + \
                [param[f"r{i}"] for i in range(16)]
            params = np.array(params)
            result = simulate(params)
        return result

    def process_param(self, param):
        new_param = deepcopy(param)
        r_bound = (resol, param["p"] / 2)
        for i in range(16):
            new_param[f"r{i}"] = np.clip(param[f"r{i}"],
                                         a_min=r_bound[0], a_max=r_bound[1])
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
        #     spaces = {
        #         "p": UniformSpace(350, 500),
        #         "ts": UniformSpace(30, 130),
        #         "tm": UniformSpace(10, 80),
        #     }
        #     for i in range(16):
        #         spaces[f"r{i}"] = UniformSpace(40, 250)
        #     self._parameter_space = CombineSpace(space_dict=spaces)
        # return self._parameter_space
    
    @property
    def get_input_dim(self):
        return self.imageh * self.imagew
        # le nombre de parametres dans parameter_space
        # return 19
    
    @property
    def get_output_dim(self):
        # return 500
        return self.imageh * self.imagew
