from .prox_fn import ProxFn
import numpy as np
from proximal.utils.utils import Impl
from proximal.halide.halide import Halide


class l2_constraint(ProxFn):
    """The function ||W.*x||_1.
    """

    def __init__(self, lin_op, sigma_noise=1.0, **kwargs):
        self.sigma_noise = sigma_noise
        super(l2_constraint, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """x = sign(v)*(|v| - |W|/rho)_+
        """
        norm = np.linalg.norm(v.ravel(), 1)

        if norm <= self.sigma_noise:
            return v
        else:
            return self.sigma_noise * v / norm

    def _eval(self, v):
        """Evaluate the function on v (ignoring parameters).
        """
        return np.inf  # TODO: IGNORE FOR NOW

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.sigma_noise]
