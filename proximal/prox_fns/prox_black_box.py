from .prox_fn import ProxFn
import numpy as np


class prox_black_box(ProxFn):
    """A black-box proxable function specified by the user.
    """

    def __init__(self, lin_op, prox, **kwargs):
        self.__prox = prox

        super(prox_black_box, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """The prox operator.
        """
        sigma = np.sqrt(1.0 / rho)

        # Params
        if 'verbose' in kwargs and kwargs['verbose'] > 1:
            print("Prox blackbox params are: [sigma = {0}]".format(sigma))

        v = v.copy()
        v_min = np.amin(v)
        v_max = np.amax(v)
        v_max = np.maximum(v_max, v_min + 0.01)
        # Scale and offset parameters d
        v -= v_min
        v /= (v_max - v_min)

        dst = self.__prox(v) * (v_max - v_min) + v_min

        np.copyto(v, dst)

        return v

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
        return [self.__prox]
