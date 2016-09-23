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
        v = v.copy()
        dst = self.__prox(v)

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
