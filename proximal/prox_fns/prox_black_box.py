from .prox_fn import ProxFn
import numpy as np
import os
import scipy.misc

class prox_black_box(ProxFn):
    """A black-box proxable function specified by the user.
    """

    def __init__(self, lin_op, prox, **kwargs):
        self._prox_operator = prox

        super(prox_black_box, self).__init__(lin_op, **kwargs)

    def _img_logger(self, v, i, log_dir, metric):
        """Input Output logging
        """
        if log_dir is None:
            return
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if i == 0:
            file_names = os.listdir(log_dir)
            for file_name in file_names:
                os.remove(os.path.join(log_dir, file_name))

        if metric is not None:
            dist = metric.eval(v)

            file_name = os.path.join(log_dir,
                                     "iter_%i_%f.png" % (i, dist))
        else:
            file_name = os.path.join(log_dir,
                                     "iter_%i.png" % (i))

        scipy.misc.imsave(file_name, v)

    def _prox(self, rho, v, *args, **kwargs):
        """The prox operator.
        """
        sigma = np.sqrt(1.0 / rho)

        # Params
        if kwargs['verbose'] > 1:
            print("Prox blackbox params are: [sigma = {0}]".format(sigma))
            self._img_logger(v, args[0], kwargs['img_log_dir'], kwargs['metric'])

        v = v.copy()
        v_min = np.amin(v)
        v_max = np.amax(v)
        v_max = np.maximum(v_max, v_min + 0.01)
        # Scale and offset parameters d
        v -= v_min
        v /= (v_max - v_min)

        dst = self._prox_operator(v) * (v_max - v_min) + v_min

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
        return [self._prox_operator]
