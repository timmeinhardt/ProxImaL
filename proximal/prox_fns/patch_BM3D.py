from .prox_fn import ProxFn
import numpy as np
import pybm3d


class patch_BM3D(ProxFn):
    """The function for BM3D patch prior
    """

    def __init__(self, lin_op, sigma_fixed=0.0, sigma_scale=6.0, patch_size=8, **kwargs):
        # Check for the shape
        if not (len(lin_op.shape) == 2 or len(lin_op.shape) == 3 and lin_op.shape[2] in [1, 3]):
            raise ValueError('BM3D needs a 3 or 1 channel image')

        self.sigma_fixed = sigma_fixed
        self.patch_size = patch_size
        self.sigma_scale = sigma_scale

        super(patch_BM3D, self).__init__(lin_op, **kwargs)

    def _prox(self, rho, v, *args, **kwargs):
        """x = denoise_gaussian_BM3D( tonemap(v), sqrt(1/rho))
        """
        if self.sigma_fixed > 0.0:
            sigma = self.sigma_fixed / 30.0 * self.sigma_scale
        else:
            sigma = np.sqrt(1.0 / rho)

         # Scale d
        v = v.copy()
        v_min = np.amin(v)
        v_max = np.amax(v)
        v_max = np.maximum(v_max, v_min + 0.01)

        # Scale and offset parameters d
        v -= v_min
        v /= (v_max - v_min)

        if len(v.shape) == 2:
            v = v.reshape(v.shape  + (1,))

        dst = np.array(pybm3d.bm3d.bm3d(v.astype(np.float32), sigma=sigma, patch_size=self.patch_size))
        dst = np.nan_to_num(dst).astype(v.dtype) * (v_max - v_min) + v_min

        np.copyto(v, dst)

        return np.squeeze(v)

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
        return [self.sigma_fixed, self.patch_size]
