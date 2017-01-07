from .lin_op import LinOp
import numpy as np
from proximal.utils.utils import Impl
from proximal.halide.halide import Halide


def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


class mul_elemwise(LinOp):
    """Elementwise multiplication weight*X with a fixed constant.
    """

    def __init__(self, weight, arg, implem=None):
        assert (is_broadcastable(weight.shape, arg.shape) or \
            is_broadcastable(weight.shape[::-1], arg.shape[::-1]))

        #TODO: find broadcasted shape
        if sum(weight.shape) > sum(arg.shape):
            shape = weight.shape
        else:
            shape = arg.shape

        self.weight = weight

        input_nodes = [arg]
        if isinstance(weight, LinOp):
            input_nodes.append(weight)

        super(mul_elemwise, self).__init__(input_nodes, shape, implem)

    def forward(self, inputs, outputs):
        """The forward operator.

        Reads from inputs and writes to outputs.
        """
        if isinstance(self.weight, LinOp):
            if len(inputs) == 2:
                weight = inputs[1]
            else:
                weight = outputs[1]
        else:
            weight = self.weight

        if self.implementation == Impl['halide'] and (len(self.shape) in [2, 3]):
            # Halide implementation

            weight = np.asfortranarray(weight.astype(np.float32))
            tmpout = np.zeros(weight.shape, dtype=np.float32, order='F')

            tmpin = np.asfortranarray(inputs[0].astype(np.float32))
            Halide('A_mask.cpp').A_mask(tmpin, weight, tmpout)  # Call
            np.copyto(outputs[0], tmpout)

        else:
            # Numpy
            transposed = False
            # multiplication is not broadcastable in trailing but leading dimensions
            if not is_broadcastable(inputs[0].shape, weight.shape) and \
                is_broadcastable(inputs[0].shape[::-1], weight.shape[::-1]):
                transposed = True
                weight = weight.T
                inputs[0] = inputs[0].T
                outputs[0] = outputs[0].T

            np.multiply(inputs[0], weight, outputs[0])

            if transposed:
                weight = weight.T
                inputs[0] = inputs[0].T
                outputs[0] = outputs[0].T

    def adjoint(self, inputs, outputs):
        """The adjoint operator.

        Reads from inputs and writes to outputs.
        """
        self.forward(inputs, outputs)

    def is_diag(self, freq=False):
        """Is the lin op diagonal (in the frequency domain)?
        """
        return not freq and self.input_nodes[0].is_diag(freq)

    def get_diag(self, freq=False):
        """Returns the diagonal representation (A^TA)^(1/2).

        Parameters
        ----------
        freq : bool
            Is the diagonal representation in the frequency domain?
        Returns
        -------
        dict of variable to ndarray
            The diagonal operator acting on each variable.
        """
        assert not freq
        if isinstance(self.weight, LinOp):
            self_diag = self.input_nodes[1].get_diag(freq)
        else:
            self_diag = np.reshape(self.weight, self.size)

        var_diags = self.input_nodes[0].get_diag(freq)
        for var in list(var_diags.keys()):
            var_diags[var] = var_diags[var] * self_diag
        return var_diags

    def norm_bound(self, input_mags):
        """Gives an upper bound on the magnitudes of the outputs given inputs.

        Parameters
        ----------
        input_mags : list
            List of magnitudes of inputs.

        Returns
        -------
        float
            Magnitude of outputs.
        """
        if isinstance(self.weight, LinOp):
            return input_mags[0] * input_mags[1]
        else:
            return np.max(np.abs(self.weight)) * input_mags[0]
