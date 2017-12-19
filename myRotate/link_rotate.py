import math

from chainer import cuda
from myRotate import function_rotate as rotate2d
from chainer import initializers
from chainer import link


class Rotate2d(link.Link):
    def __init__(self, init_r=0.0, bias=0.0, nobias=False):
        super(Rotate2d, self).__init__()

        initial_r = init_r
        r_initializer = initializers._get_initializer(initial_r)
        self.add_param('r', 1, initializer=r_initializer)

        if nobias:
            self.b = None
        else:
            initial_bias = bias
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', 2, initializer=bias_initializer)

    def __call__(self, x):
        """Applies the Rotate2d layer.
        Args:
            x (~chainer.Variable): Batch of input vectors.
        Returns:
            ~chainer.Variable: Output of the mylinear layer.
        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(1)
        return rotate2d.rotate2d(x, self.r, self.b)
