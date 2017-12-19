from chainer import function
from chainer.utils import type_check
import numpy as np


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

class Rotate2dFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, r_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            r_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            r_type.ndim == 1,
            r_type.shape[0] == 1,
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == 2,
            )

    def forward(self, inputs):
        x = _as_mat(inputs[0])
        r = inputs[1]

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(r): {0}, type(x): {1}'
                             .format(type(r), type(x)))

        R = np.array([[np.cos(r), -np.sin(r)],
                      [np.sin(r),  np.cos(r)]]).reshape(2, 2)

        y = x.dot(R.T).astype(x.dtype, copy=False)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        r = inputs[1]
        gy = grad_outputs[0]

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(r): {0}, type(x): {1}'
                             .format(type(r), type(x)))

        R = np.array([[np.cos(r), -np.sin(r)],
                      [np.sin(r),  np.cos(r)]]).reshape(2, 2)
        gx = gy.dot(R.T).astype(x.dtype, copy=False).reshape(inputs[0].shape)

        dR = np.array([[-np.sin(r), -np.cos(r)],
                       [ np.cos(r), -np.sin(r)]]).reshape(2, 2)

        xdR = x.dot(dR.T).astype(x.dtype, copy=False)
        gr = (xdR * gy).sum().reshape(r.shape)

        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gr, gb
        else:
            return gx, gr


def rotate2d(x, r, b=None):
    if b is None:
        return Rotate2dFunction()(x, r)
    else:
        return Rotate2dFunction()(x, r, b)