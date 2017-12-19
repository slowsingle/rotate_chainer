import chainer
from myRotate import link_rotate as LR
import chainer.functions as F
from chainer import Variable
import numpy as np


class Net(chainer.Chain):
    def __init__(self):
        super(Net, self).__init__(
            rotate=LR.Rotate2d(),
        )

    def __call__(self, X, Y):
        predY = self.predict(X)
        loss = F.mean_squared_error(predY, Y)
        return loss

    def predict(self, X):
        return self.rotate(X)


init_r = np.pi / 6.0
init_t = np.array([0.04, -0.05])

def make_dataset(n_data=50):
    X = np.random.uniform(low=-1.0, high=1.0, size=(n_data, 2)).astype(np.float32)
    r = init_r
    R = np.array([[np.cos(r), -np.sin(r)],
                  [np.sin(r),  np.cos(r)]])
    Y = X.dot(R.T) + init_t.reshape(1, 2)
    return X.astype(np.float32), Y.astype(np.float32)

X, Y = make_dataset(n_data=10)
print(X.shape, Y.shape)

n_epoch = 300

model = Net()
optimizer = chainer.optimizers.SGD(lr=0.1)
optimizer.setup(model)

for epo in range(n_epoch):
    model.cleargrads()
    loss = model(Variable(X), Variable(Y))
    loss.backward()
    optimizer.update()

    if (epo + 1) % 20 == 0:
        print('*** epoch {} ***'.format(epo + 1))
        print('loss :', loss.data)
        print('learned r :', model.rotate.r.data[0])
        print('learned t :', model.rotate.b.data)

print('---------------')
print('init r', init_r)
print('init t', init_t)