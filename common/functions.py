import numpy as np

def softmax(x):
    if x.ndim == 2:
        x -= x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x -= np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size: # 如果是one-hot，将t转换为数字标签形式
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    # 当t和y都为one-hot形式时，使用下式
    # return -np.sum(t * np.log(y + 1e-7)) / batch_size
    # 下式当t为非one-hot，而是[0, C-1]之间的数字时使用
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size