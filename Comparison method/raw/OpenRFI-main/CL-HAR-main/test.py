import torchvision
import torch
import numpy as np

X = np.array([[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]]
             ,[[20,21],[22,23],[24,25],[26,27],[28,29],[30,31],[32,33],[34,35],[36,37],[38,39]]])
X=torch.Tensor(X)



y_onehot = np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, 1]])
def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]
labels = onehot_to_label(y_onehot)
print(labels)

data_x = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

data_y = np.array([0, 1, 0, 1])

d = np.array([0, 1, 1, 0])


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            np.random.shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[warp, :]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)