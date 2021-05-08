#%%
import numpy as np
from mutar import GroupLasso
# create some X (n_tasks, n_samples, n_features)
X = np.array([[[3., 1.], [2., 0.]], [[0., 2.], [-1., 3.]]])
print(X.shape)

# and target y (n_tasks, n_samples)
y = np.array([[-3., 1.], [1., -2.]])
print(y.shape)

gl = GroupLasso(alpha=1.)
coef = gl.fit(X, y).coef_
print(coef.shape)

# coefficients (n_featuures, n_tasks)
# share the same support
print(coef)

# %%
from mutar import MultiLevelLasso
X = np.array([[[3, 1], [2, 0], [1, 0]], [[0, 2], [-1, 3], [1, -2]]], dtype=float)
coef = np.array([[1., 1.], [0., -1]])
y = np.array([x.dot(c) for x, c in zip(X, coef.T)])
y += 0.1
mll = MultiLevelLasso(alpha=0.1).fit(X, y)
print(mll.coef_shared_)

print(mll.coef_)
