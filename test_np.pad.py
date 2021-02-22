import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
aa = np.pad(a, [0, 5])
print(a, aa, a[:3])

b = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 9]]).T
bb = np.pad(b, [[0, 5], [0, 0]])
print(b)
print(bb)
print(b.shape)
