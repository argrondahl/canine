import matplotlib.pyplot as plt
import h5py
import numpy as np

with h5py.File('../canine_ds_3d.h5', 'r') as f:
    x = f['fold_2']['input'][:]
    y = f['fold_2']['target'][:]

i = 0
slice = 100

plt.imshow(x[i, slice][...,0], 'gray')
plt.contour(y[i, slice][...,0])
plt.show()

all_vox = x[i].flatten()

np.percentile(all_vox[all_vox > -700], 0.5)
np.percentile(all_vox[all_vox > -700], 99.5)


np.percentile(all_vox, 0.5)
np.percentile(all_vox, 99.5)

plt.hist(all_vox, bins=50)
plt.show()

np.percentile(x[i][x[i]>-3024].flatten(), 0.5)
np.percentile(x[i][x[i]>-3024].flatten(), 99.5)

plt.imshow(x[i, slice][...,0].clip(-3024.0, 1409), 'gray')
plt.contour(y[i, slice][...,0])
plt.show()


plt.imshow(x[i, slice][...,0].clip(-3023.0, 1700), 'gray')
plt.contour(y[i, slice][...,0])
plt.show()

i=0
plt.imshow(x[i, slice][...,0].clip(-912, 1550), 'gray')
plt.contour(y[i, slice][...,0])
plt.show()


i=2
slice=130
plt.imshow((x[i, slice][...,0] - 100).clip(-500, 500), 'gray')
plt.contour(y[i, slice][...,0])
plt.show()

check_x = x.flatten()[y.flatten()>0]

np.percentile(check_x, 0.5)
np.percentile(check_x, 99.5)


np.percentile(check_x, 50)

np.percentile(check_x, 10)
np.percentile(check_x, 90)