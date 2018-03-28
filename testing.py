import numpy as np
import matplotlib.pyplot as plt

#import project specific dependencies
import image_data as img
from   images import *

N = [20,20]
xmin = [0.,0.]
xmax = [1.,1.]
x0 = [0.2,0.5]
l, h = (.25,.25)
R = 0.3
delta = (xmax[0] - xmin[0] )/N[0]

Rmax = int(min(N)*2./3.)
Rmax = min(N)

#fig, ax = plt.subplots()
image = img.image_data(N,xmin,xmax)
#rr, norm = image.transform_img(Rmax)
norm = 0.
image.add_circ(x0, R)
x0 = [.75,.2]
image.add_rect(x0, l, h)

image.show_img(cmap='inferno')

fig, ax = plt.subplots()
rr, S = image.transform_img(Rmax)
ax.plot(rr*delta,S-norm)
#ax.set_aspect(aspect=1)
ax.axhline(np.sum(.5*(image.image+1)), color='k', ls='--', label='total intensity')
plt.legend()
plt.show()
