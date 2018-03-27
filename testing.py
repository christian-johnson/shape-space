import numpy as np
import matplotlib.pyplot as plt

#import project specific dependencies
import image_data as img
from   images import *

N = [20,20]
xmin = [0.,0.]
xmax = [1.,1.]
x0 = [0.5,0.5]
l, h = (.25,.25)
delta = (xmax[0] - xmin[0] )/N[0]

Rmax = int(min(N)/2)


#fig, ax = plt.subplots()
image = img.image_data(N,xmin,xmax)
rr, norm = image.transform_img(Rmax)

image.add_rect(x0, l, h)
image.show_img(cmap='inferno')

fig, ax = plt.subplots()
rr, S = image.transform_img(Rmax)
ax.plot(rr*delta,S-norm)

plt.show()
