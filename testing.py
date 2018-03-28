import numpy as np
import matplotlib.pyplot as plt

#import project specific dependencies
import image_data as img
from   images import *

#shape/image parameters
N = [20,20]
xmin = [0.,0.]
xmax = [1.,1.]
x0 = [0.3,0.5]
x1 = [0.5, 0.2]
l, h = (.25,.25)
R = 0.2
delta = (xmax[0] - xmin[0] )/N[0]

offset = -1
Rmax = min(N)
circ = True
rect = False

#initialize image
image = img.image_data(N,xmin,xmax)
#add shapes

if rect and circ :
    image.add_circ(x0, R   )
    image.add_rect(x1, l, h)
elif circ :
    image.add_circ(x0, R   )
elif rect :
    image.add_rect(x1, l, h)


#########done with making image#########
image.add_noise(0.02)
image.normalize()
intensity = np.sum(image.image+1)/2.

#plotting
fig, ax = plt.subplots(2)
ax[0].set_aspect(aspect=1)
ax[0].axis('off')
image.show_img(ax[0])
    

#transform image
rr, S = image.transform_img(Rmax,offset)
ax[1].plot(rr*delta,S/intensity)

plt.show()
