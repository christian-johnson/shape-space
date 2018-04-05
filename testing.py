import numpy as np
import matplotlib.pyplot as plt

#import project specific dependencies
import image_data as img
from   images import *

#shape/image parameters
N = [25,25]
xmin = [0.,0.]
xmax = [1.,1.]
x0 = [0.5,0.5]
x1 = [0.5, 0.2]
l, h = (.1,.4)
R = 0.2
delta = (xmax[0] - xmin[0] )/N[0]

offset = -1
Rmax = int(min(N)/2)
circ = False
rect = True

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
#image.add_noise(0.02)
#image.normalize()
intensity = np.sum(image.image+1)/2.
print(image.image[0,0])
image.ratios()
input('wait for key')

#plotting
fig, ax = plt.subplots(1,2, figsize=[9,5])
ax[0].set_aspect(aspect=1)
ax[0].axis('off')
image.show_img(ax[0])
plt.title('Image')


#transform image
#rr, S = image.transform_img(Rmax,offset)
ax[1].imshow(S)
plt.title('Transformed Image')
plt.xlabel('Theta')
plt.ylabel('Radius')
plt.show()
