import numpy as np
import matplotlib.pyplot as plt

#import project specific dependencies
import image_data as img

N = [20,20]
xmin = [0.,0.]
xmax = [1.,1.]
x0 = [0.,0.]
l, w = (.5,.6)


#fig, ax = plt.subplots()
image = img.image_data(N,xmin,xmax)
image.add_rect(x0, l, w)
image.show_img(cmap='inferno')

plt.show()
