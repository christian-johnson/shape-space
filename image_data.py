import numpy as np
import matplotlib.pyplot as plt

class image_data():
    def __init__(self, N, xmin, xmax):
        '''
        N    = [Nx  , Ny]
        xmin = [x0  , y0]
        xmax = [xmax, ymax]
        '''
        x = np.linspace(xmin[0], xmax[0], N[0])
        y = np.linspace(xmin[1], xmax[1], N[1])

        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.N = np.array(N)
        self.image = np.zeros((N[0],N[1]))             
        self.coord = [x, y]

    def show_img(self, **kwargs):
        self.ax.imshow(self.image, **kwargs)

    def normalize(self):
        #renormalizes the image to between -1 & 1
        self.image -=    np.min(self.image)  #sets zero
        self.image *= 2./np.max(self.image)  #normalizes
        self.image -= 1.

    def add_rect(self, X, l, h):
        X = np.array(X)
        x = self.coord[0]
        y = self.coord[1]
        for i in range(self.N[0]):
            for j in range(self.N[1]):
                #I think we can add a rotation here as well if we want
                pt = np.array([x[i],y[j]]) - X
                if (pt[0] > 0. and pt[1] > 0.):
                    if (pt[0] <= l and pt[1] <= h) :
                        #i,j is on the rectangle
                        self.image[i,j] += 1
        self.normalize()
        ################################
