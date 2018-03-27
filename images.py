import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm

def dispMatrix(M):
    plt.imshow(M,interpolation=None)
    plt.show()
def dispMatrices(M1, M2):
    plt.subplot(121)
    plt.imshow(M1)
    plt.subplot(122)
    plt.imshow(M2)#, norm=cm.PowerNorm(gamma=2.6), vmin=M2.min(), vmax = M2.max())
    plt.show()

#Define list of points that correspond to circle of radius R
def radiiPoints(R):
    width = 2*R+1
    xspace = np.arange(width)-R
    yspace = np.arange(width)-R
    xx, yy= np.meshgrid(xspace, yspace)
    dist = np.sqrt(xx**2+yy**2)
    xpts = np.nonzero((dist<R+0.5) & (dist>R-0.5))[0]-R
    ypts = np.nonzero((dist<R+0.5) & (dist>R-0.5))[1]-R
    order = np.argsort(np.arctan2(xpts, ypts))
    return xpts[order], ypts[order]
