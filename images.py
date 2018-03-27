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

def radiiPoints(R):
    """
    Get ordered list of points that correspond to circle of radius R
    """
    width = 2*R+1
    xspace = np.arange(width)-R
    yspace = np.arange(width)-R
    xx, yy= np.meshgrid(xspace, yspace)
    dist = np.sqrt(xx**2+yy**2)
    xpts = np.nonzero((dist<R+0.5) & (dist>R-0.5))[0]-R
    ypts = np.nonzero((dist<R+0.5) & (dist>R-0.5))[1]-R
    order = np.argsort(np.arctan2(xpts, ypts))
    return xpts[order], ypts[order]

def dottedP(P, R, i, j, k):
    """
    returns <P1|P2>(R) at pixel(i,j) with offset k
    """
    xpix, ypix = radiiPoints(R)
    P1 = P[xpix+j, ypix+i]
    P2 = P1[np.arange(-k, len(P1)-k)]
    return np.cos(np.arccos(P1)-np.arccos(P2))
