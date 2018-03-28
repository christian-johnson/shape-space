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

        self.N = np.array(N)
        self.image = np.zeros((N[0],N[1])) - 1.             
        self.coord = [x, y]

    ##################################################
    ########## visualization #########################

    def show_img(self, ax,  **kwargs):
        x, y = np.meshgrid(self.coord[0], self.coord[1])
        ax.pcolor(x, y, self.image.T, **kwargs)

    ##################################################
    ########## image manipulation ####################

    def normalize(self):
        #renormalizes the image to between -1 & 1
        self.image -=    np.min(self.image)  #sets zero
        self.image *= 2./np.max(self.image)  #normalizes
        self.image -= 1.

    def add_noise(self, amp):
        noise = amp*(2.*np.random.rand(self.N[0],self.N[1])-1.)
        self.image += noise
        
    def add_circ(self, X, R   ):
        R2 = R**2
        X = np.array(X)
        x = self.coord[0]
        y = self.coord[1]
        for i in range(self.N[0]):
            for j in range(self.N[1]):
                r2 = np.sum((X-np.array([x[i],y[j]]))**2)
                if r2 <= R2 :
                    self.image[i,j] = 1.
        #self.normalize()
        ###############
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
                        self.image[i,j] = 1
        #self.normalize()
        ################################

    ##################################################
    ############### Image transformations ############
    def radiiPoints(self, R):
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

    def pt_in(self, x, y):
        return (x >= 0) & (x < self.N[0]) & (y >= 0) & (y < self.N[1])

    def dotted_P(self, R, i, j, k):
        """
        returns <P1|P2>(R) at pixel(i,j) with offset k
        """
        xpix , ypix  = self.radiiPoints(R)
        N2 = int(len(xpix)/2)
        if k < 0 : k = N2
        xpix2, ypix2 = xpix[np.arange(-k, len(xpix)-k)], ypix[np.arange(-k, len(ypix)-k)]
        
        xpair  = xpix [(self.pt_in(xpix+i,ypix+j) & self.pt_in(xpix2+i,ypix2+j))]
        ypair  = ypix [(self.pt_in(xpix+i,ypix+j) & self.pt_in(xpix2+i,ypix2+j))]
        xpair2 = xpix2[(self.pt_in(xpix+i,ypix+j) & self.pt_in(xpix2+i,ypix2+j))]
        ypair2 = ypix2[(self.pt_in(xpix+i,ypix+j) & self.pt_in(xpix2+i,ypix2+j))]
        
        P1 = self.image[xpair , ypair ]
        P2 = self.image[xpair2, ypair2]

        temp = 1-np.abs(P1[:N2]-P2[:N2])
        #temp = np.cos(np.arccos(P1[:N2])-np.arccos(P2[:N2]))
        #temp = P1[:N2]*P2[:N2]

        if len(temp) > 0 :
            return np.sum(temp)/len(temp)
        else :
            return 0.

    def transform_img(self, Rmax, offset):
        '''
        transform to R, S space, up to max value Rmax with offset k
        '''
        rr = np.arange(1,Rmax+1)
        S =  np.zeros(Rmax)

        for i in range(self.N[0]):
            for j in range(self.N[1]):
                # loop over pixels
                for r in rr:
                    #loop over
                    rind = r - 1
                    if (self.image[i,j] >= 0.):
                        S[rind] += abs(self.dotted_P(r, i, j,offset))
        return(rr, S)
          
