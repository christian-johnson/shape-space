import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.special import comb
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
        self.thetaResolution=20

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

    def get_pairs(self, xpix,ypix,k):
        Nx = len(xpix)
        xpair  = []
        ypair  = []
        xpair2 = []
        ypair2 = []
        xpix2, ypix2 = xpix[np.arange(-k, Nx-k)], ypix[np.arange(-k, Nx-k)]
        for i in range(Nx):
               if self.pt_in(xpix[i],ypix[i]) and self.pt_in(xpix2[i],ypix2[i]) :
                   xpair .append(xpix [i])
                   ypair .append(ypix [i])
                   xpair2.append(xpix2[i])
                   ypair2.append(ypix2[i])
        ###############
        return((xpair), (ypair), (xpair2), (ypair2))

    def pixelsInImage(self, xpix, ypix):
        xpixInImage = []
        ypixInImage = []
        weights = np.zeros((len(xpix)))
        for i in range(len(xpix)):
            if self.pt_in(xpix[i], ypix[i]):
                xpixInImage.append(xpix[i])
                ypixInImage.append(ypix[i])
                weights[i] = 1.0
            else:
                xpixInImage.append(xpix[i])
                ypixInImage.append(ypix[i])

        return xpixInImage, ypixInImage, weights

    def rotateVector(self, V):
        N = len(V)
        rotatedData = np.tile(V,(N,1))
        rotatedIndices = np.tile(np.arange(N),(N,1))+np.tile(-1.*np.arange(N),(N,1)).transpose()
        rotatedData = np.take(rotatedData,rotatedIndices.astype(int),axis=None)
        return rotatedData

    def thetaP(self, R, i, j):
        xpix , ypix  = self.radiiPoints(R)
        xpixInImage, ypixInImage, weights = self.pixelsInImage(xpix+i, ypix+j)
        N = np.sum(weights)

        theta = float(N)/float(len(xpix))*2.*np.pi*float(R)#angle swept out by the pixels in the image

        circleData = self.image[xpixInImage, ypixInImage]

        rotatedData = self.rotateVector(circleData)
        rotatedWeights = self.rotateVector(weights)

        correlationArray = np.sum((1.-np.abs(rotatedData[0]-rotatedData))*rotatedWeights,axis=1)/np.sum(rotatedWeights, axis=1)
        thetaArray = np.linspace(0.0, theta/R, N)
        """
        print('R = ' + str(R))
        print('xpix, xpixInImage=' + str(len(xpix)) + ' ' + str(len(xpixInImage)))
        print('theta = ' + str(thetaArray))
        print('theta-np.pi*R = ' + str(theta-np.pi*R))
        print('weights = ' + str(weightsArray))
        """
        weightsArray = np.sum(rotatedWeights, axis=1)
        interpolatedCorrelationArray = griddata(thetaArray, correlationArray, 2.*np.pi*np.linspace(0.0, 1.0, self.thetaResolution), method='linear')
        interpolatedWeightsArray = griddata(thetaArray, weightsArray, 2.*np.pi*np.linspace(0.0, 1.0, self.thetaResolution), method='linear')
        return 2.*np.pi*np.linspace(0.0, 1.0, self.thetaResolution), interpolatedCorrelationArray#, interpolatedWeightsArray

    def similarity(self, p1, p2):
        return np.abs(p1-p2)
    def similarity3p(self, p1, p2, p3):
        return 0.333*self.similarity(p1, p2)+0.333*self.similarity(p2, p3)+ 0.333*self.similarity(p1,p3)
    def ratios(self):
        Nx = self.image.shape[0]
        Ny = self.image.shape[1] #Number of pixels in the image dimensions
        x = np.arange(Nx)
        y = np.arange(Ny)

        xx,yy = np.meshgrid(x,y)
        points = np.concatenate([np.array([xx.ravel()]), np.array([yy.ravel()])], axis=0).transpose()
        N = len(points)
        r1r2 = np.zeros((int(comb(N,3))))
        r2r3 = np.zeros((int(comb(N,3))))
        weights = np.zeros((int(comb(N,3))))
        s = 0
        for i in range(0, N-1):
            print(i)
            if self.image[points[i][0],points[i][1]]>0.0:
                for j in range(i+1, N-1):
                    for k in range(j+1, N-1):

                        weights[s] = self.similarity3p(self.image[points[i][0],points[i][1]],self.image[points[j][0],points[j][1]],self.image[points[k][0],points[k][1]])

                        distList = np.sort(pdist(np.concatenate([np.array([points[i]]), np.array([points[j]]), np.array([points[k]])], axis=0)))

                        r1r2[s] = distList[0]/distList[1]
                        r2r3[s] = distList[1]/distList[2]
                        s += 1
        r1r2 = r1r2[:s]
        r2r3 = r2r3[:s]
        weights = weights[:s]
        norm, xedges, yedges = np.histogram2d(r1r2, r2r3, bins=30)
        data, xedges, yedges = np.histogram2d(r1r2, r2r3, weights=weights, bins=30)
        plt.subplot(121)
        plt.imshow(self.image)
        plt.subplot(122)
        plt.imshow(data/norm)
        plt.show()
        input('wait for key')
        #distance ratios
        #Make 3 copies of the pairs
        pairwiseDistances = scipy.spatial.distance.pdist(np.concatenate([np.array([xx.ravel()]), np.array([yy.ravel()])], axis=0).transpose())
        pairwiseDistances = pairwiseDistances[np.nonzero(pairwiseDistances)]
        length = len(pairwiseDistances)
        rows = np.tile(np.array([pairwiseDistances]).transpose(), (1, length))
        columns = 1.0/np.tile(np.array([pairwiseDistances]), (length, 1))

        distMatrix = rows*columns

        #thetas
        thetas = np.arctan2(yy-vv, xx-ww)-np.arctan2(yy-tt, xx-ss)
        weights = 0.5*(2.*self.image[xx]-self.image[ww]-self.image[ss]+2.*self.image[yy]-self.image[vv]-self.image[tt])
        input('wait for key')
        print(weights)
        xedges = np.linspace(0.01, 1.0, 50)
        yedges = np.linspace(0.01, 2.*np.pi, 50)
        myhist, xedges, yedges = np.histogram2d(distRatios.ravel(), thetas.ravel(), weights = weights,bins = (xedges, yedges))
        plt.imshow(myhist)
        plt.show()
        #^ control histogram location
        #pixel similarity


    def dotted_P(self, R, i, j, offset):
        """
        returns <P1|P2>(R) at pixel(i,j) with offset k
        """
        xpix , ypix  = self.radiiPoints(R)
        N = len(xpix)
        k = int(N/offset)
        N2 = int(len(xpix)/2)
        if k < 0 : k = N2

        xpix2, ypix2 = xpix[np.arange(-k, len(xpix)-k)], ypix[np.arange(-k, len(ypix)-k)]

        xpair, ypair, xpair2, ypair2 = self.get_pairs(xpix+i,ypix+j,k)
        P1 = self.image[xpair , ypair ]
        P2 = self.image[xpair2, ypair2]
        N2 = int(len(xpair)/2)
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
        S =  np.zeros((Rmax, self.thetaResolution))
        count = np.zeros(Rmax)
        for i in range(self.N[0]):
            for j in range(self.N[1]):
                print(i,j)
                # loop over pixels
                for r in rr:
                    #loop over r
                    rind = r - 1
                    #if (self.image[i,j] >= 0.):
                    if True :
                    #if (i == self.N[0]/2-10 and j == self.N[1]/2-10) :
                        tempTheta, tempP = (self.thetaP(r, i, j))
                        if np.sum(tempP) > 0. : count[rind] += 1
                        S[rind,:] += tempP
        ###############
        #for r in rr:
        #    if (count[r-1] > 0): S[r-1] *= 1./count[r-1]
        return(rr, S)
def main():
    return 0
if __name__ == '__main__':
    main()
