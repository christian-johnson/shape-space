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

def corrFunction(M):
    N = np.zeros((M.shape))
    #loop thru points & find the average value of the kernel fxn
    x = np.arange(M.shape[0])
    y = np.arange(M.shape[1])
    xx, yy = np.meshgrid(x,y)
    for i in range(100):
        for j in range(100):
            D = np.sqrt((xx-j)**2+(yy-i)**2)#Distance matrix
            D = np.arctan(xx-i-0.5, yy-j-0.5)#Theta matrix

            T = np.arctan2(xx-i-0.5, yy-j-0.5)#Theta matrix
            xpos = np.array((D)/np.sqrt(2.0),'i').ravel()
            ypos = np.array(len(M)*((T)/2.0/np.pi+0.5),'i').ravel()
            N[[xpos],[ypos]] += M[j,i]
    return N

M = 0.01*np.random.rand(100,100)
for i in range(100):
    for j in range(100):
        M[i,j] += (1+np.sign(20-(i-50)**2-(j-50)**2))

N = corrFunction(M)
dispMatrices(M,N)
