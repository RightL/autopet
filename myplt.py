import matplotlib.pyplot
from monai.visualize import matshow3d
def ps(i):
    matplotlib.pyplot.imshow(i,cmap='gray')
    matplotlib.pyplot.show()
def m3(i):
    matshow3d(i)
    matplotlib.pyplot.show()