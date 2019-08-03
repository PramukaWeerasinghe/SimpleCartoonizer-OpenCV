import cv2
import numpy as np
from tkinter import filedialog
from tkinter import *
 

print("Select Image File: ")

root = Tk().withdraw()
filename =  filedialog.askopenfilename(initialdir = "C:/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

gamma_value = eval(input("Enter gamma value:"))
medianBlur_kernelsize = eval(input("Enter medianBlur kernel size:"))


img = cv2.imread(filename,cv2.IMREAD_COLOR)

def gamma_transform(img, gamma):
    I = np.zeros(img.shape, dtype=img.dtype)
    if(len(img.shape) == 3):
        for c in range(img.shape[2]):
            I[:,:,c] = 255 * (img[:,:,c]/255) ** (1/gamma)
    else:
        I = 255 * (img/255) ** (1/gamma)

    return I

img_gamma = gamma_transform(img,gamma_value)
    
img_gray = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)

img_blur = cv2.medianBlur(img_gray, medianBlur_kernelsize)

img_edge =cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY,9,2)
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

img_cartoon = cv2.bitwise_and(img, img_edge)

img_cartoon_blur = cv2.GaussianBlur(img_cartoon,(5,5),0)


cv2.namedWindow('cartoon',cv2.WINDOW_KEEPRATIO)
cv2.imshow('cartoon',np.hstack((img,img_cartoon_blur)))
#cv2.destroyAllWindows()
