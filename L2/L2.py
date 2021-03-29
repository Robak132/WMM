import cv2
import numpy as np


#%% Functions
def cv_imshow(img, img_title="image"):
    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img / 255
    else:
        img_ = img
    cv2.imshow(img_title, img_)
    cv2.waitKey(1)


def calcPSNR(img1, img2):
    imax = 255.**2
    mse = ((img1.astype(np.float64)-img2)**2).sum()/img1.size
    return 10.0*np.log10(imax/mse)


def printi(img, img_title="image"):
    print(f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}")


#%% Resources
image = cv2.imread("parrots_col.png", cv2.IMREAD_UNCHANGED)
image_noise = cv2.imread("parrots_col_noise.png", cv2.IMREAD_UNCHANGED)
image_inoise = cv2.imread("parrots_col_inoise.png", cv2.IMREAD_UNCHANGED)

#%% Zad 1
for img_noise in [image_noise, image_inoise]:
    print("\nFiltr Gaussa")
    for i in [3, 5, 7]:
        gblur_img = cv2.GaussianBlur(img_noise, (i, i), 0)
        print(f"Maska {i}x{i}: PSNR = {calcPSNR(image, gblur_img)}")
    print("\nFiltr Medianowy")
    for i in [3, 5, 7]:
        median_img = cv2.medianBlur(img_noise, i)
        print(f"Maska {i}x{i}: PSNR = {calcPSNR(image, gblur_img)}")

#%% Zad 2

""" Here be dragons """

#%% Zad 3

""" Here be dragons """
