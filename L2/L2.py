#%% Imports

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% Functions
def cv_imshow(img, img_title="image"):
    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img / 255
    else:
        img_ = img
    cv2.imshow(img_title, img_)
    cv2.waitKey(1)


def cv_imwrite(path, name, image):
    if not os.path.isdir(path):
        os.makedirs(path)
    cv2.imwrite(path + "/" + name, image)


def calcPSNR(img1, img2):
    imax = 255.**2
    mse = ((img1.astype(np.float64)-img2)**2).sum()/img1.size
    return 10.0*np.log10(imax/mse)


def printi(img, img_title="image"):
    print(f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}")


#%% Resources
BASE_DIR = ""
SAVE_DIR = "out/"

IMAGE = cv2.imread(BASE_DIR + "parrots_col.png", cv2.IMREAD_UNCHANGED)
IMAGE_NOISE = cv2.imread(BASE_DIR + "parrots_col_noise.png", cv2.IMREAD_UNCHANGED)
IMAGE_INOISE = cv2.imread(BASE_DIR + "parrots_col_inoise.png", cv2.IMREAD_UNCHANGED)

NOISE = [IMAGE_NOISE, IMAGE_INOISE]
NOISE_DESC = ["Szum Gaussa", "Szum Impulsowy"]

#%% Zad 1
for j in range(len(NOISE)):
    print(f"{NOISE_DESC[j]}: Filtr Gaussa")
    for i in [3, 5, 7]:
        gblur_img = cv2.GaussianBlur(NOISE[j], (i, i), 0)
        cv_imwrite(SAVE_DIR + "Z1", f"szum{j}_gauss{i}x{i}.png", gblur_img)
        print(f"Maska {i}x{i}: PSNR = {calcPSNR(IMAGE, gblur_img):.4f}")
    print(f"\n{NOISE_DESC[j]}: Filtr Medianowy")
    for i in [3, 5, 7]:
        median_img = cv2.medianBlur(NOISE[j], i)
        cv_imwrite(SAVE_DIR + "Z1", f"szum{j}_median{i}x{i}.png", median_img)
        print(f"Maska {i}x{i}: PSNR = {calcPSNR(IMAGE, median_img):.4f}")
    print()

#%% Zad 2
IMAGE_CONVERTED = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2YCrCb)

NEW_IMAGE = np.zeros(IMAGE_CONVERTED.shape, dtype=IMAGE_CONVERTED.dtype)

for i in range(3):
    plt.hist(IMAGE_CONVERTED[:, :, i].flatten(), 256)
    plt.title(f"Histogram {i}-tej składowej")
    plt.savefig(SAVE_DIR + f"Z2/hist_{i}.png")
    plt.show()

NEW_IMAGE[:, :, 0] = cv2.equalizeHist(IMAGE_CONVERTED[:, :, 0])
NEW_IMAGE[:, :, 1] = IMAGE_CONVERTED[:, :, 1]
NEW_IMAGE[:, :, 2] = IMAGE_CONVERTED[:, :, 2]

for i in range(3):
    plt.hist(NEW_IMAGE[:, :, i].flatten(), 256)
    plt.title(f"Histogram {i}-tej składowej po wyrównaniu")
    plt.savefig(SAVE_DIR + f"Z2/conv_hist_{i}.png")
    plt.show()


NEW_IMAGE = cv2.cvtColor(NEW_IMAGE, cv2.COLOR_YCrCb2BGR)

cv_imshow(IMAGE, "Obraz bazowy")
cv_imwrite(SAVE_DIR + "Z2", f"Bazowy.png", IMAGE)
cv_imshow(NEW_IMAGE, "Histogram")
cv_imwrite(SAVE_DIR + "Z2", "PoNormalizacji.png", NEW_IMAGE)
cv2.waitKey(0)

#%% Zad 3
W = -1.5
IMAGE_LAP = cv2.addWeighted(IMAGE, 1, cv2.Laplacian(IMAGE, cv2.CV_8U), W, 0)
cv_imshow(IMAGE, "Obraz Bazowy")
cv_imwrite(SAVE_DIR + "Z3", "Bazowy.png", IMAGE)
cv_imshow(IMAGE_LAP, "Obraz wyostrzony")
cv_imwrite(SAVE_DIR + "Z3", f"ObrazWyostrzony{W}.png", IMAGE_LAP)
cv2.waitKey(0)
