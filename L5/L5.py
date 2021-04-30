from timeit import default_timer as timer
import cv2
import dlib


def convert_and_trim_bb(image, rect):
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    w = endX - startX
    h = endY - startY
    return startX, startY, w, h


def haar_cascade_detect(image, save_name):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    time = timer()

    haar_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    rects = haar_detector.detectMultiScale(image_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE)

    time = timer() - time
    img_haar = IMAGE.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(img_haar, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("img_haar", img_haar)
    cv2.imwrite(save_name, img_haar)
    cv2.waitKey(1)

    return time


def HOG_SVM_detect(image, save_name):
    IMAGE_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    time = timer()

    svm_detector = dlib.get_frontal_face_detector()
    hog_svm_rects = svm_detector(IMAGE_RGB, 1)
    hog_svm_boxes = [convert_and_trim_bb(image, r) for r in hog_svm_rects]

    time = timer() - time
    img_hog_svm = IMAGE.copy()
    for (x, y, w, h) in hog_svm_boxes:
        cv2.rectangle(img_hog_svm, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("img_hog_svm", img_hog_svm)
    cv2.imwrite(save_name, img_hog_svm)
    cv2.waitKey(1)

    return time


def CNN_network_detect(image, save_name):
    IMAGE_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    time = timer()

    cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    cnn_rects = cnn_detector(IMAGE_RGB, 1)
    cnn_boxes = [convert_and_trim_bb(image, r.rect) for r in cnn_rects]

    time = timer() - time
    img_cnn = IMAGE.copy()
    for (x, y, w, h) in cnn_boxes:
        cv2.rectangle(img_cnn, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("img_cnn", img_cnn)
    cv2.imwrite(save_name, img_cnn)
    cv2.waitKey(1)

    return time


IMAGES = [
    ("2_Demonstration_Demonstration_Or_Protest_2_1.jpg", cv2.imread("2_Demonstration_Demonstration_Or_Protest_2_1.jpg")),
    ("18_Concerts_Concerts_18_127.jpg", cv2.imread("18_Concerts_Concerts_18_127.jpg")),
    ("26_Soldier_Drilling_Soldiers_Drilling_26_73.jpg", cv2.imread("26_Soldier_Drilling_Soldiers_Drilling_26_73.jpg"))
]

print(f"{'name':50}    {'Haar':8}    {'SVM+HOG':8}    {'CNN':8}")
for i in range(len(IMAGES)):
    IMAGE_NAME = IMAGES[i][0]
    IMAGE = IMAGES[i][1]

    print(f"{IMAGE_NAME:50}    ", end="")
    time = haar_cascade_detect(IMAGE, f"out/img{i}_haar.jpg")
    print(f"{time:.6f}    ", end="")
    time = HOG_SVM_detect(IMAGE, f"out/img{i}_hos_svm.jpg")
    print(f"{time:.6f}    ", end="")
    time = CNN_network_detect(IMAGE, f"out/img{i}_cnn.jpg")
    print(f"{time:.6f}    ")

cv2.waitKey(0)
cv2.destroyAllWindows()
