import cv2
import numpy as np
import matplotlib as plt
import os
import io
import re
from google.cloud import vision
from google.cloud.vision import types

def image_crop(ori_img,img,img_full):
    img_name = img_full[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    small = cv2.pyrDown(ori_img)
    clear_small = cv2.pyrDown(img)
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('mg', grad)
    cv2.imwrite('mg' + img_name + '.' + img_full[1], grad)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('t', bw)
    cv2.imwrite('t' + img_name + '.' + img_full[1], bw)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('mc', connected)
    cv2.imwrite('mc' + img_name + '.' + img_full[1], connected)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    for idx in range(len(contours)):
        height, width = bw.shape
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        # 가장자리의 글씨들은 주된 문제가 아닐 것이기 때문에 제외.
        if x + w == width or x == 0 or y + h == height or y == 0:
            continue
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
        if r > 0.3 and w > 10 and h > 10:
            cv2.rectangle(clear_small, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)

    # show image with contours rect
    cv2.imshow('rects', clear_small)
    cv2.waitKey()

def image_wash(filename):
    """
    1. Image Blurring by Bilateral Filtering
    경계선은 유지하며 전체적으로 밀도가 동일한 노이즈, 화이트 노이즈를 제거해
    경계선이 흐려지지 않고 이미지를 부드럽게 변환
    2. Image Dilation
    :param filename:
    :return:
    """
    img_full = filename.split('.')
    img_name = img_full[0]

    img = cv2.imread(filename,0)
    img_blur = cv2.bilateralFilter(img,10,50,50)
    cv2.imwrite('blur'+filename,img_blur)
    img_dil = cv2.dilate(img_blur,(3,3),iterations=1)
    cv2.imshow('dil',img_dil)
    cv2.waitKey()
    img_thresh = cv2.adaptiveThreshold(img_dil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 5)
    cv2.imshow('thresh', img_thresh)
    cv2.waitKey()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img_thresh)
    cv2.imshow('img', img_cl)
    cv2.waitKey()
    cv2.destroyAllWindows()

    image_crop(img,img_cl,img_full)

path = "./"
file_list = os.listdir(path)
file_list_image = [file for file in file_list if file.endswith(".jpg")]
file_list_png = [file for file in file_list if file.endswith(".PNG")]
file_list_image += file_list_png

for image in file_list_image:
    image_wash(image)
