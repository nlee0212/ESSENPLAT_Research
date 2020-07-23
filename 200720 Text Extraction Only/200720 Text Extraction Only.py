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
    #cv2.imshow('mg', grad)
    cv2.imwrite('mg' + img_name + '.' + img_full[1], grad)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow('t', bw)
    cv2.imwrite('t' + img_name + '.' + img_full[1], bw)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('mc', connected)
    cv2.imwrite('mc' + img_name + '.' + img_full[1], connected)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    crop_filename_list = list()
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
            #cv2.rectangle(clear_small, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
            cropped = clear_small[y:y + h, x:x + w]
            crop_filename = 'crop' + str(idx) + img_name + '.' + img_full[1]
            cv2.imwrite(crop_filename, cropped)
            crop_filename_list.append(crop_filename)
    crop_filename_list.reverse()
    # show image with contours rect
    #cv2.imshow('rects', clear_small)
    #cv2.waitKey()

    text_extract(crop_filename_list, img_full)

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
    img_blur = cv2.GaussianBlur(img,(3,3),0)
    cv2.imwrite('blur'+filename,img_blur)
    img_dil = cv2.dilate(img,(3,3),iterations=1)
    """cv2.imshow('dil',img_dil)
    cv2.waitKey()"""
    img_thresh = cv2.adaptiveThreshold(img_dil, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 21, 5)
    """cv2.imshow('thresh', img_thresh)
    cv2.waitKey()"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_cl = clahe.apply(img_thresh)
    """cv2.imshow('CLAHE', img_cl)
    cv2.waitKey()
    cv2.destroyAllWindows()"""
    img_denoise = cv2.fastNlMeansDenoising(img_cl,h=50)
    """cv2.imshow('Denoise', img_denoise)
    cv2.waitKey()
    cv2.destroyAllWindows()"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_denoise,-1,kernel)
    #cv2.imshow('sharp', img_sharp)
    cv2.imwrite('final'+filename,img_sharp)
    """cv2.waitKey()
    cv2.destroyAllWindows()"""

    image_crop(img,img_sharp,img_full)

def text_extract(file_list, filename_list):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Program Files/VisionAPI/rosy-clover-282218-a95092db74bf.json"
    client = vision.ImageAnnotatorClient()
    modified_text = str()
    for path in file_list:
        print(path)
        with io.open(path, 'rb') as image_file:
            content = image_file.read()
        image = vision.types.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        print('Modified Texts:')
        try:
            print(texts[0].description)
            modified_text += ' '+texts[0].description
        except IndexError:
            continue

    with io.open(filename_list[0]+'.'+filename_list[1],'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Original Texts:')
    try:
        print(texts[0].description)
        ori_text = texts[0].description
    except IndexError:
        pass

    filename = filename_list[0]+'_vision_'+'.txt'
    with open(filename, "w", encoding='utf-8') as fp:
        fp.write("원본:\n")
        fp.write(ori_text+"\n")
        fp.write("\n전처리후:\n")
        fp.write(modified_text)

path = "./"
file_list = os.listdir(path)
file_list_image = [file for file in file_list if file.endswith(".jpg")]
file_list_png = [file for file in file_list if file.endswith(".PNG")]
file_list_image += file_list_png

for image in file_list_image:
    image_wash(image)
