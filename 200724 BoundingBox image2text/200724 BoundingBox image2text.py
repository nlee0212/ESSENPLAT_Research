import cv2
import numpy as np
import matplotlib as plt
import os
import io
import re
from google.cloud import vision
from google.cloud.vision import types
import json
from math import floor, ceil

def image_crop(ori_img,img,img_full,data,h):
    """
    :param ori_img: original image without pre-processing
    :param img: image done with pre-processing
    :param img_full: list, ["image file name without extensions","file extension"]
    :param data: list of bounding box data ["boundingBox":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]],...]
    :param h: image height
    :return: -
    """
    img_name = img_full[0]
    crop_filename_list = list()
    rect_img = ori_img.copy()
    for i in range(len(data)):
        box = data[i]["boundingBox"]
        wl = min(floor(box[0][0]),floor(box[2][0]))
        if wl<0:
            wl=0
        wr = max(ceil(box[0][0]),ceil(box[2][0]))
        if wr<0:
            wr=0
        hl = min(floor(box[0][1]), floor(box[2][1]))
        if hl<0:
            hl=0
        hr = max(ceil(box[0][1]), ceil(box[2][1]))
        if hr<0:
            hr=0
        rect_img = cv2.rectangle(rect_img,(wl,hl),(wr,hr),color = (0,255,0),thickness=2)
        cropped = img[hl:hr, wl:wr]
        print(box)
        #cv2.imshow('crop',cropped)
        crop_filename = 'crop' + str(i) + img_name + '.' + img_full[1]
        cv2.imwrite(crop_filename, cropped)
        crop_filename_list.append(crop_filename)
    #cv2.imshow(img_full[0],rect_img)
    cv2.imwrite('rect'+img_name+'.'+img_full[1],rect_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    text_extract(crop_filename_list, img_full)

def image_wash(filename):

    img_full = filename.split('.')
    img_name = img_full[0]

    img = cv2.imread("./boundingbox_image/"+filename,0)

    #img_thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    #cv2.imshow('thresh', img_thresh)
    #cv2.waitKey()
    #img_denoise = cv2.fastNlMeansDenoising(img, h=5)
    """cv2.imshow('Denoise', img_denoise)
    cv2.waitKey()
    cv2.destroyAllWindows()"""

    """cv2.imshow('CLAHE', img_cl)
    cv2.waitKey()
    cv2.destroyAllWindows()"""

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img,-1,kernel)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
    img_cl = clahe.apply(img_sharp)

    #cv2.imshow('sharp', img_sharp)
    cv2.imwrite('final'+filename,img_cl)
    #test_text('final'+filename)
    """cv2.waitKey()
    cv2.destroyAllWindows()"""

    return img, img_cl, img_full

def test_text(filename):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Program Files/VisionAPI/rosy-clover-282218-a95092db74bf.json"
    client = vision.ImageAnnotatorClient()
    modified_text = str()

    print(filename)
    with io.open(filename, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Modified Texts:')
    try:
        print(texts[0].description)
        modified_text += ' ' + texts[0].description.strip()
    except IndexError:
        pass
    filename += ".txt"
    with open(filename, "w", encoding='utf-8') as fp:
        fp.write("전처리 & without crop:\n")
        fp.write(modified_text)


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
            modified_text += ' '+texts[0].description.strip()
        except IndexError:
            continue

    with io.open("./boundingbox_image/"+filename_list[0]+'.'+filename_list[1],'rb') as image_file:
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
        fp.write("\n전처리 & crop 후:\n")
        fp.write(modified_text)

path = "C:/Users/user/PycharmProjects/ESSENPLAT_Research/200724 BoundingBox image2text/boundingbox_image"
file_list = os.listdir(path)
file_list_image = [file for file in file_list if file.endswith(".jpg")]
file_list_png = [file for file in file_list if file.endswith(".PNG") or file.endswith(".png")]
file_list_image += file_list_png

for image in file_list_image:
    print(image)
    filename = image.split(".")[0]+".txt"
    with open("./boundingbox_image/"+filename, 'rt', encoding="UTF8") as json_file:
        json_data = json.load(json_file)

    img, img_sharp, img_full = image_wash(image)
    height = json_data["meta"]["img_size"][0]["height"]
    image_crop(img,img_sharp,img_full,json_data["words"],height)
