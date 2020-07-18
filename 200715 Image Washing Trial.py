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
    cv2.imshow('dil',img_dil)
    cv2.waitKey()
    img_thresh = cv2.adaptiveThreshold(img_dil, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 21, 5)
    cv2.imshow('thresh', img_thresh)
    cv2.waitKey()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_cl = clahe.apply(img_thresh)
    cv2.imshow('CLAHE', img_cl)
    cv2.waitKey()
    cv2.destroyAllWindows()
    img_denoise = cv2.fastNlMeansDenoising(img_cl,h=50)
    cv2.imshow('Denoise', img_denoise)
    cv2.waitKey()
    cv2.destroyAllWindows()
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_denoise,-1,kernel)
    cv2.imshow('sharp', img_sharp)
    cv2.waitKey()
    cv2.destroyAllWindows()

    image_crop(img,img_sharp,img_full)

def text_extract(file_list, filename_list):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Program Files/VisionAPI/rosy-clover-282218-a95092db74bf.json"
    client = vision.ImageAnnotatorClient()
    result_hangul = str()
    num_arr = list()
    ch_arr = list()
    equ_arr = list()
    for path in file_list:
        print(path)
        with io.open(path, 'rb') as image_file:
            content = image_file.read()
        image = vision.types.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        print('Texts:')
        try:
            print(texts[0].description)
        except IndexError:
            continue
        hangul, num, equ, ch = sep_lang(texts[0].description, path, '_vision_')
        result_hangul += str(hangul)
        num_arr+=num
        ch_arr+=ch
        equ_arr+=equ
        filename = filename_list[0]+'_vision_'+'.txt'
        with open(filename, "w", encoding='utf-8') as fp:
            fp.write("한글:\n")
            fp.write(str(result_hangul) + "\n\n")
            fp.write("숫자:\n")
            for item in num_arr:
                fp.write(item + "\n")
            fp.write("\n\n수식:\n")
            for item in equ_arr:
                fp.write(item + "\n")
            fp.write("\n\n기호:\n")
            for item in ch_arr:
                fp.write(item + "\n")

def sep_lang(text, filename, version):
    result_hangul = list()
    result_else = list()
    hangul = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+')  # 위와 동일
    result = hangul.sub('', text)  # 한글과 띄어쓰기를 제외한 모든 부분을 제거
    if result != '':
        result = " ".join(result.split())
        result_hangul = result
    result = hangul.findall(text)  # 정규식에 일치되는 부분을 리스트 형태로 저장
    if len(result) != 0:
        for item in result:
            arr = item.split('\n')
            for word in arr:
                word = word.rstrip('.')
                word = word.rstrip(',')
                if word != '':
                    result_else.append(word)
    print(result_hangul)
    print(result_else)
    num_arr = list()
    equ_arr = list()
    ch_arr = list()
    continue_flag = 0
    for item in result_else:
        continue_flag = 0
        if item[0].isdigit() == True:
            for letter in item:
                if letter.isalpha() == True:
                    equ_arr.append(item)
                    continue_flag = 1
                    break
            if continue_flag == 1:
                continue
            if item.isdigit() == False:
                num = re.findall('\d+', item)
                for n in num:
                    num_arr.append(n)
                for letter in item:
                    if letter.isdigit() == False:
                        ch_arr.append(letter)
                continue
            num_arr.append(item)
        elif item[0].isalpha() == True:
            equ_arr.append(item)
        else:
            for letter in item:
                if letter.isalpha() == True:
                    equ_arr.append(item)
                    continue_flag = 1
                    break
            if continue_flag == 1:
                continue
            num = re.findall('\d+', item)
            if len(num) > 0:
                for n in num:
                    num_arr.append(n)
                continue_flag = 1
            if continue_flag == 1:
                for letter in item:
                    if letter.isdigit() == False:
                        ch_arr.append(letter)
                continue
            ch_arr.append(item)

    return result_hangul, num_arr, equ_arr, ch_arr




path = "./"
file_list = os.listdir(path)
file_list_image = [file for file in file_list if file.endswith(".jpg")]
file_list_png = [file for file in file_list if file.endswith(".PNG")]
file_list_image += file_list_png

for image in file_list_image:
    image_wash(image)
