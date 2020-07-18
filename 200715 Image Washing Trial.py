import cv2
import numpy as np
import matplotlib as plt
import os
import io
import re
from google.cloud import vision
from google.cloud.vision import types

def image_wash(filename):
    # get image name
    img_full = filename.split('.')
    img_name = img_full[0]
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

    """#RGB -> Gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray' + img_name+'.'+img_full[1], gray_img)
    cv2.imshow('gray', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    """#MorphGradient
    morph_grad_img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,(2,2))
    cv2.imwrite('mg' + img_name+'.'+img_full[1], morph_grad_img)
    cv2.imshow('mg', morph_grad_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    # Adaptive Threshold
    adap_thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 10)
    cv2.imwrite('at' + img_name + '.' + img_full[1], adap_thresh_img)
    cv2.imshow('at', adap_thresh_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #MorphClose
    morph_close_img = cv2.morphologyEx(adap_thresh_img,cv2.MORPH_CLOSE,(20,5))
    cv2.imwrite('mc' + img_name + '.' + img_full[1], morph_close_img)
    cv2.imshow('mc', morph_close_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, _ = cv2.findContours(morph_close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(adap_thresh_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        if h<10 or w<40:
            continue
        cv2.imshow('cnt',rect)
        cv2.waitKey(0)

        # Cropping the text block for giving input to OCR
        cropped = adap_thresh_img[y:y + h, x:x + w]
        cv2.imwrite('crop' + img_name + '.' + img_full[1], cropped)
        text_extract('crop' + img_name + '.' + img_full[1])

def text_extract(path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Program Files/VisionAPI/rosy-clover-282218-a95092db74bf.json"

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')
    try:
        print(texts[0].description)
    except IndexError:
        return

    sep_lang(texts[0].description, path, '_vision_')


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

    fn = filename.split('.')

    filename = fn[0] + version + ".txt"

    with open(filename, "w", encoding='utf-8') as fp:
        fp.write("한글:\n")
        fp.write(result_hangul + "\n\n")
        fp.write("숫자:\n")
        for item in num_arr:
            fp.write(item + "\n")
        fp.write("\n\n수식:\n")
        for item in equ_arr:
            fp.write(item + "\n")
        fp.write("\n\n기호:\n")
        for item in ch_arr:
            fp.write(item + "\n")

path = "./"
file_list = os.listdir(path)
file_list_image = [file for file in file_list if file.endswith(".jpg")]
file_list_png = [file for file in file_list if file.endswith(".PNG")]
file_list_image += file_list_png

for image in file_list_image:
    image_wash(image)
