import cv2
import numpy as np
import matplotlib as plt
import os
import io
import re
from google.cloud import vision
from google.cloud.vision import types

def image_wash(filename):
    img_full = filename.split('.')
    img_name = img_full[0]

    large = cv2.imread(filename)
    rgb = cv2.pyrDown(large)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
            cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)

    # show image with contours rect
    cv2.imshow('rects', rgb)
    cv2.waitKey()

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
