from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications import imagenet_utils
import numpy as np

# 1. 모델 구성하기
model = VGG16(weights='imagenet')

# 2. 모델 사용하기

# 임의의 이미지 불러오기
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
xhat = image.img_to_array(img)
xhat = np.expand_dims(xhat, axis=0)
xhat = preprocess_input(xhat)

# 임의의 이미지로 분류 예측하기
yhat = model.predict(xhat)

# 예측 결과 확인하기
P = imagenet_utils.decode_predictions(yhat)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))