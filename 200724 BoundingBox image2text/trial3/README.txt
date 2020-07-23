img_thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
img_denoise = cv2.fastNlMeansDenoising(img_thresh, h=50)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
kernel = np.array([[0, 0, 0],
                       [0, 5, 0],
                       [0, 0, 0]])
    img_sharp = cv2.filter2D(img_denoise,-1,kernel)