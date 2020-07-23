    img_thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    img_denoise = cv2.fastNlMeansDenoising(img_thresh, h=20)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    kernel = np.array([[0, -1, 0],
                       [-1, 10, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_cl,-1,kernel)
