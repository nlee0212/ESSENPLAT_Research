ret, img_thresh = cv2.threshold(img,20,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
img_denoise = cv2.fastNlMeansDenoising(img_cl,h=20)
kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_denoise,-1,kernel)