    img_thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    img_cl = clahe.apply(img_thresh)
 kernel = np.array([[0, -1, 0],
                       [-1, 10, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_cl,-1,kernel)