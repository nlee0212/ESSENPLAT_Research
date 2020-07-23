    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_cl,-1,kernel)