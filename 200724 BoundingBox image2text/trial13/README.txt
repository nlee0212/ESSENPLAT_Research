    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_cl,-1,kernel)