import cv2
import numpy as np
import os
def faceDectector_copy(img):
    face_cascade1 = cv2.CascadeClassifier('./xmls/prof.xml')
    face_cascade2 = cv2.CascadeClassifier('./xmls/face_alt2.xml')
    # img = cv2.imread('./imgs/group.jpeg')
    # print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detected_faces1 = face_cascade1.detectMultiScale(gray)
    detected_faces2 = face_cascade2.detectMultiScale(gray)
    # for (column, row, width, height) in detected_faces1:
    #     cv2.rectangle(
    #         img,
    #         (column, row),
    #         (column + width, row + height),
    #         (0, 255, 0),
    #         2
    #     )
    count = 0
    os.system('rm ' + './collage_pics' + '/*')
    for (column, row, width, height) in detected_faces2:
        count += 1
        cv2.rectangle(
            img,
            (column - 10, row - 10),
            (column + width + 10, row + height + 10),
            (255, 0, 0),
            2
        )
        filename = './collage_pics/img' + str(count) + '.jpg'
        cv2.imwrite(filename, img[row - 10: row + height + 10, column - 10:column + width + 10])        
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.resizeWindow('Image', 600, 600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('./imgs/indian_cricket.jpg')
faceDectector_copy(img)
