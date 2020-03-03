import cv2
face_cascade1 = cv2.CascadeClassifier('./xmls/prof.xml')
face_cascade2 = cv2.CascadeClassifier('./xmls/face_alt2.xml')
img = cv2.imread('./imgs/group.jpeg')
# print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detected_faces1 = face_cascade1.detectMultiScale(gray)
detected_faces2 = face_cascade2.detectMultiScale(gray)

for (column, row, width, height) in detected_faces1:
    cv2.rectangle(
        img,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )
for (column, row, width, height) in detected_faces2:
    cv2.rectangle(
        img,
        (column, row),
        (column + width, row + height),
        (255, 0, 0),
        2
    )
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow("Image", img)
cv2.resizeWindow('Image', 600, 600)

cv2.waitKey(0)
cv2.destroyAllWindows()
