import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
def objectDetector( imagePath, 
                    labelsPath='./cfg/labels.txt', 
                    configPath='./cfg/config.cfg', 
                    weightsPath = './cfg/yolov3.weights',
                    confi = 0.5,
                    thresh = 0.8):
    img = cv2.imread(imagePath)
    (H,W) = img.shape[:2]
    labels = []
    with open(labelsPath, 'r') as file:
        labels = file.read().strip().split('\n')
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    out_layers = [net.getLayerNames()[i[0]-1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
    net.setInput(blob)
    out_val = net.forward(out_layers)
    boxes, confidences, classIds = [], [], []
    for output in out_val:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confi:
                box = detection[0:4] * np.array([W,H,W,H])
                (cX, cY, width, height) = box.astype("int")
                x = int(cX - (width/2))
                y = int(cY - (height/2))
                boxes.append([x,y,int(width), int(height)])
                confidences.append(float(confidence))
                classIds.append(classId)
    nms = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
    nms_boxes = []
    if len(nms) > 0:
        for i in nms.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
            nms_boxes.append(classIds[i])
    final_classes = [labels[i] for i in nms_boxes]
    return final_classes, img


def faceDetector(imgPath,
                xmlPath='./xmls/face_alt2.xml', 
                scaleFactor = 1.18 , 
                minNeighbors = 3):
    img = cv2.imread(imgPath)
    face_cascade = cv2.CascadeClassifier(xmlPath)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(grey, scaleFactor = scaleFactor, minNeighbors = minNeighbors)
    for (column, row, width, height) in detected_faces:
        cv2.rectangle(
        img,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )
    return img, detected_faces
npics = 0
def createCollage(directoryPath = './collage_pics/'):
    for root, dirs, files in os.walk(directoryPath):
        nPics = len(files)

    if nPics > 0:
        for i in range(1, nPics + 1):
            img = plt.imread('img'+str(i)+'.jpg')
            
        
# img, detected_faces = faceDetector(sys.argv[1])
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("image", 600,600)
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

createCollage()