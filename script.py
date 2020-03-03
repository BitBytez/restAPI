from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import cv2
import os
import numpy as np


def fun2(path):
    img = cv2.imread(path)
    (H, W) = img.shape[:2]
    # Reading Labels
    labels = []
    with open('./cfg/labels.txt', 'r') as file:
        text = file.read()
        labels = text.strip().split('\n')

    net = cv2.dnn.readNetFromDarknet('./cfg/config.cfg', './cfg/yolov3.weights')
    out_layers = [net.getLayerNames()[i[0]-1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
    net.setInput(blob)

    out_val = net.forward(out_layers)
    boxes, confidences, classIds = [], [], []

    for output in out_val:
        for detection in output:
            scores = detection[5:]
            # print(detection)
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (cX, cY, width, height) = box.astype("int")
                x = int(cX - (width/2))
                y = int(cY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIds.append(classId)

    nms = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    count = 0
    newCd = []
    if len(nms) > 0:
        for i in nms.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(img, (x, y), (x+w, y+h), colors[count % 3], 1)
            newCd.append(classIds[i])
            count += 1
    outs = {
        'human': 0,
        'animals': 0,
        'objects': 0
    }
    animals = ['bird', 'cat', 'dog', 'horse', 'sheep','cow', 'elephant', 'bear', 'zebra', 'giraffe', ]
    for q in newCd:
        if labels[q] in animals:
            outs['animals'] += 1
        elif labels[q] == 'person':
            outs['human'] += 1
        else:
            outs['objects'] += 1
        print(labels[q])

    return outs


app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'Message':'Welcome To My World'}
    
class ImageUpload(Resource):
    def post(self):
        files = request.files.getlist('image')
        for file in files:
            if file.filename == '':
                return {'Message':'No file uploaded'}, 400
            file.save(os.path.join('./imgs/', file.filename))
        output = fun2(os.path.join('./imgs/', file.filename))
        return output, 201

class VideoUpload(Resource):
    def post(self):
        file = request.files
        if 'video' not in file:
            return {'Message':'No video key in the request'}, 400
        if file.filename == '':
            return {'Message':'No file uploaded'}, 400
        file.save(os.path.join('./vids/', file.filename))
        return {'Message':'Video Uploaded Successful'}, 201

api.add_resource(HelloWorld,'/')
api.add_resource(ImageUpload, '/image')
api.add_resource(VideoUpload, '/video')

if __name__ == '__main__':
    app.run(debug=True)
