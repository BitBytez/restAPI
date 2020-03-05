# REST API 

### Mini Project by Ciphense, Cognitive Intelligence.

### REST API created on retrieving data by Image processing(with openCV) and Image detection(with help of You Only Look Once)

# Installation (Only for Ciphense Team)
### Please make sure you have required modules before proceeding

- Flask
- flask-restful
- openCV
- numpy
- matplotlib

### Clone the repository and download weights from YOLO
```bash
    git clone https://github.com/BitBytez/restAPI

    wget https://pjreddie.com/media/files/yolov3.weights
```
### Paste yolov3.weights in **restAPI**/cfg/

# Usage

## Task 1

Run **restAPI**/server.py
Download a image which contains persons, animals adn few objects in it. 
```bash
    curl -F "image=@/path/to/downloaded/image" 127.0.0.1:5000/getImageDetails
```

## Task 2
Run **restAPI**/server.py
Download a set of images which contains persons
```bash
    curl -F "image=@/path/to/downloaded/image1" -F "image=@/path/to/downloaded/image2" 127.0.0.1:5000/createCollage > myOutputCollage.jpg
```

# Repository Tree
```
├── cfg 
│   ├── config.cfg 
│   └── labels.txt --> Labels which model can detect
├── collage_pics --> Stores collage pics when an image request is received /createCollage
├── detector.py --> Main code where detection and collage creation happens
├── imgs --> Stores images when an image(s) request is received /getImageDetails
├── __pycache__
│   └── detector.cpython-36.pyc
├── README.MD
├── server.py --> restAPI server code
├── test_imgs --> Images used for testing
│   ├── a.jpg
│   ├── b.jpg
│   ├── c.jpeg
│   ├── frnds.jpg
│   ├── image.jpg
│   ├── indian_cricket.jpg
│   ├── mylovers.jpg
│   ├── paul.jpg
│   ├── QWEQPIKNDLSFNKL>.jpeg
│   ├── QWmnN!Ojkm!MmmL!.jpeg
│   └── QWOPSDANMNKKQLW.jpeg
└── xmls --> Haar Cascade xmls used for face detection
    ├── eye.xml
    ├── face_alt2.xml
    ├── face_alt_tree.xml
    ├── face_alt.xml
    ├── face.xml
    └── prof.xml
```