from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import os

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
        return {'Message':'Images Uploaded Successful'}, 201

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