from flask import Flask,render_template, request,Response
from multiprocessing import Process,Manager
from camera_opencv import getImage
from facer import getImageRecognized
import base64

import cv2
app = Flask(__name__)

@app.route('/')
def index():
    data = {'fps':20}
    return render_template('index.html',data=data)

@app.route('/video_feed')
def video_feed():
    return Response(getImage(None),mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/recognized_feed')
def recognized_feed():
        return render_template('new_picture.html')
@app.route('/img_stream')
def img_stream():
        test = getImageRecognized(None)
        img_stream = base64.b64encode(test).decode()
        return Response(test,mimetype='image/jpeg')

if __name__ == '__main__':

    app.run(host="0.0.0.0",port="5002",threaded=True,debug=True)