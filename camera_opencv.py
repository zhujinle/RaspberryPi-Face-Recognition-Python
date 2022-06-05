import cv2
import time
from PIL import Image, ImageDraw

face_model=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def getImage(imgQueue):
    camera = cv2.VideoCapture(0)
    ret,img = camera.read()
    backGround = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    while(True):
        # 读取图片
        read_start = time.time()
        ret,img = camera.read()
        read_end = time.time()
        change = False
        if(not ret):
            continue
        # 移动目标检测
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(backGround,gray)
        diff = cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
        diff = cv2.dilate(diff,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,4)),iterations=2)
        contours,hierarchy = cv2.findContours(diff,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        show_img = img.copy()
        for c in contours:
            if(cv2.contourArea(c)<1500):
                continue
            change = True
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(show_img,(x,y),(x+w,y+h),(0,255,0),2)
        if(change==False):
            backGround = gray
        move_detection_end = time.time()

        # 人脸检测
        if(change==True):
            # 检查人脸
            faces = face_model.detectMultiScale(gray, 1.1, 7, 0,(20,20))
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 标记人脸
            for (x, y, w, h) in faces:
                # 矩形标记
                cv2.rectangle(show_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
                cv2.putText(show_img, 'face', (int(x + w / 2 - 60), y), font, 1, (255, 255, 255), 2)
        
        # 显示信息
        cv2.putText(show_img, 'shape:%s*%s'%(img.shape[0],img.shape[1]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(show_img, 'move:%sms'%(round((move_detection_end-read_end)*1000,3)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 将图片编码为web所用格式
        frame = cv2.imencode('.jpg',show_img)[1].tobytes()
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')   

