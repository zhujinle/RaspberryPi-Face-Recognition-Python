from imutils import paths
import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
from multiprocessing import Process,Manager
from numpy import array
import os



def getImageRecognized(imgQueue):
    # 多线程
    # p = Process(target=f, args=())

    camera = cv2.VideoCapture(0)
    ret, img = camera.read()
    # img = BGR_to_RGB(img)
    # img=data_augment(img,20);
    # print(img.dtype)
    ret, img = camera.read()
    camera.release()
    imagePaths = list(paths.list_images(r'C:\Users\10260\Desktop\code\dataset'))
    print(paths)
    # imagePaths = os.listdir('./dataset/')

    knownEncodings = []
    knownNames = []
    camera.release()
    cv2.destroyAllWindows()
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-1]

        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb)

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    # Resize frame of video to 1/4 size for faster face recognition processing
    img1 = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = rgb1[:, :, ::-1]
    # rgb1 = rgb1[:, :, ::-1]
    face_locations = face_recognition.face_locations(img1)
    face_encodings = face_recognition.face_encodings(img1, face_locations)
    pil_image = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_image)
    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(knownEncodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(knownEncodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = knownNames[best_match_index]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left*4, top*4), (right*4, bottom*4)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left*4, bottom*4 - text_height - 10), (right*4, bottom*4)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left*4 + 6, bottom*4 - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw
    # img = pil_image.copy()

    # pil_image.show()
    img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imencode('.jpg', img)[1].tobytes()
    return img
