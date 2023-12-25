# Final code
# S K ABEYWEERA

# https://www.computervision.zone/courses/object-detection-course/
# https://docs.ultralytics.com/models/yolov8/#supported-modes
# Importing required libraries
from vosk import Model, KaldiRecognizer
import pyaudio
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from ClassNames import *
from t2s import *

# Initialize variables
a1 = 0
bit = 0
bit2 = 0
cls1 = 0
orion = 7
obj_name = []
obj_or = []
CX1 = []
CY1 = []
CX2 = []
CY2 = []
val = 0
OBJorORi = 0
ij = 0

# Initialize voice-related variables
texts = []
# https://alphacephei.com/vosk/models
model = Model(r"D:\semester 7\ME420\Object-Detection-101\Project_final_yr\vosk-model-en-us-daanzu-20200905-lgraph")
recognizer = KaldiRecognizer(model, 16000)
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

# Main loop
while True:
    # Voice recognition loop
    while True:
        data = stream.read(4096, exception_on_overflow=False)

        if recognizer.AcceptWaveform(data):
            texts = recognizer.Result()

            # Bypass microphone for testing
            # val = input("Enter your value: ")
            # break

            # Print recognized text
            print(texts[14:-3])

            # Check recognized text against predefined values
            for i in range(len(ori)):
                if texts[14:-3] == ori[i]:
                    print('okay')
                    t2s("okay")
                    val = ori[i]
                    bit = 1
                    break

            if texts[14:-3] == 'close':
                print('finished')
                t2s("finished")
                bit2 = 1
                break

            for j in range(len(classNames)):
                if texts[14:-3] == classNames[j]:
                    print('okay')
                    t2s("okay")
                    val = classNames[j]
                    bit = 1
                    OBJorORi = 1
                    break

        if bit == 1 or bit2 == 1:
            bit = 0
            break

    if bit2 == 1:
        bit2 = 0
        break

    # Object detection loop
    cap = cv2.VideoCapture(0)  # MAIN CAMERA VIEW
    cap.set(3, 640)
    cap.set(4, 480)

    cap1 = cv2.VideoCapture(2)  # SECONDARY CAMERA VIEW
    cap1.set(3, 640)
    cap1.set(4, 480)

    model = YOLO("../Yolo-Weights/yolov8l.pt")

    prev_frame_time = 0
    new_frame_time = 0

    # Grid limits
    limits1 = [0, 160, 640, 160]
    limits2 = [0, 320, 640, 320]
    limits3 = [210, 0, 210, 480]
    limits4 = [430, 0, 430, 480]
    limits5 = [310, 240, 330, 240]
    limits6 = [320, 230, 320, 250]

    while True:
        for n1 in range(99999):
            new_frame_time = time.time()
            success, img = cap.read()
            results = model(img, stream=True)

            d = 0

            for r in results:
                boxes = r.boxes
                obj_name = []
                obj_or = []
                CX1 = []
                CY1 = []

                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])
                    cls1 = int(box.cls[0])
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                    # Determine object orientation
                    if cx < 210 and cy < 160:
                        orion = 0
                    if 210 < cx < 430 and cy < 160:
                        orion = 1
                    if cx > 430 and cy < 160:
                        orion = 2
                    if cx < 210 and 160 < cy < 320:
                        orion = 3
                    if 210 < cx < 430 and 160 < cy < 320:
                        orion = 4
                    if cx > 430 and 160 < cy < 320:
                        orion = 5
                    if cx < 210 and cy > 320:
                        orion = 6
                    if 210 < cx < 430 and cy > 320:
                        orion = 7
                    if cx > 430 and cy > 320:
                        orion = 8

                    # Display class name, object orientation, and coordinates on the image
                    cvzone.putTextRect(img, f'{classNames[cls]} {cx} {ori[orion]}', (max(0, x1), max(35, y1)),
                                       scale=1, thickness=1)

                    obj_name.append(classNames[cls])
                    obj_or.append(ori[orion])
                    CX1.append(cx)
                    CY1.append(cy)

                # Grid lines
                cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 1)
                cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 1)
                cv2.line(img, (limits3[0], limits3[1]), (limits3[2], limits3[3]), (0, 0, 255), 1)
                cv2.line(img, (limits4[0], limits4[1]), (limits4[2], limits4[3]), (0, 0, 255), 1)
                cv2.line(img, (limits5[0], limits5[1]), (limits5[2], limits5[3]), (0, 0, 255), 1)
                cv2.line(img, (limits6[0], limits6[1]), (limits6[2], limits6[3]), (0, 0, 255), 1)

                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                cv2.imshow("Image", img)
                cv2.waitKey(1)

            new_frame_time = time.time()
            success1, img1 = cap1.read()
            results1 = model(img1, stream=True)

            for r in results1:
                boxes = r.boxes
                obj_name1 = []
                obj_or1 = []
                CX2 = []
                CY2 = []

                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img1, (x1, y1, w, h))
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])
                    cls1 = int(box.cls[0])
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img1, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                    # Determine object orientation
                    if cx < 210 and cy < 160:
                        orion = 0
                    if 210 < cx < 430 and cy < 160:
                        orion = 1
                    if cx > 430 and cy < 160:
                        orion = 2
                    if cx < 210 and 160 < cy < 320:
                        orion = 3
                    if 210 < cx < 430 and 160 < cy < 320:
                        orion = 4
                    if cx > 430 and 160 < cy < 320:
                        orion = 5
                    if cx < 210 and cy > 320:
                        orion = 6
                    if 210 < cx < 430 and cy > 320:
                        orion = 7
                    if cx > 430 and cy > 320:
                        orion = 8

                    # Display class name, object orientation, and coordinates on the image
                    cvzone.putTextRect(img1, f'{classNames[cls]} {cx} {ori[orion]}', (max(0, x1), max(35, y1)),
                                       scale=1, thickness=1)

                    obj_name1.append(classNames[cls])
                    obj_or1.append(ori[orion])
                    CX2.append(cx)
                    CY2.append(cy)

                # Grid lines
                cv2.line(img1, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 1)
                cv2.line(img1, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 1)
                cv2.line(img1, (limits3[0], limits3[1]), (limits3[2], limits3[3]), (0, 0, 255), 1)
                cv2.line(img1, (limits4[0], limits4[1]), (limits4[2], limits4[3]), (0, 0, 255), 1)
                cv2.line(img1, (limits5[0], limits5[1]), (limits5[2], limits5[3]), (0, 0, 255), 1)
                cv2.line(img1, (limits6[0], limits6[1]), (limits6[2], limits6[3]), (0, 0, 255), 1)

                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                cv2.imshow("Image1", img1)
                cv2.waitKey(1)

        # Uncomment the following block to print debug information
        '''
        print(obj_name)
        print(obj_or)
        print(obj_name1)
        print(obj_or1)
        print(CX1)
        print(CX2)
        # val = input("Enter your value: ")
        # print(val)
        '''

        # Check if the recognized value matches an object's orientation or class
        a1 = 0
        for i in range(len(obj_or)):
            if obj_or[i] == val:
                a1 = 1
                ij = i
                print(obj_name[i])
                t2s(obj_name[i])
                # print(CX1[i])
                # print(CX2[i])

        for i in range(len(obj_name)):
            if obj_name[i] == val:
                a1 = 1
                ij = i
                print(obj_or[i])
                t2s(obj_or[i])
                # print(CX1[i])
                # print(CX2[i])

        # If the recognized value doesn't match any object's orientation or class
        if a1 == 0:
            print("nothing")
            t2s("nothing")
            break

        if 0 <= ij < len(CX1) and 0 <= ij < len(CX2):
            # Calculate the distance between two points
            Dis = dis(CX1[ij], CX2[ij])

            if Dis > 1000:
                print("in")
                print(round((Dis / 1000), 2))
                print("meters")
                t2s("in")
                t2s(round((Dis / 1000), 2))
                t2s("meters")
            else:
                print("in")
                print(round(Dis, 2))
                print("centimeters")
                t2s("in")
                t2s(round(Dis, 2))
                t2s("centimeters")

        OBJorORi = 0

        break  # End of the main loop
