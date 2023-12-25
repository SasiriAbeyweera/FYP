# Lists and functions

directory = r"C:\Users\sasir\Desktop\proj_out"

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

ori = ["top left", "up", "top right", "left", "center", "right", "bottom left", "down", "bottom right"]

'''
Webcams
0 - buit-in
1 - iriun 1
2 - iriun 2

Iphone 7
Rear camera focal length	28 mm
'''
# print(len(classNames))

B = 141  # mm
f = 28  # mm
k = 0.1


def dis(x1, x2):
    delta = x1 - x2
    if delta < 0:
        delta = delta * (-1)
    # d = k * B * f / delta
    d = 519600 * (1 / delta) * (1 / delta) + 73980 * (1 / delta) - 17.37
    return d
