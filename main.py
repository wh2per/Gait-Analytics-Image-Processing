# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
import datetime, time
import os

def allBody():
    for k, v in BODY_PARTS.items():
        if v < 14:
            if points[BODY_PARTS[k]] == None:
                return False
    return True


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=200, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=200, type=int, help='Resize input to specific height.')

args = parser.parse_args()

DirToday = datetime.date.today().strftime("%y%m%d")
DirTime = time.strftime("%H%M%S")
print(DirToday)
print(DirTime)
mp4Dir = "Video/"+DirToday
if not os.path.isdir(mp4Dir):
    os.mkdir(mp4Dir)
timeDir = os.path.join(mp4Dir, DirTime)
os.mkdir(timeDir)

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

# POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
             ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["RHip", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"]]


inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture(args.input if args.input else 0)

index = 100
tin = 0

while cv.waitKey(1) < 0:
    flag = False
    hasFrame, frame = cap.read()
    frame = cv.flip(frame, 1)       # 좌우반전
    #dst = cv.transpose(frame)
    #frame = cv.flip(dst, 0)

    #height, width, channel = frame.shape
    #matrix = cv.getRotationMatrix2D((width/2, height/2), 90, 1)
    #frame = cv.warpAffine(frame, matrix, (height, width))

    if not hasFrame:
        cv.waitKey()
        break
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert (len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    nframe= np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]

        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(nframe, points[idFrom], points[idTo], (0, 255, 0), 2)
            # cv.ellipse(nframe, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            # cv.ellipse(nframe, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255),3, cv.FILLED)

    # 골반 생성
    a = BODY_PARTS["RHip"]
    b = BODY_PARTS["LHip"]
    g = BODY_PARTS["Neck"]
    if points[a] != None and points[b] != None and points[BODY_PARTS["Neck"]] != None:
        c = (int((points[a][0] + points[b][0])/2), int((points[a][1] + points[b][1])/2))
        d = points[g]

        # cv.line(nframe, c, d, (0, 255, 0), 3)
        # cv.ellipse(nframe, c, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        # cv.ellipse(nframe, d, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        # 척추 생성
        f = (c[0], int((c[1] + d[1])/2))     # 척추 포인트
        cv.line(nframe, d, f, (0, 255, 0), 2)       # 목에서 척추
        cv.line(nframe, f, c, (0, 255, 0), 2)       # 척추에서 골반

    nframe = cv.cvtColor(nframe, cv.COLOR_RGB2GRAY)
    n_,nframe = cv.threshold(nframe,128,255,cv.THRESH_BINARY)
    nframe = cv.resize(nframe, dsize=(120,120))
    # eframe = cv.equalizeHist(gframe)
    # threframe = cv.inRange(eframe, 0, 10)


    # 영상 저장 여부 검사
    if allBody():
        # tin += 1
        # if tin % 3 == 0:
        index += 1
        name = str(index) + ".png"
        cv.imwrite(os.path.join(timeDir, name), nframe, [cv.IMWRITE_PNG_BILEVEL, 1])

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('OpenPose using OpenCV', frame)
    cv.imshow('new',nframe)