# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
import datetime, time
import os
import math

"""
summary : 두 벡터 사이각을 측정
parameter[1] - inputA :  시작 Point
parameter[2] - inputB :  중간 Point
parameter[3] - inputC :  끝 Point
"""
def checkDegree(inputA, inputB, inputC):            # 목, 척추, 골반, 방향
    # Vector 변환
    vA = (float(inputA[0] - inputB[0]), float(inputA[1] - inputB[1]))
    vB = (float(inputC[0] - inputB[0]), float(inputC[1] - inputB[1]))

    # Vector 크기
    vASize = math.sqrt(vA[0] ** 2 + vA[1] ** 2)
    vBSize = math.sqrt(vB[0] ** 2 + vB[1] ** 2)

    # Vector 내적
    vAIn = vA[0] * vB[0] + vA[1] * vB[1]

    # Vector 외적
    vAOut = vASize * vBSize

    # Radian
    Rad = math.acos(vAIn / vAOut)
    print("R : ", Rad)
    Deg = math.degrees(Rad)
    print("D : ", Deg)

    if Deg < 145:
        cv.line(frame, inputA, inputB, (0, 0, 255), 2)
        cv.line(frame, inputB, inputC, (0, 0, 255), 2)
        return Deg


def allBody():
    for k, v in BODY_PARTS.items():
        if 1 < v < 14 :
            if points[BODY_PARTS[k]] == None:
                return False
    return True

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=200, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=200, type=int, help='Resize input to specific height.')

args = parser.parse_args()

mp4Dir = "Video"
if not os.path.isdir(mp4Dir):
    os.mkdir(mp4Dir)
mp4Dir += "/a"
if not os.path.isdir(mp4Dir):
    os.mkdir(mp4Dir)

# DirToday = datetime.date.today().strftime("%y%m%d")
# DirTime = time.strftime("%H%M%S")
# print(DirToday)
# print(DirTime)
# mp4Dir = "Video/"+DirToday
# if not os.path.isdir(mp4Dir):
#     os.mkdir(mp4Dir)
# timeDir = os.path.join(mp4Dir, DirTime)
# os.mkdir(timeDir)

BODY_PARTS = {  "Nose": 0,      "Neck": 1,      "RShoulder": 2,     "RElbow": 3,        "RWrist": 4,
                "LShoulder": 5, "LElbow": 6,    "LWrist": 7,        "RHip": 8,          "RKnee": 9,
                "RAnkle": 10,   "LHip": 11,     "LKnee": 12,        "LAnkle": 13,       "REye": 14,
                "LEye": 15,     "REar": 16,     "LEar": 17,         "Background": 18,   "Waist": 19,
                "Groin": 20}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
             ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["RHip", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"]]

#ROI_PAIRS = [480,420,360,300,260,200]

inWidth = args.width
inHeight = args.height

vid_writer = cv.VideoWriter('Video/a_result.mp4',cv.VideoWriter_fourcc('m','p','4','v'), 24, (480,480))

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")  # ".protobuf" 파일

cap = cv.VideoCapture(args.input if args.input else 0)     # Args or Cam
#cap = cv.VideoCapture(0)                                   # Cam
#cap = cv.VideoCapture("a.mp4")                              # a.mp4

index = 100
roi = 0
roiX = 480

while cv.waitKey(1) < 0:
    flag = False
    hasFrame, frame = cap.read()

    frame = cv.flip(frame, 1)       # 좌우반전
    #dst = cv.transpose(frame)
    #frame = cv.flip(dst, 0)

    if not hasFrame:
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :21, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

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
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 2)
            cv.line(nframe, points[idFrom], points[idTo], (0, 255, 0), 2)
            cv.ellipse(frame, points[idFrom], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # 골반 생성
    indexRHip = BODY_PARTS["RHip"]
    indexLHip = BODY_PARTS["LHip"]
    indexNeck = BODY_PARTS["Neck"]
    indexWaist = BODY_PARTS["Waist"]
    indexGroin = BODY_PARTS["Groin"]
    if points[indexRHip] != None and points[indexLHip] != None and points[indexNeck] != None:
        points[indexGroin] = (int((points[indexRHip][0] + points[indexLHip][0])/2), int((points[indexRHip][1] + points[indexLHip][1])/2))    # c(x,y) : 양쪽 골반 사이 포인트

        cv.ellipse(frame, points[indexGroin], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)
        cv.ellipse(frame, points[indexNeck], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)

        # 척추 생성
        points[indexWaist] = (points[indexGroin][0], int((points[indexGroin][1] + points[indexNeck][1])/2.2))            # 척추 포인트 : X값은 골반사이포인트값, Y값은 목과 골반사이포인트의 중간점
        cv.line(frame, points[indexNeck], points[indexWaist], (0, 255, 0), 2)        # 목에서 척추
        cv.line(frame, points[indexWaist], points[indexGroin], (0, 255, 0), 2)        # 척추에서 골반
        cv.line(nframe, points[indexNeck], points[indexWaist], (0, 255, 0), 2)  # 목에서 척추
        cv.line(nframe, points[indexWaist], points[indexGroin], (0, 255, 0), 2)  # 척추에서 골반

    nframe = cv.cvtColor(nframe, cv.COLOR_RGB2GRAY)
    n_,nframe = cv.threshold(nframe,128,255,cv.THRESH_BINARY)
    nframe = cv.resize(nframe, dsize=(120,120))

    dir = 1
    # 영상 저장 여부 검사
    if allBody():
        #checkDegree(points[BODY_PARTS["RHip"]],points[BODY_PARTS["RKnee"]],points[BODY_PARTS["RAnkle"]])    # 오른다리 검사
        #checkDegree(points[BODY_PARTS["LHip"]], points[BODY_PARTS["LKnee"]], points[BODY_PARTS["LAnkle"]])  # 왼다리 검사
        checkDegree(points[BODY_PARTS["Neck"]], points[BODY_PARTS["Waist"]], points[BODY_PARTS["Groin"]])  # 허리 검사
        #print(ROI_PAIRS[roi], int((points[a][0] + points[b][0])/2))
        if roi <= 5 and roiX >= int((points[indexNeck][0] + points[indexGroin][0])/2):
            index += 1
            name = str(index) + ".png"
            #cv.imwrite(os.path.join(timeDir, name), nframe, [cv.IMWRITE_PNG_BILEVEL, 1])
            cv.imwrite("Video/a/" + name, nframe, [cv.IMWRITE_PNG_BILEVEL, 1])
            roi += 1
            roiX -= 60

    #t, _ = net.getPerfProfile()
    #freq = cv.getTickFrequency() / 1000
    #cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('OpenPose using OpenCV', frame)
    cv.imshow('new',nframe)

    vid_writer.write(frame)

vid_writer.release()