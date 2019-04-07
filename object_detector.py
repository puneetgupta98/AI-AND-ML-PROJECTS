import cv2
import numpy as np

MIN_MATCH_COUNT=30

detector=cv2.SURF()
FLANN_INDEX_KDTREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDTREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread('trainingData/TrainImg.jpg',0)  #to make gray scale
trainKP,trainDecs=detector.detectAndCompute(trainImg,None)

cam=cv2.VideoCapture(0)
while True:
    ret,QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDecs,k=2)
    
    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)

    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,gp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,gp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainingBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainingBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,200,0),2)
    else:
        print "Not enough matches-%d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
    cv2.imshow('result',QueryImgBGR)
    cv2.waitKey(10)
