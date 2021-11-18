import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

TRESHOLD = 15

'''Taking all images that we want to classify for them'''
path= "\\FeatureBasedTemplateMatching\\Class\\"
images = []
classname = []
image_list = os.listdir(path)

'''Creating ORB object'''#Fast and Free to use
orb = cv2.ORB_create()

'''Creating classes via image names'''
for clss in image_list:
    imgCurrent = cv2.imread(f'{path}{clss}',0)
    images.append(imgCurrent)
    classname.append(os.path.splitext(clss)[0])

'''Finding All Descriptors'''
def findDesc(images):
    descList = []
    for image in images:
        kp,desc = orb.detectAndCompute(image,None)
        descList.append(desc)
    return descList

'''Finding image id via using descritor list'''
def findID(img, descList):
    kp2, desc2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalval = -1
    try:
        for des in descList:
            matches = bf.knnMatch(des,desc2,k=2)
            goodmatches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    goodmatches.append([m])
            matchList.append(len(goodmatches))
    except:
        pass
    if matchList:
        if max(matchList) > TRESHOLD:
            finalval = matchList.index(max(matchList))
    return finalval


'''Image that we want to detect'''
detection_image = cv2.imread("\\FeatureBasedTemplateMatching\\10kmmatch.jpg")
img_gray = cv2.cvtColor(detection_image,cv2.COLOR_BGR2GRAY)


descList = findDesc(images)
id =findID(img_gray,descList)

if id != -1:
    cv2.putText(detection_image,classname[id],(50,50),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),3)



plt.imshow(detection_image)
plt.show()