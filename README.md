# Feature Detection Based Template Matching

The classification of the photos was made using the OpenCv template Matching method.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install OpenCV and Matplotlib

```bash
pip install opencv-python
pip install matplotlib
```

## Code Review
### Loading Images
```python
'''Taking all images that we want to classify for them'''
path= "..\\FeatureBasedTemplateMatching\\Class\\"
images = []
classname = []
image_list = os.listdir(path)

```
### Creating Classes

```python
'''Creating classes via image names'''
for clss in image_list:
    imgCurrent = cv2.imread(f'{path}{clss}',0)
    images.append(imgCurrent)
    classname.append(os.path.splitext(clss)[0])

```

### Creating ORB Object
About [ORB ](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html)
```python
'''Creating ORB object'''#Fast and Free to use
orb = cv2.ORB_create()

```
### Finding all Decriptors
#### Computed descriptors. Output concatenated vectors of descriptors. Each descriptor is a 32-element vector, as returned by cv.ORB.descriptorSize, so the total size of descriptors will be numel(keypoints) * obj.descriptorSize(), i.e a matrix of size N-by-32 of class uint8, one row per keypoint.

```python
'''Finding All Descriptors'''
def findDesc(images):
    descList = []
    for image in images:
        kp,desc = orb.detectAndCompute(image,None)
        descList.append(desc)
    return descList

```

### Finding Detection Image ID
```python
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

```
### Detection
```python
'''Image that we want to detect'''
detection_image = cv2.imread("..\\FeatureBasedTemplateMatching\\10kmmatch.jpg")
img_gray = cv2.cvtColor(detection_image,cv2.COLOR_BGR2GRAY)


descList = findDesc(images)
id =findID(img_gray,descList)

if id != -1:
    cv2.putText(detection_image,classname[id],(50,50),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),3)
```

### Output
![alt text](https://github.com/muhammederem/FeatureDetectionBasedTemplateMatching/blob/main/Source/output.jpg?raw=true)




## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)