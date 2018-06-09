# OPENCV SSD Face Detection and Evaluation

This repository contains OPENCV SSD face detection implementation. Advandages of this implementations is it gives real time runtime performance in CPU.  

### 1. Usage

At first you need to install OpenCV2.0.x+

```
$git clone https://github.com/ghimiredhikura/mtcnn-cpp.git
$cd mtcnn-cpp
$mkdir build
$cd build/
$cmake ..
$make 
```
#### 1. Test webcam
```
$./face_detection -mode=0 -webcam=0
```
#### 2. Test single image
```
$./face_detection -mode=1 -path=../image/1.jpg
```
#### 3. Test image lists
```
$./face_detection -mode=2 -path=../image/
```
#### 4. Evaluation in benchmark dataset, detection files will be stored in "detections" folder. 
```
a) afw dataset
$./face_detection -mode=3 -dataset=AFW -path=/path/to/afw/dataset/

b) PASCAL dataset
$./face_detection -mode=3 -dataset=PASCAL -path=/path/to/pascal/dataset/
```