# MarkerDemo: Automatically Prepare Your Own Pascal VOC2007 Dataset

## Function
This demo is a tool to mark given images or videos into a dataset with format of PASCAL VOC2007 
.

PASCAL VOC2007 is a format of dataset which can be used in later deep learning training
like YOLOv4. It usually contains five folders:
- Annotations: store files in xml, every xml file represents the locations of objects in a corresponding image
- JPEGImages: store images in jpeg, including all the images used for training and test
- ImageSets: with three folders: Layout,Main,Segmentation
- SegmentationClass
- SegmentationObject

Typically, people use tools like LabelImg to mark the objects in images manually. For some objects which can be recognized using 
simple image processing methods, we could automatically mark them into PASCAL VOC2007 format without 
repeatedly hand work.



## How-to-use Tutorial
1. Install packages in PyCharm: opencv-python, numpy, lxml
2. Create a new folder under project directory, named: videosource
3. Drag the video into the folder
4. Modify the video name in main.py
5. Nodify thresholds in functions.py according to color.etc
6. Run main.py