import os
import cv2 as cv
import functions

v_path = './videosource/clip.MP4'

cap = cv.VideoCapture(v_path)
frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv.CAP_PROP_FPS)

functions.menu_create()


# Save original images into folder JPEGImages
for i in range(int(frame_count)):
  _, pic = cap.read()
  if i%20 == 0: # Sample the video every 20 frames
      cv.imwrite('./VOC2007/JPEGImages/image{}.jpg'.format(i), pic)

# Rename the images
demo = functions.BatchRename()
demo.rename()


path = './VOC2007/JPEGImages'
filelist = os.listdir(path)


for pic in filelist:
    #print(pic)
    p = cv.imread("./VOC2007/JPEGImages/{}".format(pic))
    p1 = functions.preprocess(p)
    cnts1=functions.find_cnt(p1)
    functions.make_annotation('{}'.format(pic),cnts1,p)
