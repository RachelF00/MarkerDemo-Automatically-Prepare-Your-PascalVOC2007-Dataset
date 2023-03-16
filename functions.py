# where demo begins

# import function_source
import os
import cv2 as cv
import numpy as np
from lxml.etree import Element, SubElement, tostring


# create directories in PASCAL VOC2007
def menu_create():
 os.makedirs('./VOC2007/Annotations')
 os.makedirs('./VOC2007/ImageSets')
 os.makedirs('./VOC2007/ImageSets/Main')
 os.makedirs('./VOC2007/ImageSets/Layout')
 os.makedirs('./VOC2007/ImageSets/Segmentation')
 os.makedirs('./VOC2007/JPEGImages')
 os.makedirs('./VOC2007/SegmentationClass')
 os.makedirs('./VOC2007/SegmentationObject')

# create folder to save visible results
 os.makedirs('./FrameResult')


# function to preprocess images: binary, dilate,
def preprocess(src):
        src1 = src
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, (20, 20, 170), (70, 70, 230))  # 通过hsv的阈值分割，适用于clip

        result = cv.bitwise_and(src, src, mask=mask)
        result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

        # make binary images
        ret, im_binary = cv.threshold(result, 80, 255, cv.THRESH_BINARY)

        # dilate
        im_dilate = cv.dilate(im_binary, (5, 5), iterations=15)
        return(im_dilate)


# function to find contours of clips
def find_cnt(im_dilate):
        # find contours
        cnts, hie = cv.findContours(
            im_dilate,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE)
        temp = np.ones(im_dilate.shape, np.uint8) * 255

        # draw contours
        im_cnts = cv.drawContours(temp,
                                  cnts,
                                  -1,
                                  (0, 0, 255), 5)
        return(cnts)


# function to find objects and make annotation (substitutes LabelImg manually)
def make_annotation(im_name, cnts, src):
    image_name = im_name
    save_dir = './VOC2007/Annotations'
    width = 1080
    height = 1920
    channel = 3
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    # traverse all the contours
    for j in range(len(cnts)):
        x, y, w, h = cv.boundingRect(cnts[j])
        if w > 60 or h > 60:

            rect = cv.rectangle(src, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 4)

            left, top, right, bottom = x-5, y-5, x + w+5, y + h+5
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = 'clip'
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = '%s' % left
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = '%s' % top
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = '%s' % right
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = '%s' % bottom
          #  print("0")
            cv.imwrite('./FrameResult/{}.jpg'.format(im_name), rect)

    xml = tostring(node_root, pretty_print=True)
    # dom = parseString(xml)
    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)


class BatchRename():
    '''
    Rename all the images in given directory
    '''

    def __init__(self):
        # directory of images
        self.path = './VOC2007/JPEGImages'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1       # set the name of first image
        n = 6       # set the length of file name, e.g.000001，with a length of 6
        for item in filelist:
            # rename all the jpg files
            if item.endswith('.jpg'):
                n = 6 - len(str(i))
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(0) * n + str(i) + '.jpg')
                try:
                    os.rename(src, dst)
                 #   print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i-1))






