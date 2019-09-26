import os
import cv2
import sys
import glob
import argparse
import progressbar
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from utils import label_map_util
from utils import visualization_utils as vis_util
import time
import math

# Add command line arguments
parser = argparse.ArgumentParser(
                                 description = 'Dook Robotics - Model Accuracy Script',
                                 epilog = "Dook Robotics - https://github.com/dook-robotics"
                                )

parser.add_argument(
                               '--fp',
                     dest    = 'fpCLA',
                     action  = 'store_true',
                     default = 'False',
                     help    = 'Prints out all files with a false positive.'
                    )

parser.add_argument(
                               '--save',
                     dest    = 'saveCLA',
                     action  = 'store_true',
                     default = 'False',
                     help    = 'Saves all images to an output file.'
                    )


args = parser.parse_args()


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]))

# Detection Threshold
THRESH = 0.5

# Directories and Models
BASE   = "D:/Models/poop/inference_graphs"
LABELS = "D:/Models/poop/labelmap.pbtxt"
MODEL_NAMES  =  [
                "faster_rcnn_inception_v2_pets_v1",
                "ssd_mobilenet_v2_0052_v1", # 0.98 (v1 of database)
                "ssd_mobilenet_v2_0037_v2", # 0.97 (v1 of database)
                "ssd_mobilenet_v2_0037_v3", # 0.95 (v1 of database)
                "ssd_mobilenet_v2_v4", # 0.94 (v1 of database)
                "ssd_mobilenet_v2_datav2_v1", # 96% overall #1386 Detections #38 False positives #15 Missed Objects
                # ssd_mobilenet_v2_datav2_v2 (0.75 thresh) 90% overall #1242 Detections #7 False positives #129 Missed Objects
                # Those 7 FD break down like: 4 'thank you' rock 2 Red thing 1 normal rock
                "ssd_mobilenet_v2_datav2_v2",
                # ssd_mobilenet_v2_datav2_v3 (0.75 thresh) 73% overall #998 Detections  #0 False positives #367 Missed Objects
                # ssd_mobilenet_v2_datav2_v3 (0.5 thresh)  80% overall #1092 Detections #0 False positives #273 Missed Objects
                # ssd_mobilenet_v2_datav2_v3 (0.25 thresh) 86% overall #1169 Detections #0 False positives #196 Missed Objects
                "ssd_mobilenet_v2_datav2_v3",
                "ssd_mobilenet_v2_datav2_v4",
                "ssd_mobilenet_v2.2.5", # Terrible soo many high confidence false detections
                # ssd_mobilenet_v2.2.6 (0.6 thresh)  93% overall #1274 Detections #4 False positives #95 Missed Objects
                # ssd_mobilenet_v2.2.6 (0.5 thresh)  95% overall #1291 Detections #8 False positives #82 Missed Objects
                "ssd_mobilenet_v2.2.6"
                ]

# Choose a model
# MODEL_NAME = MODEL_NAMES[7]
MODEL_NAME = MODEL_NAMES[len(MODEL_NAMES) - 1]
FROZEN_INFERENCE_GRAPH = os.path.join(BASE,MODEL_NAME,'frozen_inference_graph.pb').replace("\\","/")

# Image Directories
PATH_TO_TRAIN_IMAGES = "D:/Database/reduced/train/*.jpg"
PATH_TO_TEST_IMAGES  = "D:/Database/reduced/test/*.jpg"
PATH_TO_TRAIN_XML    = "D:/Database/reduced/train/*.xml"
PATH_TO_TEST_XML     = "D:/Database/reduced/test/*.xml"
OUTPUT               = "D:/Database/tests/*"
NUM_CLASSES          = 1

IM_WIDTH  = 960
IM_HEIGHT = 540

trainingImages = glob.glob(PATH_TO_TRAIN_IMAGES)
testingImages = glob.glob(PATH_TO_TEST_IMAGES)

# Grab all images
IMAGES = glob.glob(PATH_TO_TRAIN_IMAGES) + glob.glob(PATH_TO_TEST_IMAGES)
XML = glob.glob(PATH_TO_TRAIN_XML) + glob.glob(PATH_TO_TEST_XML)

# Clean output folder
OUTPUT_FILES = glob.glob(OUTPUT)
for file in OUTPUT_FILES:
    os.remove(file)

# Load labels and categories
label_map = label_map_util.load_labelmap(LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load tensorflow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# Define inputs and outputs
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

totalDetections = 0
totalObjects = 0
testingObjects = 0
testingDetections = 0
falseNumberDetection = 0
falseNumberDetectionTesting = 0

falsePositiveImages = []

# Run the model on every image
os.system('cls')
with progressbar.ProgressBar(max_value=len(IMAGES)) as bar:
    for xmlIndex, image in enumerate(IMAGES):

        isTestImage = False
        if image in testingImages:
            isTestImage = True

        # Get image name
        image = image.replace("\\","/")
        name = image.split("/")
        name = name[len(name) - 1]

        xml_file = XML[xmlIndex]
        tree = ET.parse(xml_file)
        root = tree.getroot()


        for menber in root.findall('object'):
            totalObjects = totalObjects + 1
            if isTestImage:
                testingObjects = testingObjects + 1

        # Load current image
        image = cv2.imread(image)
        image_expanded = np.expand_dims(image, axis=0)

        # Run object detection
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        numberDetection = 0;
        for index, box in enumerate(np.squeeze(boxes)):
            ymin = int((box[0]*IM_HEIGHT))
            xmin = int((box[1]*IM_WIDTH))
            ymax = int((box[2]*IM_HEIGHT))
            xmax = int((box[3]*IM_WIDTH))
            falsePositive = True
            if(scores[0][index] >= THRESH):
                centerx = int((xmin + xmax)/2)
                centery = int((ymin + ymax)/2)
                for menber in root.findall('object'):
                    truthxmin = int(menber[4][0].text)
                    truthymin = int(menber[4][1].text)
                    truthxmax = int(menber[4][2].text)
                    truthymax = int(menber[4][3].text)
                    truthcenterx = int((truthxmin + truthxmax)/2)
                    truthcentery = int((truthymin + truthymax)/2)
                    if distance((truthcenterx,truthcentery),(centerx,centery)) < 50:
                        falsePositive = False
                cv2.circle(image,(centerx,centery),5,(0,255,0),-1)
                numberDetection = numberDetection + 1
                if falsePositive:
                    if not name in falsePositiveImages:
                        falsePositiveImages.append(name)
                    falseNumberDetection = falseNumberDetection + 1
                    if isTestImage:
                        falseNumberDetectionTesting = falseNumberDetectionTesting + 1

            pass
        totalDetections = totalDetections + numberDetection

        if isTestImage:
            testingDetections = testingDetections + numberDetection

        # Draw boxes
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=THRESH)

        # Write to output
        if args.saveCLA:
            cv2.imwrite('D:/Database/tests/test' + name, image)
        bar.update(xmlIndex)

print("\nModel:", MODEL_NAME)
print("Threshold:", THRESH)

print("\n=========== Testing =========== ")
print("Detections               :", testingDetections)
print("Testing False Detections :", falseNumberDetectionTesting)
print("Successful Detections    :", testingDetections - falseNumberDetectionTesting)
print("Total objects            :", testingObjects)
print("True Accuracy            :", round((testingDetections - falseNumberDetectionTesting) / testingObjects, 2))
print("Effective Accuracy       :", round((testingDetections - falseNumberDetectionTesting * 2) / testingObjects, 2))
print("=============================== ")

print("\n============ Total ============ ")
print("Detections               :", totalDetections)
print("Total False Detections   :", falseNumberDetection)
print("Successful Detections    :", totalDetections - falseNumberDetection)
print("Total objects            :", totalObjects)
print("True Accuracy            :", round((totalDetections - falseNumberDetection) / totalObjects, 2))
print("Effective Accuracy       :", round((totalDetections - falseNumberDetection * 2) / totalObjects, 2))
print("=============================== ")

print("\n======= False Positives =======")
if len(falsePositiveImages) < 10 or args.fpCLA == "True":
    if(len(falsePositiveImages) == 0):
        print("No false positives!")
    else:
        for image in falsePositiveImages:
            print(image)
            pass
else:
    print("False Positive Files:", len(falsePositiveImages))
    print("Run with --fp option to show files")
