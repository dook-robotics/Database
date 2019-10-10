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

# python accuracy.py -m ssd_mobilenet_v2.2.6 -d v1
# python accuracy.py -m 10 -d verification -savefp
# python accuracy.py -save

# Add command line arguments
parser = argparse.ArgumentParser(
                                 description = 'Dook Robotics - Model Accuracy Script',
                                 epilog = "Dook Robotics - https://github.com/dook-robotics"
                                )

parser.add_argument(
                               '-fp',
                     dest    = 'fpCLA',
                     action  = 'store_true',
                     default = 'False',
                     help    = 'Prints out all files with a false positive.'
                    )

parser.add_argument(
                               '-save',
                     dest    = 'saveCLA',
                     action  = 'store_true',
                     default = 'False',
                     help    = 'Saves all images to an output file.'
                    )

parser.add_argument(
                               '-savefp',
                     dest    = 'savefpCLA',
                     action  = 'store_true',
                     default = 'False',
                     help    = 'Saves images with a false positive to an output file.'
                    )

parser.add_argument(
                               '-m',
                     dest    = 'modelCLA',
                     action  = 'store',
                     default = -1,
                     help    = 'Selects a saved model.'
                    )

parser.add_argument(
                               '-d',
                     dest    = 'dataCLA',
                     action  = 'store',
                     default = 0,
                     help    = 'Select a version of the database.'
                    )

parser.add_argument(
                               '-t',
                     dest    = 'threshCLA',
                     action  = 'store',
                     default = 0.5,
                     help    = 'Select a version of the database.'
                    )


args = parser.parse_args()

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]))

# Detection Threshold
threshCLA = float(args.threshCLA)
if threshCLA >= 0 and threshCLA <= 1:
    THRESH = threshCLA
else:
    THRESH = 0.5

# Directories and Models
BASE   = "D:/Models/poop/inference_graphs"
LABELS = "D:/Models/poop/labelmap.pbtxt"
MODEL_NAMES  =  [
                "ssd_mobilenet_v2.1.0",
                    # ==== Total - v2.1.0 - 0.50 ====
                    # Detections               : 748
                    # Total False Detections   : 0
                    # Successful Detections    : 748
                    # Total objects            : 767
                    # True Accuracy            : 0.98
                    # Effective Accuracy       : 0.98
                    # ===============================

                "ssd_mobilenet_v2.1.1",
                    # ==== Total - v2.1.1 - 0.50 ====
                    # Detections               : 745
                    # Total False Detections   : 0
                    # Successful Detections    : 745
                    # Total objects            : 767
                    # True Accuracy            : 0.97
                    # Effective Accuracy       : 0.97
                    # ===============================

                "ssd_mobilenet_v2.1.2",
                    # ==== Total - v2.1.2 - 0.50 ====
                    # Detections               : 727
                    # Total False Detections   : 0
                    # Successful Detections    : 727
                    # Total objects            : 767
                    # True Accuracy            : 0.95
                    # Effective Accuracy       : 0.95
                    # ===============================

                "ssd_mobilenet_v2.1.3",
                    # ==== Total - v2.1.3 - 0.50 ====
                    # Detections               : 718
                    # Total False Detections   : 0
                    # Successful Detections    : 718
                    # Total objects            : 767
                    # True Accuracy            : 0.94
                    # Effective Accuracy       : 0.94
                    # ===============================

                "ssd_mobilenet_v2.2.0",
                    # ==== Total - v2.2.0 - 0.50 ====
                    # Detections               : 1386
                    # Total False Detections   : 39
                    # Successful Detections    : 1347
                    # Total objects            : 1365
                    # True Accuracy            : 0.99
                    # Effective Accuracy       : 0.96
                    # ===============================

                "ssd_mobilenet_v2.2.1",
                    # ==== Total - v2.2.1 - 0.50 ====
                    # Detections               : 1325
                    # Total False Detections   : 15
                    # Successful Detections    : 1310
                    # Total objects            : 1365
                    # True Accuracy            : 0.96
                    # Effective Accuracy       : 0.95
                    # ===============================

                "ssd_mobilenet_v2.2.2",
                    # ==== Total - v2.2.2 - 0.50 ====
                    # Detections               : 1092
                    # Total False Detections   : 0
                    # Successful Detections    : 1092
                    # Total objects            : 1365
                    # True Accuracy            : 0.8
                    # Effective Accuracy       : 0.8
                    # ===============================

                "ssd_mobilenet_v2.2.3",
                    # ==== Total - v2.2.3 - 0.50 ====
                    # Detections               : 2066
                    # Total False Detections   : 778
                    # Successful Detections    : 1288
                    # Total objects            : 1365
                    # True Accuracy            : 0.94
                    # Effective Accuracy       : 0.37
                    # ===============================

                "ssd_mobilenet_v2.2.4",
                    # ==== Total - v2.2.4 - 0.50 ====
                    # Detections               : 56900
                    # Total False Detections   : 49941
                    # Successful Detections    : 6959
                    # Total objects            : 1365
                    # True Accuracy            : 5.1
                    # Effective Accuracy       : 0
                    # ===============================

                "ssd_mobilenet_v2.2.5",

                    # ===== SSD - v2.2.5 - 0.75 =====
                    # Detections               : 1213
                    # Total False Detections   : 3
                    # Successful Detections    : 1210
                    # Total objects            : 1365
                    # True Accuracy            : 0.89
                    # Effective Accuracy       : 0.88
                    # ========== Data - v2 ==========

                    # ===== SSD - v2.2.5 - 0.50 =====
                    # Detections               : 1291
                    # Total False Detections   : 8
                    # Successful Detections    : 1283
                    # Total objects            : 1365
                    # True Accuracy            : 0.94
                    # Effective Accuracy       : 0.93
                    # ========== Data - v2 ==========

                    # ===== SSD - v2.2.5 - 0.25 =====
                    # Detections               : 1359
                    # Total False Detections   : 18
                    # Successful Detections    : 1341
                    # Total objects            : 1365
                    # True Accuracy            : 0.98
                    # Effective Accuracy       : 0.97
                    # ========== Data - v2 ==========

                    # ==== Total - v2.2.5 - 0.50 ====
                    # Detections               : 2637
                    # Total False Detections   : 363
                    # Successful Detections    : 2274
                    # Total objects            : 2607
                    # True Accuracy            : 0.87
                    # Effective Accuracy       : 0.73
                    # ========== Data - v3 ==========

                    # ===== SSD - v2.2.5 - 0.25 =====
                    # Detections               : 2924
                    # Total False Detections   : 500
                    # Successful Detections    : 2424
                    # Total objects            : 2607
                    # True Accuracy            : 0.93
                    # Effective Accuracy       : 0.74
                    # ========== Data - v3 ==========

                "ssd_mobilenet_v2.3.0",
                    # ======= Total - v2.3.0  =======
                    # Detections               : 1973
                    # Total False Detections   : 30
                    # Successful Detections    : 1943
                    # Total objects            : 2607
                    # True Accuracy            : 0.75
                    # Effective Accuracy       : 0.73
                    # ===============================

                "ssd_mobilenet_v2.3.1",
                    # ======= Total - v2.3.1  =======
                    # Detections               : 1628
                    # Total False Detections   : 19
                    # Successful Detections    : 1609
                    # Total objects            : 2607
                    # True Accuracy            : 0.62
                    # Effective Accuracy       : 0.61
                    # ===============================

                "ssd_mobilenet_v2.3.2",
                    # ======= Total - v2.3.2  =======
                    # Detections               : 1528
                    # Total False Detections   : 40
                    # Successful Detections    : 1488
                    # Total objects            : 2607
                    # True Accuracy            : 0.57
                    # Effective Accuracy       : 0.56
                    # ===============================

                "ssd_mobilenet_v2.3.3",
                    # ==== Total - v2.3.3 - 0.25 ====
                    # Detections               : 2268
                    # Total False Detections   : 165
                    # Successful Detections    : 2103
                    # Total objects            : 2607
                    # True Accuracy            : 0.81
                    # Effective Accuracy       : 0.74
                    # ===============================

                    # ==== Total - v2.3.3 - 0.50 ====
                    # Detections               : 2067
                    # Total False Detections   : 117
                    # Successful Detections    : 1950
                    # Total objects            : 2607
                    # True Accuracy            : 0.75
                    # Effective Accuracy       : 0.7
                    # ===============================

                "ssd_mobilenet_v2.3.4",
                    # ==== Total - v2.3.4 - 0.25 ====
                    # Detections               : 1895
                    # Total False Detections   : 65
                    # Successful Detections    : 1830
                    # Total objects            : 2607
                    # True Accuracy            : 0.7
                    # Effective Accuracy       : 0.68
                    # ===============================

                    # ==== Total - v2.3.4 - 0.50 ====
                    # Detections               : 1752
                    # Total False Detections   : 51
                    # Successful Detections    : 1701
                    # Total objects            : 2607
                    # True Accuracy            : 0.65
                    # Effective Accuracy       : 0.63
                    # ===============================

                "frcnn_v2.1.0",
                    # ==== Total - v2.1.0 - 0.50 ====
                    # Detections               : 766
                    # Total False Detections   : 0
                    # Successful Detections    : 766
                    # Total objects            : 767
                    # True Accuracy            : 1.0
                    # Effective Accuracy       : 1.0
                    # ===============================

                "frcnn_v2.3.0",
                    # ==== Total - v2.3.0 - 0.50 ====
                    # Detections               : 2451
                    # Total False Detections   : 56
                    # Successful Detections    : 2395
                    # Total objects            : 2607
                    # True Accuracy            : 0.92
                    # Effective Accuracy       : 0.9
                    # ===============================

                "frcnn_v2.3.1"
                    # ==== Total - v2.3.1 - 0.50 ====
                    # Detections               : 2146
                    # Total False Detections   : 27
                    # Successful Detections    : 2119
                    # Total objects            : 2607
                    # True Accuracy            : 0.81
                    # Effective Accuracy       : 0.8
                    # ===============================
                ]

# Choose a model
if "." in args.modelCLA:
    MODEL_NAME = args.modelCLA
else:
    if int(args.modelCLA) >= 0:
        MODEL_NAME = MODEL_NAMES[int(args.modelCLA)]
    else:
        MODEL_NAME = MODEL_NAMES[len(MODEL_NAMES) - 1]

FROZEN_INFERENCE_GRAPH = os.path.join(BASE,MODEL_NAME,'frozen_inference_graph.pb').replace("\\","/")

# Image Directories
PATH_TO_TRAIN_IMAGES   = "D:/Database/reduced/train/*.jpg"
PATH_TO_TRAIN_XML      = "D:/Database/reduced/train/*.xml"
PATH_TO_TEST_IMAGES    = "D:/Database/reduced/test/*.jpg"
PATH_TO_TEST_XML       = "D:/Database/reduced/test/*.xml"
PATH_TO_V_IMAGES       = "D:/Database/reduced/verification/*.jpg"
PATH_TO_V_XML          = "D:/Database/reduced/verification/*.xml"
PATH_TO_SG_IMAGES      = "D:/Database/reduced/all/images/singleGrid/*.jpg"
PATH_TO_SG_XML         = "D:/Database/reduced/all/xml/singleGrid/*.xml"
PATH_TO_TR_IMAGES      = "D:/Database/reduced/all/images/tripple/*.jpg"
PATH_TO_TR_XML         = "D:/Database/reduced/all/xml/tripple/*.xml"
PATH_TO_TG_IMAGES      = "D:/Database/reduced/all/images/trippleGrid/*.jpg"
PATH_TO_TG_XML         = "D:/Database/reduced/all/xml/trippleGrid/*.xml"
PATH_TO_TROCKS_IMAGES  = "D:/Database/reduced/all/images/trippleRocks/*.jpg"
PATH_TO_TROCKS_XML     = "D:/Database/reduced/all/xml/trippleRocks/*.xml"
PATH_TO_TROCKS2_IMAGES = "D:/Database/reduced/all/images/trippleRocks2/*.jpg"
PATH_TO_TROCKS2_XML    = "D:/Database/reduced/all/xml/trippleRocks2/*.xml"
OUTPUT                 = "D:/Database/tests/*"
NUM_CLASSES            = 1

IM_WIDTH  = 960
IM_HEIGHT = 540

trainingImages = glob.glob(PATH_TO_TRAIN_IMAGES)
testingImages  = glob.glob(PATH_TO_TEST_IMAGES)
vImages        = glob.glob(PATH_TO_V_IMAGES)

# Grab all images
if args.dataCLA == 0:
    IMAGES = glob.glob(PATH_TO_TRAIN_IMAGES) + glob.glob(PATH_TO_TEST_IMAGES) + glob.glob(PATH_TO_V_IMAGES)
    XML = glob.glob(PATH_TO_TRAIN_XML) + glob.glob(PATH_TO_TEST_XML) + glob.glob(PATH_TO_V_XML)
if args.dataCLA == "ve":
    IMAGES = glob.glob(PATH_TO_V_IMAGES)
    XML = glob.glob(PATH_TO_V_XML)
if args.dataCLA == "tr":
    IMAGES = glob.glob(PATH_TO_TRAIN_IMAGES)
    XML = glob.glob(PATH_TO_TRAIN_XML)
if args.dataCLA == "te":
    IMAGES = glob.glob(PATH_TO_TEST_IMAGES)
    XML = glob.glob(PATH_TO_TEST_XML)
if args.dataCLA == "v1":
    IMAGES  = glob.glob(PATH_TO_SG_IMAGES)
    XML     = glob.glob(PATH_TO_SG_XML)
    IMAGES += glob.glob(PATH_TO_TR_IMAGES)
    XML    += glob.glob(PATH_TO_TR_XML)
    IMAGES += glob.glob(PATH_TO_TG_IMAGES)
    XML    += glob.glob(PATH_TO_TG_XML)
if args.dataCLA == "v2":
    IMAGES  = glob.glob(PATH_TO_SG_IMAGES)
    XML     = glob.glob(PATH_TO_SG_XML)
    IMAGES += glob.glob(PATH_TO_TR_IMAGES)
    XML    += glob.glob(PATH_TO_TR_XML)
    IMAGES += glob.glob(PATH_TO_TG_IMAGES)
    XML    += glob.glob(PATH_TO_TG_XML)
    IMAGES += glob.glob(PATH_TO_TROCKS_IMAGES)
    XML    += glob.glob(PATH_TO_TROCKS_XML)
if args.dataCLA == "v3":
    IMAGES  = glob.glob(PATH_TO_SG_IMAGES)
    XML     = glob.glob(PATH_TO_SG_XML)
    IMAGES += glob.glob(PATH_TO_TR_IMAGES)
    XML    += glob.glob(PATH_TO_TR_XML)
    IMAGES += glob.glob(PATH_TO_TG_IMAGES)
    XML    += glob.glob(PATH_TO_TG_XML)
    IMAGES += glob.glob(PATH_TO_TROCKS_IMAGES)
    XML    += glob.glob(PATH_TO_TROCKS_XML)
    IMAGES += glob.glob(PATH_TO_TROCKS2_IMAGES)
    XML    += glob.glob(PATH_TO_TROCKS2_XML)

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
vObjects = 0
vDetections = 0
vFalseDetections = 0
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

        isVImage = False
        if image in vImages:
            isVImage = True

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
            if isVImage:
                vObjects = vObjects + 1

        # Load current image
        image = cv2.imread(image)
        image_expanded = np.expand_dims(image, axis=0)

        # Run object detection
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        numberDetection = 0;
        printPictureFP = False
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
                        falsePositive  = False
                cv2.circle(image,(centerx,centery),5,(0,255,0),-1)
                numberDetection = numberDetection + 1
                if falsePositive:
                    printPictureFP = True
                    if not name in falsePositiveImages:
                        falsePositiveImages.append(name)
                    falseNumberDetection = falseNumberDetection + 1
                    if isTestImage:
                        falseNumberDetectionTesting = falseNumberDetectionTesting + 1
                    if isVImage:
                        vFalseDetections = vFalseDetections + 1

            pass
        totalDetections = totalDetections + numberDetection

        if isTestImage:
            testingDetections = testingDetections + numberDetection
        if isVImage:
            vDetections = vDetections + numberDetection

        if args.saveCLA != "False" or (printPictureFP and args.savefpCLA != "False"):

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

            cv2.imwrite('D:/Database/tests/test' + name, image)
        bar.update(xmlIndex)

print("\nModel     :", MODEL_NAME)
if args.dataCLA == 0:
    print("Data      : All Images")
else:
    print("Data      :", args.dataCLA)
print("Threshold :", THRESH)

if vObjects > 0:
    trueAcc = max(0, round((vDetections - vFalseDetections) / vObjects, 2))
    effectiveAcc = max(0, round((vDetections - vFalseDetections * 2) / vObjects, 2))
else:
    trueAcc = 0
    effectiveAcc = 0

if vObjects > 0:
    print("\n======== Verification ======== ")
    print("Detections               :", vDetections)
    print("Testing False Detections :", vFalseDetections)
    print("Successful Detections    :", vDetections - vFalseDetections)
    print("Total objects            :", vObjects)
    print("True Accuracy            :", trueAcc)
    print("Effective Accuracy       :", effectiveAcc)
    print("=============================== ")

if testingObjects > 0:
    trueAcc = max(0, round((testingDetections - falseNumberDetectionTesting) / testingObjects, 2))
    effectiveAcc = max(0, round((testingDetections - falseNumberDetectionTesting * 2) / testingObjects, 2))
else:
    trueAcc = 0
    effectiveAcc = 0

if testingObjects > 0:
    print("\n=========== Testing =========== ")
    print("Detections               :", testingDetections)
    print("Testing False Detections :", falseNumberDetectionTesting)
    print("Successful Detections    :", testingDetections - falseNumberDetectionTesting)
    print("Total objects            :", testingObjects)
    print("True Accuracy            :", trueAcc)
    print("Effective Accuracy       :", effectiveAcc)
    print("=============================== ")

if MODEL_NAME.split("_")[0] == "ssd":
    print("\n===== SSD -", MODEL_NAME.split("_")[2], "- {:0.2f} ===== ".format(THRESH))
else:
    print("\n==== FRCNN -", MODEL_NAME.split("_")[1], "- {:0.2f} ==== ".format(THRESH))

print("Detections               :", totalDetections)
print("Total False Detections   :", falseNumberDetection)
print("Successful Detections    :", totalDetections - falseNumberDetection)
print("Total objects            :", totalObjects)
print("True Accuracy            :", max(0, round((totalDetections - falseNumberDetection) / totalObjects, 2)))
print("Effective Accuracy       :", max(0, round((totalDetections - falseNumberDetection * 2) / totalObjects, 2)))

if args.dataCLA == 0:
    print("====== Data - All Images ======")
else:
    print("========== Data -", args.dataCLA, "==========")

print("\n======= False Positives =======")
if args.fpCLA != "False":
    if(len(falsePositiveImages) == 0):
        print("No false positives!")
    else:
        for image in falsePositiveImages:
            print(image)
            pass
else:
    print("False Positive Files:", len(falsePositiveImages))
    print("Run with -fp option to show files")
