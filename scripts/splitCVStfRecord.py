from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import glob
import io
import random
import shutil
import math
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'poop':
        return 1
    else:
        None

def splittf(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def record(dirpath):
    writer = tf.io.TFRecordWriter('D:/Database/reduced/'+dirpath+'.record')
    path = os.path.join(os.getcwd(), 'D:/Database/reduced/'+dirpath)
    examples = pd.read_csv('D:/Database/reduced/'+dirpath+'.csv')
    grouped = splittf(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), 'D:/Database/reduced/'+dirpath+'.record')
    print('Successfully created the TFRecords: {}'.format(output_path))

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            # print(root.find('filename').text)
            dirs = xml_file.replace('\\', '/').split('/')
            dirs = dirs[len(dirs) - 1].split('.')[0] + ".jpg"
            # root.find('filename').text
            value = (dirs,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

os.system('cls')

split = .2

test = glob.glob("D:/Database/reduced/test/*")
verification = glob.glob("D:/Database/reduced/verification/*")

counttest = 0
for file in test:
    shutil.move(test[counttest], test[counttest].replace('test', 'train'))
    counttest = counttest + 1

countv = 0
for file in verification:
    shutil.move(verification[countv], verification[countv].replace('verification', 'train'))
    countv = countv + 1

trainImages = glob.glob("D:/Database/reduced/train/*.jpg")
trainXML = glob.glob("D:/Database/reduced/train/*.xml")

print("\nImages moved from test to train         :", int(counttest/2))
print("Images moved from verification to train :", int(countv/2))
print("\nImages in train folder                  :", len(trainImages))
print("XML in train folder                     :", len(trainXML))

testSize         = int((int(split * len(trainImages))) / 2)
verificationSize = testSize

count = 0
while count != testSize:
    x = random.randint(0, len(trainImages)-1-count)
    shutil.move(trainImages[x], trainImages[x].replace('train', 'test'))
    shutil.move(trainXML[x], trainXML[x].replace('train', 'test'))
    trainImages.remove(trainImages[x])
    trainXML.remove(trainXML[x])
    count = count + 1

print("\nImages moved from train to test         :", count)

count = 0
while count != verificationSize:
    x = random.randint(0, len(trainImages)-1-count)
    shutil.move(trainImages[x], trainImages[x].replace('train', 'verification'))
    shutil.move(trainXML[x], trainXML[x].replace('train', 'verification'))
    trainImages.remove(trainImages[x])
    trainXML.remove(trainXML[x])
    count = count + 1
print("Images moved from train to verification :", count)

xml_df = xml_to_csv("D:/Database/reduced/test/")
print("\nCreating: D:/Database/reduced/all/csv/test.csv")
xml_df.to_csv(('D:/Database/reduced/all/csv/test.csv'), index=None)
print("Creating: D:/Database/reduced/test.csv")
xml_df.to_csv(('D:/Database/reduced/test.csv'), index=None)

xml_df = xml_to_csv("D:/Database/reduced/train/")
print("\nCreating: D:/Database/reduced/all/csv/train.csv")
xml_df.to_csv(('D:/Database/reduced/all/csv/train.csv'), index=None)
print("Creating: D:/Database/reduced/train.csv\n")
xml_df.to_csv(('D:/Database/reduced/train.csv'), index=None)

record("train")
record("test")
