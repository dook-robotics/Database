import sys
import glob
import random
import shutil
import math

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
