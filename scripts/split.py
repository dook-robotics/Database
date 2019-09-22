import sys
import glob
import random
import shutil
import math

split = .2

test = glob.glob("D:/Database/reduced/test/*")

count = 0
for file in test:
    shutil.move(test[count], test[count].replace('test', 'train'))
    count = count + 1

trainImages = glob.glob("D:/Database/reduced/train/*.jpg")
trainXML = glob.glob("D:/Database/reduced/train/*.xml")

print("\nImages moved from test to train:", count)
print("\nImages in test folder  :", int(len(test)/2))
print("Images in train folder :", len(trainImages))
print("XML in train folder    :", len(trainXML))

testSize = (int(split * len(trainImages)))

count = 0
while count != testSize:
    x = random.randint(0, len(trainImages)-1-count)
    shutil.move(trainImages[x], trainImages[x].replace('train', 'test'))
    shutil.move(trainXML[x], trainXML[x].replace('train', 'test'))
    trainImages.remove(trainImages[x])
    trainXML.remove(trainXML[x])
    count = count + 1

print("\nImages moved from train to test:", count)
