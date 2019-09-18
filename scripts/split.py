import sys
import glob
import random
import shutil

def getList():
    list = []
    for i in range(40):
        x = random.randint(0, totalImage-1)
        if not x in list:
            list.append(x)
    return list

# Dirs
trainImages = glob.glob("C:/Users/bobar/Documents/GitHub/Database/reduced/train/*.jpg")
trainXML = glob.glob("C:/Users/bobar/Documents/GitHub/Database/reduced/train/*.xml")
test = glob.glob("C:/Users/bobar/Documents/GitHub/Database/reduced/test/*")

print("\nImages in test folder  :", int(len(test)/2))
print("Images in train folder :", len(trainImages))
print("XML in train folder    :", len(trainXML))

count = 0
for file in test:
    shutil.move(test[count], test[count].replace('test', 'train'))
    count = count + 1
print("\nImages moved from test to train:", count)

totalImage = len(trainImages)
listLen = 0
list = []

while listLen != 40:
    list = getList()
    listLen = len(list)

list.sort(reverse=True)
print("\nList size:", listLen)

count = 0
for number in list:
    shutil.move(trainImages[number], trainImages[number].replace('train', 'test'))
    shutil.move(trainXML[number], trainXML[number].replace('train', 'test'))
    count = count + 1
print("\nImages moved from train to test:", count)
