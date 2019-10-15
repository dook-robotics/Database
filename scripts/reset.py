import os
import sys
import glob
import shutil
import argparse

parser = argparse.ArgumentParser(
                                 description = 'Dook Robotics',
                                 epilog = "Dook Robotics - https://github.com/dook-robotics"
                                )

parser.add_argument(
                               '-g',
                     dest    = 'groupCLA',
                     action  = 'store',
                     default = -1,
                     help    = 'Select a version of the database.'
                    )

args = parser.parse_args()

images = glob.glob("C:/Users/bobar/Documents/GitHub/Database/reduced/all/images/*/*")
xml = glob.glob("C:/Users/bobar/Documents/GitHub/Database/reduced/all/xml/*/*")

test = glob.glob("C:/Users/bobar/Documents/GitHub/Database/reduced/test/*")
verification = glob.glob("C:/Users/bobar/Documents/GitHub/Database/reduced/verification/*")

groupChoice = int(args.groupCLA)
group = [
    # 0
    [
    "singleGrid",
    "trippleGrid",
    "tripple"
    ],
    # 1
    [
    "trippleGrass"
    ]
]

counttest = 0
for file in test:
    shutil.move(test[counttest], test[counttest].replace('test', 'train'))
    counttest = counttest + 1

countv = 0
for file in verification:
    shutil.move(verification[countv], verification[countv].replace('verification', 'train'))
    countv = countv + 1

print("\nFiles moved from test         :", counttest)
print("Files moved from verification :", countv)

countr = 0
files = glob.glob('C:/Users/bobar/Documents/GitHub/Database/reduced/train/*')
for f in files:
    os.remove(f.replace('\\', '/'))
    countr = countr + 1

print("\nFiles removed from train      :", countr)

imageCount = 0
trainDir = 'C:/Users/bobar/Documents/GitHub/Database/reduced/train/'
for i, f in enumerate(images):
    images[i]  = images[i].replace('\\', '/')
    xml[i]  = xml[i].replace('\\', '/')
    filename   = images[i].split('/')
    foldername = filename[len(filename) - 2]
    filename   = filename[len(filename) - 1]
    filenamenoext = filename.replace("jpg","xml")
    if groupChoice == -1:
        shutil.copyfile(images[i], 'C:/Users/bobar/Documents/GitHub/Database/reduced/train/' + filename)
        shutil.copyfile(xml[i], 'C:/Users/bobar/Documents/GitHub/Database/reduced/train/' + filenamenoext)
        imageCount = imageCount + 1
    elif foldername in group[groupChoice]:
        shutil.copyfile(images[i], 'C:/Users/bobar/Documents/GitHub/Database/reduced/train/' + filename)
        shutil.copyfile(xml[i], 'C:/Users/bobar/Documents/GitHub/Database/reduced/train/' + filenamenoext)
        imageCount = imageCount + 1
    pass

print("Files moved to train          :", imageCount)
