## Author: Mikian Musser

import glob
import os
import sys

print("Active:", str(sys.argv[1]))
images = glob.glob(str(sys.argv[1]))
print("Total Images:", len(images))

count = 0

for file in images:
    src = file.replace('\\', '/')
    extension = os.path.splitext(file)[1]
    if(extension != '.py' and extension != '.jpg'):
        dst = os.path.splitext(file)[0] + '.jpg'
        os.rename(src, dst)
        count = count + 1

print("Images renamed:", count)
