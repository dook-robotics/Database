## rename.py ##
## Rename files with .JPG to .jpg ##
#
# Authors:
#   Mikian Musser - https://github.com/mm909
#   Eric Becerril-Blas - https://github.com/lordbecerril
#   Zoyla O - https://github.com/ZoylaO
#   Austin Janushan - https://github.com/Janushan-Austin
#   Giovanny Vazquez - https://github.com/giovannyVazquez
#
# Organization:
#   Dook Robotics - https://github.com/dook-robotics
#
# Usage:
#   python rename.py {Path to dir}/*
#
# Documentation:
#
#

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
