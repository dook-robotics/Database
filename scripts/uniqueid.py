# python uniqueid.py D:/Database/reduced/all/images/*/*
# python uniqueid.py D:/Database/reduced/all/xml/*/*
# python uniqueid.py C:/Users/bobar/Documents/GitHub/Database/reduced/all/images/*/*

import glob
import os
import sys

print("Active:", str(sys.argv[1]))
images = glob.glob(str(sys.argv[1]))
print("Total Images:", len(images))

count = 0
for file in images:
    src = file.replace('\\', '/')
    name = src.split('/')
    folder = name[len(name) - 2]
    name = name[len(name) - 1]
    if not "sg" in name and not "t" in name and not "tg" in name and not "tr" in name and not "trtwo" in name and not "tgrass" in name and not "r" in name and not "rp" in name and not "rptwo" in name:
        dst = src
        if(folder == "singleGrid"):
            dst = dst.replace(name, "sg" + name)
        if(folder == "tripple"):
            dst = dst.replace(name, "t" + name)
        if(folder == "trippleGrid"):
            dst = dst.replace(name, "tg" + name)
        if(folder == "trippleRocks"):
            dst = dst.replace(name, "tr" + name)
        if(folder == "trippleRocks2"):
            dst = dst.replace(name, "trtwo" + name)
        if(folder == "trippleGrass"):
            dst = dst.replace(name, "tgrass" + name)
        if(folder == "rope"):
            dst = dst.replace(name, "r" + name)
        if(folder == "ropePoop"):
            dst = dst.replace(name, "rp" + name)
        if(folder == "ropePoop2"):
            dst = dst.replace(name, "rptwo" + name)
        if(folder == "ropePoop3"):
            dst = dst.replace(name, "rpthree" + name)
        if(folder == "ropePoop4"):
            dst = dst.replace(name, "rpfour" + name)
        os.rename(src, dst)
        count = count + 1
        print(dst)

print("Images renamed:", count)
