from PIL import Image
import glob
# images = glob.glob("D:/Database/raw/*/*")
# images = glob.glob("D:/Database/raw/trippleGrass/*")
images = glob.glob("C:/Users/bobar/Documents/GitHub/Database/raw/ropePoop4/*")

# 1.7777
width = 960
height = 540

for file in images:
    imageFile = file.replace("\\","/")
    # print(imageFile)
    im1 = Image.open(imageFile)
    imageFile = imageFile.replace("raw", "reduced")
    im = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
    print(imageFile)
    im.save(imageFile)
    pass
