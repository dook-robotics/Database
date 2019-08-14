## picture.py ##
## Take a picture on the pi and store it on a USB ##
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
#   python picture.py
#
# Todo:
#   Take in the file storage path as CLA
#

from picamera import PiCamera
from pynput.keyboard import Key, Listener
import glob
import os

def onPress(key):
    if key == Key.f1:
        global camera
        global index
        camera.capture('/media/pi/349E-21882/data/%s.jpg' % index)
        print('Taken: %s X %s -> %s.jpg' % (IM_WIDTH,IM_HEIGHT,index))
        index = index + 1
    if key == Key.esc:
        global pic
        pic = 0
        listener.stop()
        exit()
pic = 1
latest = 0;
#images = glob.glob('/home/pi/Desktop/data/*')
images = glob.glob('/media/pi/349E-21882/data/*')

if(len(images) > 0):
    for file in images:
        if(int(int(os.path.basename(file)[0])) > latest):
            latest = int(os.path.basename(file)[0])

if latest != 0:
    index = latest + 1
else:
    index = 0

print('Starting with image: %s.jpg' % index)

camera = PiCamera()
IM_WIDTH = 1280 #640 #2592 #1296 #1280
IM_HEIGHT = 1280 #640 #1944 #972 #720

camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.start_preview()

with Listener(on_press=onPress) as listener:
    listener.join()

while pic:
    continue
camera.stop_preview()

listener.stop()
