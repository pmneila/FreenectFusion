
import time
import numpy as np
from scipy.misc import imsave
import freenect
import cv

def capture_pair(index, ir_img):
    
    rgb_img = freenect.sync_get_video(format=freenect.VIDEO_RGB)[0]
    detected, corners = cv.FindChessboardCorners(rgb_img, (8,6))
    if not detected:
        return index
    imsave("ir%03d.png"%index, ir_img)
    imsave("rgb%03d.png"%index, rgb_img)
    return index + 1

def prepare_ir_img(ir_img):
    new_img = np.zeros(ir_img.shape + (3,), dtype=ir_img.dtype)
    new_img[:] = ir_img[:,:,np.newaxis]
    return new_img

def main():
    
    print "This is the capture utility for the FreenetFusion calibration."
    print "1 - Place the calibration template in front of the Kinect sensor"
    print "2 - Make sure the calibration template is detected (you will see a colored grid)"
    print "3 - Press SPACE. DO NOT MOVE the calibration template"
    print "4 - Repeat from 1 with many (~10-20) different poses of the calibration pattern"
    print "5 - Press ESC when done\n"
    
    cv.NamedWindow('Capture')
    index = 0
    while True:
        img = prepare_ir_img(freenect.sync_get_video(format=freenect.VIDEO_IR_8BIT)[0])
        
        detected, corners = cv.FindChessboardCorners(img, (8,6))
        if detected:
            img_show = np.copy(img)
            cv.DrawChessboardCorners(img_show, (8,6), corners, 1)
            cv.ShowImage("Capture", img_show)
        else:
            cv.ShowImage("Capture", img)
        
        key = cv.WaitKey(1)
        if key == 27:
            break
        
        if key != 32:
            continue
        
        if not detected:
            print "Chessboard template not detected. Please, retry.\n"
            continue
        
        print "Capturing and detecting the chessboard template in the RGB image..."
        new_index = capture_pair(index, img)
        if new_index == index:
            print "The calibration template was not found in the RGB image. Please, retry.\n"
            continue
        index = new_index
        print "Success! Images have been saved. You have taken %d correct image pairs." % index
        if index < 10:
            print "At least 10 images are required for a good calibration."
            print "Please, change the position of the calibration template and repeat.\n"

if __name__ == '__main__':
    main()
