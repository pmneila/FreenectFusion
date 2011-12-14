
import numpy as np
from scipy.misc import imsave
import freenect
import cv

def prepare_cv(video):
    image = cv.CreateImageHeader((video.shape[1], video.shape[0]),
                                 cv.IPL_DEPTH_8U,
                                 3)
    cv.SetData(image, video.tostring(),
               video.dtype.itemsize * 3 * video.shape[1])
    return image

def gray2color(img):
    new_img = np.zeros(img.shape + (3,), dtype=img.dtype)
    new_img[:] = img[:,:,np.newaxis]
    return new_img

def capture_pair(index, ir_img):
    
    rgb_img = freenect.sync_get_video(format=freenect.VIDEO_RGB)[0]
    detected, corners = cv.FindChessboardCorners(prepare_cv(rgb_img), (8,6))
    if not detected:
        return index
    imsave("ir%03d.png"%index, ir_img)
    imsave("rgb%03d.png"%index, rgb_img)
    return index + 1

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
        img = gray2color(freenect.sync_get_video(format=freenect.VIDEO_IR_8BIT)[0])
        img_cv = prepare_cv(img)
        
        detected, corners = cv.FindChessboardCorners(img_cv, (8,6))
        if detected:
            cv.DrawChessboardCorners(img_cv, (8,6), corners, 1)
        
        cv.ShowImage("Capture", img_cv)
        
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
