
import glob
import cv
from operator import itemgetter
import numpy as np
from scipy.misc import imread

def get_chessboard_points(chessboard_shape, dx, dy):    
    return np.array([(x*dx,y*dy,0) for x in xrange(chessboard_shape[1])
                                    for y in xrange(chessboard_shape[0])])

def calibrate(image_corners, chessboard_points, image_size):
    """Calibrate a camera.
    
    This function determines the intrinsic matrix and the extrinsic
    matrices of a camera.
    
    Parameters
    ----------
    image_corners : list
        List of the M outputs of cv.FindChessboardCorners, where
        M is the number of images.
    chessboard_points : ndarray
        Nx3 matrix with the (X,Y,Z) world coordinates of the
        N corners of the calibration chessboard pattern.
    image_size : tuple
        Size (height,width) of the images captured by the camera.
    
    Output
    ------
    intrinsic : ndarray
        3x3 intrinsic matrix
    extrinsic : list of ndarray
        List of M 4x4 transformation matrices or None values. For the images
        which had good detections given by cv.FindChessboardCorners,
        the corresponding cells have the extrinsic matrices. For the images
        with bad detections, the corresponding cells are None.
    """
    valid_corners = filter(itemgetter(0), image_corners)
    num_images = len(image_corners)
    num_valid_images = len(valid_corners)
    num_corners = len(valid_corners[0][1])
    
    # Input data.
    object_points = np.vstack([chessboard_points] * num_valid_images)
    image_points = np.vstack(map(itemgetter(1), valid_corners))
    point_counts = np.array([[num_corners]*num_valid_images])
    
    # Output matrices.
    intrinsic = np.zeros((3,3))
    dist_coeffs = np.zeros((1,4))
    rvecs = np.zeros((len(valid_corners),9))
    tvecs = np.zeros((len(valid_corners),3))
    
    # Calibrate.
    cv.CalibrateCamera2(np.array(object_points, dtype=np.float_),
                        np.array(image_points, dtype=np.float_),
                        point_counts,
                        image_size, intrinsic, dist_coeffs,
                        rvecs, tvecs)
    
    # Build the transformation matrices.
    rvecs = iter(rvecs)
    tvecs = iter(tvecs)
    def vecs2matrices(c):
        if c[0] == 0:
            return None
        R = np.reshape(rvecs.next(), (3,3))
        t = np.reshape(tvecs.next(), (3,1))
        return np.vstack([np.hstack([R,t]), [0,0,0,1]])
    
    extrinsic = map(vecs2matrices, image_corners)
    
    return intrinsic, extrinsic

def main():
    cb_points = get_chessboard_points((8,6), 30, 30)
    
    for prefix in ['ir', 'rgb']:
        filenames = glob.glob("%s*.png" % prefix)
        images = [imread(fname) for fname in filenames]
        corners = [cv.FindChessboardCorners(i, (8,6)) for i in images]
        
        intrinsic, extrinsic = calibrate(corners, cb_points, images[0].shape[:2])
        np.savez(prefix, intrinsic=intrinsic, extrinsic=extrinsic)

if __name__ == '__main__':
    main()
