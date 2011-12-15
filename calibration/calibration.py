
import os.path
import glob
import cv
from operator import itemgetter
import numpy as np
from numpy import linalg as la
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

def main(path, fileformat, ir_prefix, rgb_prefix,
        key_image, outfile, save_py):
    cb_points = get_chessboard_points((8,6), 30, 30)
    
    for it, prefix in enumerate([ir_prefix, rgb_prefix]):
        wildcard = os.path.join(path, "%s*.%s" % (prefix, fileformat))
        print "Processing images %s..." % wildcard
        filenames = glob.glob(wildcard)
        if len(filenames) == 0:
            raise ValueError, "no matching images in the given path"
        
        images = [imread(fname) for fname in filenames]
        print "Extracting corners..."
        corners = [cv.FindChessboardCorners(i, (8,6)) for i in images]
        
        print "Calibrating..."
        intrinsic, extrinsic = calibrate(corners, cb_points, images[0].shape[:2])
        if it == 0:
            K_ir = intrinsic
            T_ir = extrinsic
        else:
            K_rgb = intrinsic
            T_rgb = extrinsic
    
    # Estimate the relative transformation from IR to RGB camera.
    print "Estimating the relative transformation between cameras using the image %s..." % key_image
    T = np.dot(T_rgb[key_image], la.inv(T_ir[key_image]))
    
    print "Saving the results in %s..." % outfile
    if save_py:
        with open(outfile, "w") as f:
            f.write("K_rgb = %r\n" % K_rgb)
            f.write("K_ir = %r\n" % K_ir)
            f.write("T = %r\n" % T)
        return
    
    with open(outfile, "w") as f:
        f.write("# Intrinsics of the RGB camera (f_x,f_y,pp_x,pp_y)\n")
        f.write("%s %s %s %s\n"%(K_rgb[0,0], K_rgb[1,1], K_rgb[0,2], K_rgb[1,2]))
        f.write("# Intrinsics of the IR camera (f_x,f_y,pp_x,pp_y)\n")
        f.write("%s %s %s %s\n"%(K_ir[0,0], K_ir[1,1], K_ir[0,2], K_ir[1,2]))
        f.write("# Relative orientation of camera IR from camera RGB (rotation matrix)\n")
        f.write(("%s "*9 + "\n")%(tuple(list(T[:3,:3].ravel()))))
        f.write("# Relative translation of camera IR from camera RGB\n")
        f.write(("%s "*3 + "\n")%(tuple(list(T[:3,3].ravel()))))
    
    print "Success!"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate the Kinect cameras from images.")
    parser.add_argument("-f", "--format", default="png", type=str,
                        help="the format of the images")
    parser.add_argument("--ir-prefix", default="ir", type=str, dest="ir_prefix",
                        help="the prefix of the IR images")
    parser.add_argument("--rgb-prefix", default="rgb", type=str, dest="rgb_prefix",
                        help="the prefix of the RGB images")
    parser.add_argument("-o", "--output", default="kinect_calib.txt",
                        help="the name of the output file")
    parser.add_argument("--py", default=False, const=True, action="store_const",
                        dest="save_py",
                        help="save the results with a Python format")
    parser.add_argument("-k", "--key-image", dest="key_image", default=0, type=int,
                        help="the index of the image that will be used to estimate the relative transformation between cameras")
    parser.add_argument("path", nargs='?', default=".",
                        help="the path where the images ir* and rgb* are")
    args = parser.parse_args()
    main(args.path, args.format, args.ir_prefix, args.rgb_prefix,
        args.key_image, args.output, args.save_py)
