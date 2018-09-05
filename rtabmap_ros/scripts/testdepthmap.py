# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

np.set_printoptions(threshold='nan') 

if __name__ == "__main__":
    
    input_dir = "./rgbdImages/rgbd_dataset_freiburg3_long_office_household/depth_sync"

    #monodepth_dir = 
    #image_files = [f for f in list(os.walk(input_dir))[0][2]]
    #image_paths = [os.path.join(input_dir,f) for f in image_files]

    orig_depth_image = scipy.misc.imread("./orig_depth.png")
    calc_depth_image = scipy.misc.imread("./depth3/orig_rgb.png")
    orig_rgb_image = scipy.misc.imread("./rgb3/orig_rgb.png")
    print(orig_depth_image.dtype, orig_depth_image.shape)
    print(calc_depth_image.dtype, calc_depth_image.shape)

    print(np.median(orig_depth_image),np.median(calc_depth_image))
    """
    input_dir = "../AR/monodepth/depth_sync_sample"
    image_files = [f for f in list(os.walk(input_dir))[0][2]]
    image_paths = [os.path.join(input_dir,f) for f in image_files]
    print("Reading images and converting them to 16 bit uint")
    output_dir = "../AR/monodepth/depth_sync_sample"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i,filepath in tqdm(enumerate(image_paths)):
        input_image = scipy.misc.imread(filepath)
        
            
        #input_image = input_image.astype(np.float32) / 255
        input_image = np.array(input_image, dtype=np.uint16)


        print(input_image.dtype)
        print(input_image.shape)
        #input_image = 1/input_image ##Convert to depth
        #input_image *= 256
        output_name = os.path.join(output_dir,image_files[i])
        plt.imshow(input_image)
        cv2.imwrite(output_name,input_image)
        #print(input_image)

        #print(np.amax(input_image),np.amin(input_image))
    """
        
        
    