# Inspired from 'https://github.com/harvitronix/five-video-classification-methods'.
# README!!
# 1. Download the dataset from UCF into the data folder: cd 'opt.ucf101_raw_dir' && wget http://crcv.ucf.edu/data/UCF101/UCF101.rar
# 2. Extract it with 'unrar e UCF101.rar' in 'opt.ucf101_raw_dir'. (unrar e UCF101.rar ::: extract files without archived paths)
# 3. Run Preprocess_UCF101.py to extract the RGB & Optical Flow images from the videos and also create a data file we can use for training and testing later.
# 3.1 Or Copy two tar.gz files from my computer.

import cv2
import os
import os.path
import csv
import glob
import argparse
from subprocess import call
import numpy as np

''' Parser ---------------------------------------------------------------------------------------------------------------------- '''
def parseConfiguration():
    # Initialize Parser
    parser = argparse.ArgumentParser()
    # Add Parser Arguments
    parser = parseAddArguments(parser)
    # Parse Arguments
    args = parser.parse_args()
    return args

def parseAddArguments(parser):
    # Initialize DataLoader Parameters - Directories
    parser.add_argument('--ucf101_dir',             type=str,       default='/home/dehlix/MyFolder/Dataset/UCF101/',                        help='UCF101 Dataset Directory')
    parser.add_argument('--ucf101_raw_dir',         type=str,       default='/home/dehlix/MyFolder/Dataset/UCF101/RawData/',                help='UCF101 Dataset RawData Directory')
    parser.add_argument('--ucf101_rgb_dir',         type=str,       default='/home/dehlix/MyFolder/Dataset/UCF101/RGB/',                    help='UCF101 Dataset RGB Directory')
    parser.add_argument('--ucf101_flow_dir',        type=str,       default='/home/dehlix/MyFolder/Dataset/UCF101/Flow/',                   help='UCF101 Dataset Optical Flow Directory')
    parser.add_argument('--ucf101_flow_low_bound',  type=float,     default=-20.0,                                                          help='UCF101 Dataset Optical Flow Low Bound, take the value from Charades Dataset')
    parser.add_argument('--ucf101_flow_high_bound', type=float,     default=+20.0,                                                          help='UCF101 Dataset Optical Flow High Bound, take the value from Charades Dataset')
    parser.add_argument('--ucf101_anno_dir',        type=str,       default='/home/dehlix/MyFolder/Dataset/UCF101/ucfTrainTestlist/',       help='UCF101 Dataset Annotation Directory')
    parser.add_argument('--ucf101_anno_version',    type=str,       default='01',                                                           help='UCF101 Dataset Annotation Split Version, 01, 02, 03')
    parser.add_argument('--ucf101_data_dir',        type=str,       default='Data/UCF101/',                                                 help='')
    return parser
''' Parser ---------------------------------------------------------------------------------------------------------------------- '''

''' Move Files ------------------------------------------------------------------------------------------------------------------ '''
def moveFiles(opt):
    # Get the videos in groups so we can move them
    train_test_list = getTrainTestLists(opt)
    # Move the files
    replaceFiles(opt, train_test_list)

def getTrainTestLists(opt):
    # Get our files based on version
    test_file  = os.path.join(opt.ucf101_anno_dir, 'testlist'  + opt.ucf101_anno_version + '.txt')
    train_file = os.path.join(opt.ucf101_anno_dir, 'trainlist' + opt.ucf101_anno_version + '.txt')
    if os.path.exists(test_file) and os.path.exists(train_file):
        # Build the test list
        with open(test_file) as f:
            test_list = [row.strip() for row in list(f)]  #
        # Build the train list. Extra step to remove the class index
        with open(train_file) as f:
            train_list = [row.strip() for row in list(f)]
            train_list = [row.split(' ')[0] for row in train_list]
        # Set the groups in a dictionary
        file_groups = {
            'Train': train_list,
            'Test': test_list
        }
    else:
        raise Exception('Annotation File(s) does not exist.')
    return file_groups

def replaceFiles(opt, file_groups):
    # Do each of our groups
    for group, videos in file_groups.items():  # group: train/test, videos: train_list/test_list
        # Do each of our videos
        for video in videos:  # videos: train_list/test_list
            # Get the parts
            parts = video.split('/')  # ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi -> ApplyEyeMakeup, v_ApplyEyeMakeup_g08_c01.avi
            classname = os.path.join(opt.ucf101_raw_dir, group, parts[0])
            filename  = os.path.join(opt.ucf101_raw_dir, parts[1])
            destname  = os.path.join(opt.ucf101_raw_dir, group, parts[0], parts[1])
            # Check if this class exists
            if not os.path.exists(classname):  # UCF101/RawData/train/classname
                print("Creating folder for %s ..." % classname)
                os.makedirs(classname)
            # Check if we have already moved this file, or at least that it exists to move
            if not os.path.exists(filename):
                print("Can't find %s to move. Skipping." % filename)
                continue
            # Move it.
            print("Moving %s to %s" % (filename, destname))
            os.rename(filename, destname)
    print("Done.")
''' Move Files ------------------------------------------------------------------------------------------------------------------ '''

''' Extract RGB ----------------------------------------------------------------------------------------------------------------- '''
def extractRGB(opt):
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.
    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:
    [train|test], class, filename, nb frames
    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    folders = ['Train/', 'Test/']
    # Check if a RGB folder exists
    folder_RGB = os.path.join(opt.ucf101_rgb_dir)
    if not os.path.exists(folder_RGB):
        print("Creating folder for %s ..." % folder_RGB)
        os.mkdir(folder_RGB)
    for folder in folders:  # For Train | Test
        # Check if a Train | Test folder exists
        folder_train_test = os.path.join(opt.ucf101_rgb_dir, folder)
        if not os.path.exists(folder_train_test):  # UCF101/RGB/Train|Test/
            print("Creating folder for %s ..." % folder_train_test)
            os.mkdir(folder_train_test)
        # Get a list of folders  (glob: dir(), ls())
        class_folders = glob.glob(os.path.join(opt.ucf101_raw_dir, folder) + '*')  # ['Dataset/UCF101/RawData/Train/BandMarching', ...]
        for vid_class in class_folders:  # For each class,
            # Check if a class folder exists
            folder_class = os.path.join(opt.ucf101_rgb_dir, folder, vid_class.split('/')[-1])
            if not os.path.exists(folder_class):  # UCF101/RGB/Train|Test/vid_class/
                print("Creating folder for %s ..." % folder_class)
                os.mkdir(folder_class)
            # Get a list of avi files
            class_files = glob.glob(vid_class + '/*.avi')  # ['Dataset/UCF101/RawData/Train/BandMarching/v_BandMarching_g23_c01.avi', ...]
            for video_path in class_files:  # For each video file,
                # Get the parts of the file.
                video_parts = getVideoParts(video_path)  # [Train|Test], Class, FileName_wo_ext, FileName
                train_or_test, classname, filename_no_ext, filename = video_parts
                # Only extract if we haven't done it yet. Otherwise, just get the info.
                if not checkAlreadyExtractedRGB(opt, video_parts):
                    # Now extract it.
                    src  = os.path.join(opt.ucf101_raw_dir, train_or_test, classname, filename)  # video
                    dest = os.path.join(opt.ucf101_rgb_dir, train_or_test, classname, filename_no_ext + '-%06d.jpg')  # rgb
                    call(["ffmpeg", "-i", src, dest])  # call(): eval() in MATLAB
                # Now get how many frames it is.
                nb_frames = getNFrameforVideoRGB(opt, video_parts)
                data_file.append([train_or_test, classname, filename_no_ext, nb_frames])
                print("Generated %d RGB frames for %s" % (nb_frames, filename_no_ext))

    with open(os.path.join(opt.ucf101_data_dir, 'data_ucf101.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data_file)
    print("Extracted and wrote %d video files." % (len(data_file)))

def getVideoParts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split('/')
    filename = parts[-1]
    filename_no_ext = filename.split('.')[0]
    classname = parts[-2]
    train_or_test = parts[-3]
    return train_or_test, classname, filename_no_ext, filename

def checkAlreadyExtractedRGB(opt, video_parts):
    """Check to see if we created the -000001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    fileName = os.path.join(opt.ucf101_rgb_dir, train_or_test, classname, filename_no_ext + '-000001.jpg')
    return bool(os.path.exists(fileName))

def getNFrameforVideoRGB(opt, video_parts):
    """Given video parts of an (assumed) already extracted video, return the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    fileName = os.path.join(opt.ucf101_rgb_dir, train_or_test, classname, filename_no_ext + '*.jpg')
    generated_files = glob.glob(fileName)
    return len(generated_files)
''' Extract RGB ----------------------------------------------------------------------------------------------------------------- '''

''' Extract Optical Flow -------------------------------------------------------------------------------------------------------- '''
def extractOpticalFlow(opt):
    folders = ['Train/', 'Test/']
    # Check if a Flow folder exists
    folder_Flow = os.path.join(opt.ucf101_flow_dir)
    if not os.path.exists(folder_Flow):
        print("Creating folder for %s ..." % folder_Flow)
        os.mkdir(folder_Flow)
    for folder in folders:  # For Train | Test
        # Check if a Optical Flow folder exists
        folder_train_test = os.path.join(opt.ucf101_flow_dir, folder)
        if not os.path.exists(folder_train_test):  # UCF101/Flow/Train|Test/
            print("Creating folder for %s ..." % folder_train_test)
            os.mkdir(folder_train_test)
        # Get a list of folders  (glob: dir(), ls())
        class_folders = glob.glob(os.path.join(opt.ucf101_raw_dir, folder) + '*')  # ['Dataset/UCF101/RawData/Train/BandMarching', ...]
        for vid_class in class_folders:  # For each class,
            # Check if a class folder exists
            folder_class = os.path.join(opt.ucf101_flow_dir, folder, vid_class.split('/')[-1])
            if not os.path.exists(folder_class):  # UCF101/Flow/Train|Test/vid_class/
                print("Creating folder for %s ..." % folder_class)
                os.mkdir(folder_class)
            # Get a list of avi files
            class_files = glob.glob(vid_class + '/*.avi')  # ['Dataset/UCF101/RawData/Train/BandMarching/v_BandMarching_g23_c01.avi', ...]
            for video_path in class_files:  # For each video file,
                # Get the parts of the file.
                video_parts = getVideoParts(video_path)  # [Train|Test], Class, FileName_wo_ext, FileName
                train_or_test, classname, filename_no_ext, filename = video_parts
                # Only extract if we haven't done it yet. Otherwise, just get the info.
                if not checkAlreadyExtractedFlow(opt, video_parts):
                    saveOpticalFlow(opt, video_parts)
                # Now get how many frames it is.
                nb_frames = getNFrameforVideoFlow(opt, video_parts)
                print("Generated %d Optical Flow frames for %s" % (nb_frames, filename_no_ext))

def saveOpticalFlow(opt, video_parts):
    # Initialize Variable
    idx = 1
    train_or_test, classname, filename_no_ext, filename = video_parts
    # Read a video
    video_file_path = os.path.join(opt.ucf101_raw_dir, train_or_test, classname, filename)
    cap = cv2.VideoCapture(video_file_path)
    ret, frame1 = cap.read()
    if ret:
        prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        while 1:
            # Read a Frame
            ret, frame2 = cap.read()
            if frame2 is None:
                break
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # Calculate Optical Flow
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # parameters from http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
            '''
             - prev             first 8-bit single-channel input image.
             - next             second input image of the same size and the same type as prev.
             - flow             computed flow image that has the same size as prev and type CV_32FC2.
             - pyr_scale        parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
             - levels           number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
             - winsize          averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
             - iterations       number of iterations the algorithm does at each pyramid level.
             - poly_n           size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n = 5 or 7.
             - poly_sigma       standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
             - flags
            '''
            # Normalize Optical Flow (Charades Readme.txt)
            flow_normalized = 255.0 * (flow - opt.ucf101_flow_low_bound) / float(opt.ucf101_flow_high_bound - opt.ucf101_flow_low_bound)
            # Save the Optical Flow to Files
            file_name_x = os.path.join(opt.ucf101_flow_dir, train_or_test, classname, filename_no_ext + '-%06dx.jpg' % idx)
            file_name_y = os.path.join(opt.ucf101_flow_dir, train_or_test, classname, filename_no_ext + '-%06dy.jpg' % idx)
            cv2.imwrite(file_name_x, flow_normalized[:, :, 0])
            cv2.imwrite(file_name_y, flow_normalized[:, :, 1])
            # Update Variables
            idx += 1
            prev = next
    else:
        print("Can't find the video, %s. Skipping." % video_file_path)
    # Release the Video
    cap.release()

def checkAlreadyExtractedFlow(opt, video_parts):
    """Check to see if we created the -000001x frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    fileName = os.path.join(opt.ucf101_flow_dir, train_or_test, classname, filename_no_ext + '-000001x.jpg')
    return bool(os.path.exists(fileName))

def getNFrameforVideoFlow(opt, video_parts):
    """Given video parts of an (assumed) already extracted video, return the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    fileName = os.path.join(opt.ucf101_flow_dir, train_or_test, classname, filename_no_ext + '*.jpg')
    generated_files = glob.glob(fileName)
    return len(generated_files)
''' Extract Optical Flow -------------------------------------------------------------------------------------------------------- '''

if __name__ == "__main__":
    # Parse Arguments
    opt = parseConfiguration()
    # Move Files
    moveFiles(opt)
    # Extract RGB Images from Videos and Build new image files that we can use as our RGB data input files.
    extractRGB(opt)
    # Extract Optical Flow Images from Videos and Build new image files that we can use as our optical flow data input files
    extractOpticalFlow(opt)
    # LMDB? Need to Caffe?
    print('Done.')
