"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import sys
sys.path.append('')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
import cv2
import torch
import glob

from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

def runC(resoult_path = 'main/tymczasowe',image_path = 'main/tymczasowe/obraz.jpg',landmarks_name = "twarz"):
    torch.nn.Module.dump_patches = True

    file_count = len(glob.glob(str(resoult_path)+"/"+str(landmarks_name)+"*.txt"))
    for i in range(file_count):
        image_info_file = str(resoult_path)+"/"+str(landmarks_name)+str(i)+".txt"
        line = open(image_info_file).readline().strip()
        landmarks_str = line.split(' ')
        landmarks = [float(num) for num in landmarks_str]
        face_cropper = FaceRecImageCropper()
        image = cv2.imread(image_path)
        cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
        cv2.imwrite(str(resoult_path)+"/"+str(landmarks_name)+str(i)+".jpg", cropped_image)
        #logger.info('Crop image successful!')
