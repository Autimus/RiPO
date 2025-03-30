
import sys
sys.path.append('')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')


import numpy as np

from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader

import re
import yaml
import cv2
import torch
import glob
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

def wytnijTwarzeBazy():
    #face_DETECT
    resoult_path = 'main/baza_twarzy'
    for image_path in glob.glob(str(resoult_path)+"/osoba*.jpg"):
        with open('config/model_conf.yaml') as f:
            model_conf = yaml.load(f, Loader=yaml.SafeLoader)

        iterator = re.search(r"(\d+(\.\d+)?)", image_path).group(1)
        # common setting for all model, need not modify.
        model_path = 'models'

        torch.nn.Module.dump_patches = True

        # model setting, modified along with model
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name =  model_conf[scene][model_category]

        # load model
        try:
            faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        except Exception as e:
            logger.error('Failed to parse model configuration file!')
            logger.error(e)
            sys.exit(-1)

        try:
            model, cfg = faceDetModelLoader.load_model()
        except Exception as e:
            logger.error('Model loading failed!')
            logger.error(e)
            sys.exit(-1)

        # read image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)

        try:
            dets = faceDetModelHandler.inference_on_image(image)
        except Exception as e:
           logger.error('Face detection failed!')
           logger.error(e)
           sys.exit(-1)


    #face_ALIGMENT
        landmarks_name = "twarz"
        with open('config/model_conf.yaml') as f:
            model_conf = yaml.load(f, Loader=yaml.SafeLoader)

        # common setting for all model, need not modify.
        model_path = 'models'
        torch.nn.Module.dump_patches = True;

        # model setting, modified along with model
        scene = 'non-mask'
        model_category = 'face_alignment'
        model_name =  model_conf[scene][model_category]

        # load model
        try:
            faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        except Exception as e:
            logger.error('Failed to parse model configuration file!')
            logger.error(e)
            sys.exit(-1)

        try:
            model, cfg = faceAlignModelLoader.load_model()
        except Exception as e:
            logger.error('Model loading failed!')
            logger.error(e)
            sys.exit(-1)

        faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        line = dets[0]
        det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
        landmarks = faceAlignModelHandler.inference_on_image(image, det)
        landmarks = landmarks.flatten().astype(np.int32).tolist()
        #face_CROP
        torch.nn.Module.dump_patches = True
        image_info_file = str(resoult_path) + "/" + str(landmarks_name) + str(iterator) + ".jpg"

        face_cropper = FaceRecImageCropper()
        image = cv2.imread(image_path)
        cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
        cv2.imwrite(image_info_file, cropped_image)
