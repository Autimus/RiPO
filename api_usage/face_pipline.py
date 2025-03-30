"""
@author: JiXuan Xu, Jun Wang
@date: 20201024
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

import yaml
import cv2
import numpy as np
import torch
import re
import glob
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler


def runD(image_path = 'main/tymczasowe', resoult_folder= 'main/wyniki/dopasowania.txt'):
    with open('config/model_conf.yaml') as f:
        model_conf = yaml.load(f, Loader=yaml.SafeLoader)
    # common setting for all models, need not modify.
    model_path = 'models'

    torch.nn.Module.dump_patches = True
    # face detection model setting.
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name =  model_conf[scene][model_category]
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Falied to load face detection Model.')
        logger.error(e)
        sys.exit(-1)

    # face landmark model setting.
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face landmark model.')
        logger.error(e)
        sys.exit(-1)

    # face recognition model setting.
    model_category = 'face_recognition'
    model_name =  model_conf[scene][model_category]
    try:
        faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
        model, cfg = faceRecModelLoader.load_model()
        faceRecModelHandler = FaceRecModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face recognition model.')
        logger.error(e)
        sys.exit(-1)

    # TODO: tu zmienilem
    wynik = []
    for image_path in glob.glob(str(image_path)+"/*porownanie*.jpg"):
        numbers = re.findall(r"\d+\.\d+|\d+", image_path)
        # read image and get face features.
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        face_cropper = FaceRecImageCropper()
        try:
            dets = faceDetModelHandler.inference_on_image(image)
            face_nums = dets.shape[0]
            if face_nums != 2:
                logger.info('Input image should contain two faces to compute similarity!')
            feature_list = []
            for i in range(face_nums):
                landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])
                landmarks_list = []
                for (x, y) in landmarks.astype(np.int32):
                    landmarks_list.extend((x, y))
                cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
                feature = faceRecModelHandler.inference_on_image(cropped_image)
                feature_list.append(feature)

            wynik.append([int(numbers[0]),float(np.dot(feature_list[0], feature_list[1])),int(numbers[1])])
        except Exception as e:
            logger.error('Pipeline failed!')
            logger.error(e)
            sys.exit(-1)
    # Słownik przechowujący najlepsze wartości dla każdego numer1
    best_values = {}

    for numer1, value, numer2 in wynik:
        # Jeśli numer1 nie jest w słowniku lub znaleźliśmy większą wartość, aktualizujemy
        if numer1 not in best_values or value > best_values[numer1][1]:
            best_values[numer1] = [numer1, value, numer2]

    # Konwersja do listy
    result = list(best_values.values())
    print(result)
    with open(resoult_folder, "w") as fd:
        fd.write("TwarzZeZdjecia\tDopasowanie\tOsobaZBazy\n")
        for line in result:
            fd.write("\t".join(map(str,line))+"\n")


