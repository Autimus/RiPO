"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com
"""
import sys
sys.path.append('')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import yaml
import cv2
import numpy as np
import torch
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler


def runB(resoult_path = 'main/tymczasowe',image_path = 'main/tymczasowe/obraz.jpg',landmarks_name = "twarz"):

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

    # read image
    image_det_txt_path = f'{resoult_path}/lista_twarzy.txt'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    with open(image_det_txt_path, 'r') as f:
        lines = f.readlines()
    try:
        for i, line in enumerate(lines):
            line = line.strip().split()
            det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
            landmarks = faceAlignModelHandler.inference_on_image(image, det)

            #save_path_img = 'api_usage/temp/test1_' + 'landmark_res' + str(i) + '.jpg'
            save_path_txt = str(resoult_path) + "/" + str(landmarks_name) + str(i) + '.txt'
            image_show = image.copy()
            with open(save_path_txt, "w") as fd:
                for (x, y) in landmarks.astype(np.int32):
                    cv2.circle(image_show, (x, y), 2, (255, 0, 0),-1)
                    line = str(x) + ' ' + str(y) + ' '
                    fd.write(line)
            #cv2.imwrite(save_path_img, image_show)
    except Exception as e:
        logger.error('Face landmark failed!')
        logger.error(e)
        sys.exit(-1)