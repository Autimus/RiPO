import sys
import logging
import yaml
import cv2
import torch
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.show import show_bchw
from utils.draw import draw_bchw
from core.model_loader.face_parsing.FaceParsingModelLoader import FaceParsingModelLoader
from core.model_handler.face_parsing.FaceParsingModelHandler import FaceParsingModelHandler
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

# Inicjalizacja logowania
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

# Ładowanie konfiguracji modelu
with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    model_path = 'models'

    # Ładowanie modelu detekcji twarzy
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]
    logger.info('Start to load the face detection model...')

    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face detection Model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # Ładowanie modelu alignowania twarzy
    model_category = 'face_alignment'
    model_name = model_conf[scene][model_category]
    logger.info('Start to load the face landmark model...')

    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face landmark model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # Ładowanie modelu segmentacji twarzy
    scene = 'non-mask'
    model_category = 'face_parsing'
    model_name = model_conf[scene][model_category]
    logger.info('Start to load the face parsing model...')

    try:
        faceParsingModelLoader = FaceParsingModelLoader(model_path, model_category, model_name)
        model, cfg = faceParsingModelLoader.load_model()
        faceParsingModelHandler = FaceParsingModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face parsing Model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # Przetwarzanie obrazu
    image_path = 'api_usage/test_images/test1.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    try:
        dets = faceDetModelHandler.inference_on_image(image)
        face_nums = dets.shape[0]
        landmarks_five = None

        with torch.no_grad():
            for i in range(face_nums):
                landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])
                landmarks = torch.from_numpy(landmarks[[104, 105, 54, 84, 90]]).float()

                if landmarks_five is None:
                    landmarks_five = landmarks
                else:
                    landmarks_five = torch.stack([landmarks_five, landmarks], dim=0)

            print(landmarks_five.shape)
            faces = faceParsingModelHandler.inference_on_image(face_nums, image, landmarks_five)
            seg_logits = faces['seg']['logits']
            seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

            # Sprawdzenie formatu obrazu
            if image.shape[2] == 3:  # Jeśli obraz jest w formacie BGR, przekształć na RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Rysowanie twarzy na obrazie
            image_with_faces = draw_bchw(image_rgb, faces)
            print(f"Image with faces: {image_with_faces.shape}")

            # Wyświetlanie obrazu
            show_bchw(image_with_faces)

    except Exception as e:
        logger.error('Parsing failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')
