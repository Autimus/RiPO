import cv2
def stop_klatka(film = "main/filmy/film3.MOV",milisekunda=57000,docelowe="main/tymczasowe/obraz.jpg"):
    video = cv2.VideoCapture(film)
    video.set(cv2.CAP_PROP_POS_MSEC, milisekunda)
    success, frame = video.read()
    if success:
        cv2.imwrite(docelowe, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    video.release()