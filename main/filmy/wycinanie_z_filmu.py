import cv2

def stop_klatka(film = "main/filmy/film3.MOV",milisekunda=57000,docelowe="main/tymczasowe/obraz.jpg"):
    # Otwórz plik wideo
    video = cv2.VideoCapture(film)

    # Przejdź do 5 sekundy
    video.set(cv2.CAP_PROP_POS_MSEC, milisekunda)

    # Odczytaj klatkę
    success, frame = video.read()

    if success:
        # Zapisz w najwyższej jakości
        cv2.imwrite(docelowe, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # Zamknij plik wideo
    video.release()
