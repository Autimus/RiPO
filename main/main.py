import re
import sys

sys.path.append('')

import os
from api_usage.face_detect import runA
from api_usage.face_aligment import runB
from api_usage.face_crop import runC
from filmy.wycinanie_z_filmu import stop_klatka
from api_usage.face_pipline import runD
from baza_twarzy.wytnij_twarze import wytnijTwarzeBazy
from PIL import Image
import glob

# Ścieżka do katalogu
folder = 'main/tymczasowe'
image_path = 'main/tymczasowe/obraz.jpg'
landmarks_name = "twarz"
baza_twarzy = "main/baza_twarzy"

# Iteruj przez wszystkie pliki w katalogu
for filename in os.listdir(os.getcwd()+"/"+folder):
    file_path = os.path.join(folder, filename)
    # Sprawdź, czy to plik (a nie katalog)
    if os.path.isfile(file_path):
        os.remove(file_path)

#TODO: używasz tego gdy nie masz pliku 'obraz.jpg' w main/tymczasowe (wymaga filmów)
stop_klatka()

#TODO: uzywasz tego gdy nie masz plików 'twarz*.jpg' w main/baza_twarzy
wytnijTwarzeBazy()

runA()
runB()
runC()

regex = r"(\d+(\.\d+)?)"
for sciezka1 in glob.glob(str(folder)+"/"+str(landmarks_name)+"*.jpg"):
    znalezionaTwarz = Image.open(sciezka1)  # Podmień na rzeczywistą nazwę pliku

    for sciezka2 in glob.glob(str(baza_twarzy) + "/" + str(landmarks_name) + "*.jpg"):
        twarzZBazy = Image.open(sciezka2)  # Podmień na rzeczywistą nazwę pliku

        # Upewnij się, że mają poprawny rozmiar 112x112
        znalezionaTwarz = znalezionaTwarz.resize((112, 112))
        twarzZBazy = twarzZBazy.resize((112, 112))

        # Tworzymy nowy obraz o rozmiarze 224x112
        new_img = Image.new("RGB", (224, 112))

        # Wklejamy obrazy obok siebie
        new_img.paste(znalezionaTwarz, (0, 0))
        new_img.paste(twarzZBazy, (112, 0))

        # Zapisujemy wynik
        new_img.save(str(folder)+"/"+re.search(regex,sciezka1).group(1)+"porownanie"+re.search(regex,sciezka2).group(1)+".jpg")

runD()