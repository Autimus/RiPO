import sys
sys.path.append('')

from sympy import false
import re
import os
from api_usage.face_detect import runA
from api_usage.face_aligment import runB
from api_usage.face_crop import runC
from main.filmy.wycinanie_z_filmu import stop_klatka
from api_usage.face_pipline import runD
from main.baza_twarzy.wytnij_twarze import wytnijTwarzeBazy
from PIL import Image
import glob

def wykrywanie(
folder = 'main/tymczasowe',
image_path = 'main/tymczasowe/obraz.jpg',
landmarks_name = "twarz",
baza_twarzy = "main/baza_twarzy",
resoult_file='main/wyniki/okwadratowane.jpg',
resoult_file2= 'main/wyniki/dopasowania.txt',
czy_wycinac_wiedo = False,
film = "main/filmy/film2.MOV",
milisekunda=85500,
docelowe="main/tymczasowe/obraz.jpg",
czy_wyciac_twarze = False):

    # Iteruj przez wszystkie pliki w katalogu
    for filename in os.listdir(os.getcwd()+"/"+folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and not filename == "obraz.jpg":
            os.remove(file_path)

    if czy_wycinac_wiedo:
    # używasz tego gdy nie masz pliku 'obraz.jpg' w main/tymczasowe (wymaga filmów)
        stop_klatka(film = film, milisekunda=milisekunda,docelowe=docelowe)

    if czy_wyciac_twarze:
        # uzywasz tego gdy nie masz plików 'twarz*.jpg' w main/baza_twarzy
        wytnijTwarzeBazy()

    runA(resoult_path=folder,image_path=image_path, resoult_file=resoult_file)
    runB(resoult_path=folder,image_path=image_path,landmarks_name=landmarks_name)
    runC(resoult_path=folder,image_path=image_path,landmarks_name=landmarks_name)

    regex = r"(\d+(\.\d+)?)"
    for sciezka1 in glob.glob(str(folder)+"/"+str(landmarks_name)+"*.jpg"):
        znalezionaTwarz = Image.open(sciezka1)

        for sciezka2 in glob.glob(str(baza_twarzy) + "/" + str(landmarks_name) + "*.jpg"):
            twarzZBazy = Image.open(sciezka2)

            znalezionaTwarz = znalezionaTwarz.resize((112, 112))
            twarzZBazy = twarzZBazy.resize((112, 112))

            new_img = Image.new("RGB", (224, 112))

            new_img.paste(znalezionaTwarz, (0, 0))
            new_img.paste(twarzZBazy, (112, 0))

            new_img.save(str(folder)+"/"+re.search(regex,sciezka1).group(1)+"porownanie"+re.search(regex,sciezka2).group(1)+".jpg")

    runD(resoult_path=folder,resoult_file=resoult_file2)