from flask import Flask, request, jsonify, render_template
import os
import cv2

from main import wykrywanie

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    # Renderuj plik index.html
    return render_template('index.html')

@app.route('/analyse_frame', methods=['POST'])
def analyse_frame():
    video_file = request.files.get('video')
    photo_files = request.files.getlist('photos')  # Pobierz wszystkie pliki zdjęć
    frame_number = request.form['frame_number']

    if video_file:
        # Zapis wideo
        video_path = os.path.join(os.getcwd(),"main/uploads", video_file.filename)
        video_file.save(video_path)

        # Zapis zdjęć
        photo_paths = []
        iterator = 0

        for photo in photo_files:
            photo_path = os.path.join(os.getcwd(), "main/baza_twarzy", "osoba" + str(iterator) + ".jpg")
            iterator += 1
            photo.save(photo_path)
            photo_paths.append(photo_path)

        # Analiza wideo (np. klatka)
        video = cv2.VideoCapture(video_path)
        print(frame_number)
        video.set(cv2.CAP_PROP_POS_MSEC, int(float(frame_number)*1000))
        ret, frame = video.read()
        cv2.imwrite(os.path.join(os.getcwd(), "main/tymczasowe", "obraz.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        video.release()

        if not ret:
            return jsonify({'error': 'Nie udało się odczytać klatki'})

        wykrywanie(czy_wyciac_twarze = True)

        return jsonify({
            'message': 'Analiza zakończona pomyślnie',
            'frame': frame_number,
            'saved_image': f'/uploads/',
            'uploaded_photos': photo_paths  # Ścieżki do zapisanych zdjęć
        })

    return jsonify({'error': 'Nie przesłano pliku wideo'})

if __name__ == '__main__':
    app.run(debug=True)