from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
import cv2

from flask_cors import CORS
from main import wykrywanie

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

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
        #Zapis wideo
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

        # Analiza wideo (klatka)
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_MSEC, int(float(frame_number)*1000))
        ret, frame = video.read()
        cv2.imwrite(os.path.join(os.getcwd(), "main/tymczasowe", "obraz.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        video.release()

        if not ret:
            return jsonify({'error': 'Nie udało się odczytać klatki'})

        wykrywanie(czy_wyciac_twarze = True)

        # Wczytaj dane dopasowania
        matching_results = []
        with open(os.path.join(os.getcwd(),"main/wyniki",'dopasowania.txt'), 'r') as file:
            for line in file:
                columns = line.strip().split('\t')
                if columns[1]=="Dopasowanie":
                    continue

                matching_results.append({
                    'photo_face': url_for('serve_temporary', filename=f"twarz{columns[0]}.jpg", _external=True),
                    'similarity': float(columns[1]),
                    'database_person': url_for('serve_database', filename=f"twarz{columns[2]}.jpg", _external=True)
                })

        return jsonify({
            'message': 'Analiza zakończona pomyślnie',
            'matching_results': matching_results  # Dopasowania z podobieństwem
        })
    return jsonify({'error': 'Nie przesłano pliku wideo'})

@app.route('/main/tymczasowe/<path:filename>')
def serve_temporary(filename):
    return send_from_directory(os.path.abspath("main/tymczasowe"), filename)

@app.route('/main/baza_twarzy/<path:filename>')
def serve_database(filename):
    return send_from_directory(os.path.abspath("main/baza_twarzy"), filename)


@app.route('/main/wyniki/<path:filename>')
def serve_wyniki(filename):
    return send_from_directory(os.path.abspath("main/wyniki"), filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)