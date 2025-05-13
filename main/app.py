from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
import cv2
import shutil

from flask_cors import CORS
from main import wykrywanie
from main.functionality.wytnij_twarze import wytnijTwarzeBazy

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


@app.route('/upload_photos', methods=['POST'])
def upload_photos():
    pth = "main/baza_twarzy"
    photo_files = request.files.getlist("photos")
    os.makedirs(pth, exist_ok=True)

    for idx, photo in enumerate(photo_files, start=1):
        filename = f"osoba{idx}.jpg"
        save_path = os.path.join(pth, filename)
        photo.save(save_path)

    wytnijTwarzeBazy(pth)

    return jsonify({"status": "success", "message": "Zdjęcia zapisane"})



@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files.get("video")
    os.makedirs("main/uploads", exist_ok=True)

    if video_file:
        video_path = os.path.join("main/uploads", video_file.filename)
        video_file.save(video_path)
        return jsonify({"status": "success", "filename": video_file.filename})

    return jsonify({"status": "error", "message": "Brak pliku"}), 400



@app.route('/upload_image', methods=['POST'])
def upload_image():
    image = request.files.get("image")
    os.makedirs("main/uploads", exist_ok=True)

    if image:
        image.save(os.path.join("main/uploads", image.filename))
        return jsonify({"status": "success", "filename": image.filename})
    return jsonify({"status": "error", "message": "Brak pliku"}), 400




@app.route('/analyse_frame', methods=['POST'])
def analyse_frame():
    threshold = request.form.get('threshold', default=20, type=float)
    modelValue = request.form.get('model')

    uploads_dir = os.path.join("main", "uploads")
    if not os.path.exists(uploads_dir):
        return jsonify({'error': 'Brak katalogu z plikami'}), 500

    # Znajdź ostatnio przesłane zdjęcie
    image_files = sorted(
        [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: os.path.getmtime(os.path.join(uploads_dir, x)),
        reverse=True
    )
    if not image_files:
        return jsonify({'error': 'Brak przesłanych zdjęć do analizy'}), 404
    latest_image = image_files[0]
    source_path = os.path.join(uploads_dir, latest_image)

    # Skopiuj/Przenieś do katalogu tymczasowego analizy
    temp_dir = os.path.join("main", "tymczasowe")
    os.makedirs(temp_dir, exist_ok=True)
    analysis_path = os.path.join(temp_dir, "obraz.jpg")
    shutil.copy2(source_path, analysis_path)  # lub move jeśli nie chcesz zostawiać kopii

    # Uruchom analizę (detekcja i porównanie)
    wykrywanie(czy_wyciac_twarze=False,czy_wycinac_wiedo=False, prog=threshold,selected_model=modelValue)

    # Wczytaj dane dopasowania
    results = []
    results_path = os.path.join("main", "wyniki", "dopasowania.txt")
    if not os.path.exists(results_path):
        return jsonify({'error': 'Brak pliku wyników dopasowania'}), 500

    with open(results_path, 'r') as file:
        for line in file:
            cols = line.strip().split('\t')
            if cols[1] == "Dopasowanie":
                continue

            similarity = float(cols[1])
            if similarity >= threshold:
                results.append({
                    'photo_face': url_for('serve_temporary', filename=f"twarz{cols[0]}.jpg", _external=True),
                    'similarity': similarity,
                    'database_person': url_for('serve_database', filename=f"twarz{cols[2]}.jpg", _external=True)
                })

    return jsonify({
        'message': 'Analiza zakończona pomyślnie',
        'matching_results': results
    })

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