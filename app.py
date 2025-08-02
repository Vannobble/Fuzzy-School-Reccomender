from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
from flask import Flask, render_template, request, redirect, url_for


app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data.csv')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
IMAGES_FOLDER = os.path.join(STATIC_FOLDER, 'images')

# Create images directory if it doesn't exist
os.makedirs(IMAGES_FOLDER, exist_ok=True)

IMAGE_MAP = {
    'SMA Negeri 1 Kepanjen': 'SMAN 1 KEPANJEN.jpg',
    'SMK Negeri 1 Turen': 'SMKN 1 TUREN.jpg',
    'MAN 1 Malang': 'MAN1 KOTA MALANG.jpg',
    'MAN 2 Malang': 'MAN2 KOTA MALANG.jpg',
    'SMA Negeri 1 Tumpang': 'SMAN 1 TUMPANG.jpg',
    'SMA Negeri 1 Malang': 'SMAN 1 MALANG.jpg',
    'SMA Negeri 8 Malang': 'SMAN 8 MALANG.jpg',
    'SMA Negeri 9 Malang': 'SMAN 9 MALANG.jpg',
    'SD Negeri 2 Turen': 'SDN 2 TUREN.jpg',
    'SMP Negeri 15 Kota Malang': 'SMPN 15 MALANG.jpg',
    'SMK Negeri 4 Malang': 'SMAN 4 MALANGpg.webp',
    'SMA Katolik St. Albertus': 'SMAK ST ALBERTUS.jpg',
    'SMP Negeri 1 Malang': 'SMPN 1 MALANG.jpg',
    'SMA Negeri 3 Malang': 'SMAN 3 MALANG.jpg',
    'SMA Katolik Kolese Santo Yusup': 'SMAK KOLESE SANTO YUSUF.jpg',
    'SMP Negeri 8 Malang': 'SMPN 8 MALANG.jpg',
    'SMP Negeri 3 Malang': 'SMPN 3 MALANG.jpg',
    'SD Negeri 2 Wandanpuro': 'SDN 2 WANDANPURO.png',
    'SMA Negeri 5 Malang': 'SMAN 5 MALANG.jpg',
    'SMA Negeri 4 Malang': 'SMKN 4 MALANG.jpg',
    'SMA Negeri 10 Malang': 'SMAN 10 MALANG.jpg',
    'SD Negeri Bunulrejo 1': 'SD NEGERI BUNULREJO.jpg',
    'SMK Negeri 3 Malang': 'SMKN 3 MALANG.jpg',
    'MAN 3 Malang': 'MAN3 MALANG.jpg',
    'SD Muhammadiyah 01 Malang': 'SD MUHAMMADIYAH 1 MALANG.jpg',
    'SD Negeri Rampal Celaket 1': 'SDN RAMPAL CELAKET 1 MALANG.jpg',
    'SMK Negeri 5 Malang': 'SMKN 5 MALANG.jpg',
    'SMK Negeri 2 Singosari': 'SMKN 2 SINGOSARI.jpg',
    'SMP Negeri 2 Turen': 'SMPN 2 Turen.jpg'
}

def hitung_jarak(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Simpan versi asli (untuk ditampilkan ke user)
    df['fasilitas'] = df['fasilitas'].apply(lambda x: [f.strip() for f in x.split(',')])
    df['ekskul'] = df['ekskul'].apply(lambda x: [e.strip() for e in x.split(',')])

    # Buat versi lowercase (untuk pencocokan)
    df['fasilitas_match'] = df['fasilitas'].apply(lambda x: [f.lower() for f in x])
    df['ekskul_match'] = df['ekskul'].apply(lambda x: [e.lower() for e in x])

    return df

def calculate_fuzzy_score(jarak_km, fasilitas_match, ekskul_match, pref_fasilitas, pref_ekskul, pref_jarak):
    # Jika semua preferensi nol, skor = 0
    if pref_fasilitas == 0 and pref_ekskul == 0 and pref_jarak == 0:
        return 0

    # Normalisasi match ke 0-10
    fasilitas_input = fasilitas_match * 10
    ekskul_input = ekskul_match * 10
    jarak_input = jarak_km  # dalam km

    # Tetapkan sekolah
    fasilitas = ctrl.Antecedent(np.arange(0, 11, 1), 'fasilitas')
    ekskul = ctrl.Antecedent(np.arange(0, 11, 1), 'ekskul')
    jarak = ctrl.Antecedent(np.arange(0, 6, 0.1), 'jarak')
    skor = ctrl.Consequent(np.arange(0, 101, 1), 'skor')

    # Membership functions
    fasilitas.automf(3)
    ekskul.automf(3)
    jarak['poor'] = fuzz.trimf(jarak.universe, [2.5, 5, 5])
    jarak['average'] = fuzz.trimf(jarak.universe, [1.5, 2.5, 3.5])
    jarak['good'] = fuzz.trimf(jarak.universe, [0, 0, 2.5])
    skor['rendah'] = fuzz.trimf(skor.universe, [0, 0, 40])
    skor['sedang'] = fuzz.trimf(skor.universe, [35, 60, 85])
    skor['tinggi'] = fuzz.trimf(skor.universe, [80, 100, 100])

    rules = []
    if pref_fasilitas > 0:
        rules.append(ctrl.Rule(fasilitas['good'], skor['tinggi']))
        rules.append(ctrl.Rule(fasilitas['average'], skor['sedang']))
        rules.append(ctrl.Rule(fasilitas['poor'], skor['rendah']))

    if pref_ekskul > 0:
        rules.append(ctrl.Rule(ekskul['good'], skor['tinggi']))
        rules.append(ctrl.Rule(ekskul['average'], skor['sedang']))
        rules.append(ctrl.Rule(ekskul['poor'], skor['rendah']))

    if pref_jarak > 0:
        rules.append(ctrl.Rule(jarak['good'], skor['tinggi']))
        rules.append(ctrl.Rule(jarak['average'], skor['sedang']))
        rules.append(ctrl.Rule(jarak['poor'], skor['rendah']))

    scoring_ctrl = ctrl.ControlSystem(rules)
    scoring = ctrl.ControlSystemSimulation(scoring_ctrl)

    if pref_fasilitas > 0:
        scoring.input['fasilitas'] = fasilitas_input
    if pref_ekskul > 0:
        scoring.input['ekskul'] = ekskul_input
    if pref_jarak > 0:
        scoring.input['jarak'] = jarak_input

    scoring.compute()
    return round(scoring.output['skor'], 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.get_json()

    # Ambil input dengan default
    lat = float(user_input.get('lat', 0))
    lon = float(user_input.get('lon', 0))
    fasilitas_user = [f.strip().lower() for f in user_input.get('fasilitas', [])]
    ekskul_user = [e.strip().lower() for e in user_input.get('ekskul', [])]
    pref_fasilitas = int(user_input.get('pref_fasilitas', 0))
    pref_ekskul = int(user_input.get('pref_ekskul', 0))
    pref_jarak = int(user_input.get('pref_jarak', 0))
    jenjang = user_input.get('jenjang', '').strip().lower()

    # ======== VALIDASI PENTING SESUAI INSTRUKSI ========

    # 1. Jenis sekolah wajib diisi
    if not jenjang:
        return jsonify({'error': 'Pilih jenis sekolah terlebih dahulu!'}), 400

    # 2. Preferensi fasilitas diberikan tapi tidak mengisi fasilitas
    if pref_fasilitas > 0 and not fasilitas_user:
        return jsonify({'error': 'Silakan isi fasilitas yang diinginkan jika preferensinya tidak nol.'}), 400

    # 3. Preferensi ekskul diberikan tapi tidak mengisi ekskul
    if pref_ekskul > 0 and not ekskul_user:
        return jsonify({'error': 'Silakan isi ekstrakurikuler yang diinginkan jika preferensinya tidak nol.'}), 400

    # 5. Jika semua preferensi nol → tidak masuk akal
    if pref_fasilitas == 0 and pref_ekskul == 0 and pref_jarak == 0:
        return jsonify({'error': 'Setidaknya satu preferensi harus memiliki nilai lebih dari 0.'}), 400

    # ========== PROSES REKOMENDASI ==========
    df = load_data()
    df_filtered = df[df['jenjang'].str.lower() == jenjang]

    hasil = []
    for _, row in df_filtered.iterrows():
        skor_fasilitas = len(set(row['fasilitas_match']).intersection(fasilitas_user)) / max(len(row['fasilitas_match']), 1)
        skor_ekskul = len(set(row['ekskul_match']).intersection(ekskul_user)) / max(len(row['ekskul_match']), 1)

        jarak = hitung_jarak(lat, lon, row['latitude'], row['longitude'])

        skor_total = calculate_fuzzy_score(
            jarak_km=jarak,
            fasilitas_match=skor_fasilitas,
            ekskul_match=skor_ekskul,
            pref_fasilitas=pref_fasilitas,
            pref_ekskul=pref_ekskul,
            pref_jarak=pref_jarak
        )

        bobot_total = pref_fasilitas + pref_ekskul + pref_jarak or 1
        bobot_fasilitas = pref_fasilitas / bobot_total
        bobot_ekskul = pref_ekskul / bobot_total
        bobot_jarak = pref_jarak / bobot_total

        skor_akhir = skor_total * (bobot_fasilitas + bobot_ekskul + bobot_jarak)

        # Check if image exists
        image_filename = IMAGE_MAP.get(row['nama'], '')
        image_path = os.path.join(STATIC_FOLDER, image_filename)
        has_image = os.path.exists(image_path) and image_filename

        hasil.append({
            "nama": row['nama'],
            "alamat": row['alamat'],
            "kecamatan": row['kecamatan'],
            "jenjang": row['jenjang'],
            "fasilitas": ", ".join(row['fasilitas']),
            "ekskul": ", ".join(row['ekskul']),
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "jarak_km": round(jarak, 2),
            "skor": round(skor_akhir, 2),
            "kekurangan": row.get('kekurangan', 'Tidak tersedia'),
            "image": image_filename if has_image else None
        })

    hasil_sorted = sorted(hasil, key=lambda x: x['skor'], reverse=True)
    session['results'] = hasil_sorted
    return jsonify(hasil_sorted)

@app.route('/results')
def results():
    results = session.get('results', [])
    return render_template('results.html', results=results, enumerate=enumerate)

@app.route('/detail/<int:id>')
def detail(id):
    results = session.get('results', [])
    if id < 0 or id >= len(results):
        return "Sekolah tidak ditemukan", 404
    
    school = results[id]
    
    # Cek gambar dengan cara yang lebih robust
    image_file = IMAGE_MAP.get(school['nama'], '')
    image_exists = False
    image_url = None
    
    if image_file:
        image_path = os.path.join(IMAGES_FOLDER, image_file)
        if os.path.exists(image_path):
            image_exists = True
            image_url = url_for('static', filename=f'images/{image_file}')
    
    # Tambahkan informasi gambar ke data sekolah
    school['has_image'] = image_exists
    school['image_url'] = image_url if image_exists else None
    
    return render_template('detail.html', school=school)

if __name__ == '__main__':
    app.run(debug=True)