---
title: Sign Language Detection
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🤟 ASL Hand Gesture Recognition — Notebook-Based ML Pipeline & Web App

Proyek ini membangun sistem **ASL (American Sign Language) alphabet recognition** dengan dua pendekatan utama:

1. **Baseline CNN berbasis citra** menggunakan **MobileNetV2**
2. **Model final berbasis hand landmarks** menggunakan **MediaPipe Hands + MLP**

Versi repository ini disusun mengikuti **alur notebook eksperimen (`.ipynb`)**, sehingga pembaca bisa memahami proses project secara bertahap: mulai dari konfigurasi, data loading, preprocessing, training, evaluasi, hingga inference dan implementasi web interaktif.

---

## 🎯 Tujuan Proyek

Tujuan utama project ini adalah membuat sistem pengenalan gesture tangan ASL yang:

- dapat mengklasifikasikan huruf dan label khusus ASL,
- bisa diuji pada dataset gambar,
- dapat diimplementasikan ke **web app interaktif**,
- dan dibandingkan antara pendekatan **image-based** dan **landmark-based**.

Output akhirnya adalah:

- **Notebook analysis / modeling** untuk dokumentasi eksperimen
- **Web app interaktif** untuk demo/penggunaan langsung
- **Artikel/penjelasan singkat** yang merangkum logic dan hasil model

---

## 🧠 Ringkasan Pendekatan

### 1) Baseline Image Model
Pendekatan awal menggunakan **RGB hand image classification**:

- input gambar di-resize ke **224 × 224**
- preprocessing dan augmentasi ringan
- model **MobileNetV2 transfer learning**
- output multi-class classification untuk label ASL

Model ini bekerja baik di data terstruktur, tetapi kurang stabil saat dipakai pada webcam real-world.

### 2) Final Landmark Model
Pendekatan final menggunakan **hand landmarks**:

- deteksi tangan dengan **MediaPipe Hand Landmarker**
- ekstraksi **21 titik landmark**
- normalisasi landmark
- penambahan fitur geometris
- klasifikasi dengan **MLPClassifier**

Pipeline ini dipilih untuk deployment karena lebih robust terhadap:

- background yang bervariasi
- perbedaan pengguna
- pencahayaan yang berubah
- input webcam real-time

---

## 🏷️ Label Kelas

Model mengenali label berikut:

- **Huruf:** `A-Z`
- **Label khusus:** `del`, `nothing`, `space`

Total kelas: **29**

---

## 📘 Struktur Utama Notebook

Notebook utama mengikuti alur eksperimen berikut:

### 1. Konfigurasi Global
Berisi pengaturan path dataset, path model, ukuran gambar, hyperparameter training, dan daftar kelas.

### 2. Utilities
Berisi helper seperti:

- set random seed
- memastikan direktori output tersedia
- simpan / load file JSON

### 3. Data Loading & EDA
Notebook membaca dataset dari folder train, membangun dataframe, lalu melakukan:

- train/validation split secara stratified
- analisis distribusi kelas
- audit data
- visualisasi contoh gambar

### 4. Preprocessing & tf.data Pipeline
Untuk baseline CNN, dilakukan:

- load file gambar
- decode JPG
- resize ke `224×224`
- normalisasi pixel ke `[0,1]`
- augmentasi ringan (brightness dan contrast)
- batching + prefetch dengan `tf.data`

### 5. Build & Compile CNN
Model baseline menggunakan:

- **MobileNetV2**
- base model dibekukan (`trainable=False`)
- `GlobalAveragePooling`
- `Dropout`
- output softmax

### 6. Training CNN
Notebook melakukan training model baseline dengan callback:

- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau

### 7. Evaluasi CNN
Hasil CNN dianalisis melalui:

- training history
- classification report
- confusion matrix

### 8. Hand ROI Detector
Notebook juga mendokumentasikan proses pendeteksian **ROI tangan** menggunakan MediaPipe sebagai tahap bantu untuk memperbaiki fokus area tangan.

### 9. Landmark Feature Extraction
Tahap ini merupakan inti pipeline final:

- ekstraksi 21 landmark tangan
- normalisasi koordinat
- penanganan handedness
- pembentukan fitur geometris tambahan

### 10. Training Landmark MLP
Notebook membangun dataset landmark, lalu melatih model MLP dengan:

- `StandardScaler`
- `MLPClassifier`

Selain landmark mentah, pipeline juga memasukkan:
- augmentasi koordinat
- sample sintetis untuk kelas `nothing`

### 11. Evaluasi Landmark MLP
Model final dievaluasi dengan:

- classification report
- confusion matrix
- loss curve

### 12. Inference
Notebook menyediakan dua jalur inference:

- **CNN inference**
- **Landmark MLP inference**

Termasuk top-k prediction dan logic threshold confidence.

### 13. Demo Prediksi
Disediakan demo prediksi untuk beberapa gambar validasi sebagai contoh hasil inferensi model.

### 14. Ringkasan Hasil
Bagian akhir notebook merangkum:
- jumlah data
- akurasi model
- model yang dipilih untuk deployment
- alasan pemilihan final approach

---

## 📂 Struktur Repository

```bash
.
├── notebooks/
│   └── asl_recognition_notebook.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── hand_roi.py
│   ├── landmark_features.py
│   ├── train_landmark_model.py
│   ├── inference.py
│   ├── landmark_inference.py
│   └── utils.py
│
├── app/
│   ├── main.py
│   ├── templates/
│   └── static/
│
├── models/
│   ├── best_model.keras
│   ├── label_map.json
│   ├── landmark_mlp.joblib
│   ├── landmark_label_encoder.joblib
│   └── landmark_label_map.json
│
├── reports/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── confusion_matrix.png
│
├── dataset/ or data/
│   └── (dataset / extracted data / processed data)
│
└── README.md
```

---

## 🧪 Notebook Flow Singkat

Secara ringkas, alur notebook adalah:

```text
Config
→ Utilities
→ Data Loading + EDA
→ Image Preprocessing
→ CNN Training
→ CNN Evaluation
→ Hand ROI
→ Landmark Feature Extraction
→ Landmark MLP Training
→ Landmark Evaluation
→ Inference Demo
→ Summary
```

Jadi README ini memang mengikuti **flow notebook**, bukan hanya daftar file `.py`.

---

## 🌐 Implementasi Web App

Selain notebook eksperimen, project ini juga memiliki implementasi web interaktif yang mendukung:

### 1. Single Capture Prediction
- ambil satu frame dari webcam
- crop area gesture
- lakukan prediksi satu label ASL
- tampilkan hasil, confidence, dan top-k prediction

### 2. Spell Builder Mode
- melakukan prediksi berulang dari webcam
- menggunakan stabilisasi prediksi
- menyusun huruf menjadi string
- mendukung:
  - `del` untuk backspace
  - `space` untuk spasi
  - `nothing` untuk no-hand / release state

Model yang dipakai untuk deployment adalah **landmark-based model** karena lebih stabil untuk input webcam.

---

## ✅ Kenapa Model Landmark Dipilih untuk Deployment

Walaupun baseline CNN memberikan hasil yang baik pada data citra, model final yang dipakai di web app adalah **MediaPipe + Landmark MLP** karena:

- lebih robust terhadap background
- lebih konsisten di kondisi webcam
- lebih ringan untuk inference
- lebih tahan terhadap variasi antar pengguna

Namun demikian, beberapa gesture tetap menantang, terutama:

- gesture dinamis seperti **J** dan **Z**
- gesture dengan occlusion tinggi seperti **O**, **C**, atau pose jari tertutup

---

## ⚠️ Keterbatasan Project

Beberapa keterbatasan sistem saat ini:

- gesture dinamis belum ideal jika hanya diprediksi dari single frame
- beberapa huruf dengan bentuk mirip masih berpotensi tertukar
- performa sangat dipengaruhi kualitas framing tangan dan deteksi landmark
- deployment realtime tetap membutuhkan gesture yang cukup jelas dan stabil

---

## 🚀 Cara Menjalankan Project

### Opsi 1 — Jalankan Notebook
Buka notebook utama:

```bash
jupyter notebook notebooks/asl_recognition_notebook.ipynb
```

atau gunakan Google Colab / JupyterLab sesuai kebutuhan.

### Opsi 2 — Jalankan Web App
Jika web app disertakan dalam repo:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Lalu buka browser ke:

```text
http://localhost:7860
```

---

## 📌 Kebutuhan Environment

Library utama yang digunakan:

- Python 3.x
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- Pillow
- joblib
- FastAPI
- Uvicorn
- MediaPipe

Jika notebook memiliki cell instalasi dependency, ikuti sesuai urutan cell awal notebook.

---

## 📝 Catatan Pengumpulan

Untuk kebutuhan pengumpulan, komponen utama project ini adalah:

1. **Notebook analysis / preprocessing / modeling**
2. **Source code modular (`src/`)**
3. **Implementasi web app**
4. **README**
5. **Model artifacts**
6. **Artikel / penjelasan singkat berdasarkan notebook**

Dengan demikian, notebook berfungsi sebagai dokumentasi eksperimen utama, sedangkan folder `src/` dan `app/` menjadi codebase pendukung implementasi.

---

## 📚 Kesimpulan

Project ini menunjukkan proses pengembangan sistem ASL recognition dari dua sudut:

- **baseline image classification**
- **final landmark-based deployment**

Secara eksperimen, CNN memberikan baseline yang baik.  
Secara implementasi nyata, pipeline **MediaPipe Hands + Landmark MLP** lebih cocok untuk aplikasi webcam interaktif.

Pendekatan akhir ini dipilih karena memberikan trade-off terbaik antara:
- akurasi praktis,
- kestabilan real-time,
- dan kemudahan deployment.

---
