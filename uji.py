import cv2
import numpy as np
from keras.models import load_model

# Menggunakan model yang telah dilatih
model = load_model('emotion_recognition_model.h5')

# Ekspresi yang digunakan
expressions = {0: "senyum", 1: "marah", 2: "sedih"}

# Menyiapkan Haarcascades untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Menyiapkan kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    ret, img = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Membalikkan gambar agar lebih alami
    img = cv2.flip(img, 1)

    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Gambar persegi panjang di sekitar wajah
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Ambil gambar wajah dan resize menjadi ukuran yang sesuai dengan model
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (64, 64))

        # Normalisasi gambar
        face_resized = face_resized.astype("float32") / 255.0

        # Membuat array dengan bentuk yang sesuai untuk input model
        face_resized = np.expand_dims(face_resized, axis=-1)  # Menambahkan dimensi untuk channel (grayscale)
        face_resized = np.expand_dims(face_resized, axis=0)  # Menambahkan dimensi batch

        # Prediksi ekspresi wajah
        prediction = model.predict(face_resized)
        predicted_label = np.argmax(prediction)

        # Menampilkan label pada wajah yang terdeteksi
        cv2.putText(img, f"Ekspresi: {expressions[predicted_label]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Menampilkan hasil di jendela
    cv2.imshow("Prediksi Ekspresi Wajah", img)

    # Menunggu input ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Menutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
