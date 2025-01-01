import cv2
import numpy as np
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Menyiapkan data untuk pelatihan
face_images = []
labels = []
expressions = {"senyum": 0, "marah": 1, "sedih": 2}

# Menyiapkan Haarcascades untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Folder untuk menyimpan gambar wajah dan dataset
if not os.path.exists('wajah_dataset'):
    os.mkdir('wajah_dataset')

if not os.path.exists('images'):
    os.mkdir('images')

# Ekspresi yang digunakan
print("Tekan tombol berikut untuk menambahkan data ekspresi wajah:")
print("1: Senyum, 2: Marah, 3: Sedih, ESC: Keluar")

# Mengambil gambar dari kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

notification = ""
notification_frames = 0

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

        # Ambil gambar wajah
        face = gray[y:y + h, x:x + w]

        # Simpan gambar wajah untuk dataset
        face_filename = f"wajah_dataset/{x}_{y}_{w}_{h}.jpg"
        cv2.imwrite(face_filename, face)

        # Simpan gambar wajah ke folder 'images' dengan nama unik
        image_filename = f"images/{x}_{y}_{w}_{h}.jpg"
        cv2.imwrite(image_filename, face)

        # Tampilkan wajah yang terdeteksi
        cv2.putText(
            img,
            "Wajah Deteksi",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

    # Tampilkan notifikasi jika ada
    if notification_frames > 0:
        cv2.putText(
            img,
            notification,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        notification_frames -= 1

    cv2.imshow("Emotion Recognition Database", img)

    # Deteksi tombol untuk memilih ekspresi wajah
    key = cv2.waitKey(30) & 0xFF
    if key == ord('1'):  # Senyum
        label = "senyum"
        notification = "Ekspresi Senyum Tersimpan!"
        notification_frames = 50
    elif key == ord('2'):  # Marah
        label = "marah"
        notification = "Ekspresi Marah Tersimpan!"
        notification_frames = 50
    elif key == ord('3'):  # Sedih
        label = "sedih"
        notification = "Ekspresi Sedih Tersimpan!"
        notification_frames = 50
    elif key == 27:  # ESC untuk keluar
        break
    else:
        continue

    # Simpan data ke file CSV dengan menyimpan lokasi gambar wajah dan label ekspresi
    face_images.append(cv2.resize(face, (64, 64)))  # Resize gambar wajah
    labels.append(expressions[label])

# Menutup kamera
cap.release()
cv2.destroyAllWindows()

# Pastikan data ada sebelum melanjutkan ke pelatihan
if len(face_images) == 0 or len(labels) == 0:
    print("Tidak ada data wajah yang terkumpul!")
else:
    # Mengonversi label menjadi kategori (one-hot encoding)
    labels = to_categorical(labels, num_classes=len(expressions))

    # Mengonversi gambar menjadi array NumPy
    face_images = np.array(face_images)

    # Normalisasi gambar
    face_images = face_images.astype("float32") / 255.0

    # Membagi data menjadi data pelatihan dan data uji
    X_train, X_test, y_train, y_test = train_test_split(face_images, labels, test_size=0.2, random_state=42)

    # Pastikan data terbagi dengan benar
    print(f"Data pelatihan: {X_train.shape}, Data pengujian: {X_test.shape}")

    # Membangun model CNN
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(expressions), activation='softmax')
    ])

    # Kompilasi model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Latih model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Simpan model
    model.save('emotion_recognition_model.h5')
    print("Model telah disimpan sebagai 'emotion_recognition_model.h5'.")
