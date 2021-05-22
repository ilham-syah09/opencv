#  Created by od3ng on 03/04/2019 01:37:26 PM.
#  Project: face-detection-pi
#  File: face-detection.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0

import cv2

# Untuk mengambil gambar menggunakan streaming video
cap = cv2.VideoCapture(0)
# Load lokasi hasil training dari haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cv2.namedWindow("face detection", cv2.WINDOW_GUI_EXPANDED)

# Looping terus untuk membaca gambar frame demi frame
while True:
    ret, frame = cap.read()
    if frame is None:
        continue
    # Ubah gambar dari BRG ke grayscale
    gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

    # Hasil dari deteksi wajah adalah list rectangle atau kotak
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    print("found {0} faces!".format(len(faces)))

    # Membuat kotak dengan opencv berdasarkan wajah-wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("face detection", frame)
    # Frame akan keluar ketika tombol q pada keyboard ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Jika selesai release semua resource
cap.release()
cv2.destroyAllWindows()