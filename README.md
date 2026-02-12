# Face Recognition Attendance System

This project is a Python-based face recognition system that automatically captures and recognizes faces using a webcam.

It allows adding new faces dynamically and stores their encodings for future recognition.

---

## Project Features

- Real-time face detection and recognition
- Add new faces using webcam capture
- Stores face encodings in text files
- Multithreading for parallel frame capture and recognition
- Frame skipping optimization for better performance
- Automatic creation of required folders

---

## Technologies Used

- Python
- OpenCV (cv2)
- face_recognition
- threading module
- os module

---

## How It Works

1. The program captures frames from the webcam.
2. Face encodings are generated using the face_recognition library.
3. Encodings are saved in `.txt` files inside the `known_faces` folder.
4. During recognition, stored encodings are compared with live camera frames.
5. Recognized faces are labeled in real time.

---

## Installation

1. Clone the repository:

git clone https://github.com/Yashwin-Sharma/face-recognition-attendance-system.git

2. Navigate into the project folder:

cd face-recognition-attendance-system

3. Install required libraries:

pip install -r requirements.txt

---

##Run the program

python face_recognition_attendance.py

## Projektbeschreibung (Deutsch)

Dieses Projekt wurde eigenständig entwickelt, um ein Gesichtserkennungssystem zur automatischen Anwesenheitserfassung zu realisieren. Dabei wurden Multithreading, Optimierung der Bildverarbeitung und strukturierte Dateiverwaltung umgesetzt.
