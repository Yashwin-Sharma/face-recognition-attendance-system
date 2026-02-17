import os
import cv2
import numpy as np
import face_recognition
import datetime
import threading
import time


# PROJECT PATH SETUP (Portable)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)


# GLOBAL VARIABLES


current_frame = None
frame_lock = threading.Lock()
recognized_today = set()


# ADD NEW FACE


def add_face(name):
    cap = cv2.VideoCapture(0)
    print("Press 's' to capture image")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        cv2.imshow("Capture Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Encode face
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print("No face detected. Try again.")
        return

    encoding = encodings[0]

    encoding_file = os.path.join(KNOWN_FACES_DIR, f"{name}.npy")
    np.save(encoding_file, encoding)

    print(f"Face encoding saved for {name}")


# LOAD KNOWN FACES


def load_known_faces():
    encodings = []
    names = []

    for file in os.listdir(KNOWN_FACES_DIR):
        if file.endswith(".npy"):
            path = os.path.join(KNOWN_FACES_DIR, file)
            encoding = np.load(path)
            encodings.append(encoding)
            names.append(file.replace(".npy", ""))

    return encodings, names


# MARK ATTENDANCE


def mark_attendance(name):
    file_exists = os.path.isfile(ATTENDANCE_FILE)

    with open(ATTENDANCE_FILE, "a") as f:
        if not file_exists:
            f.write("Name,Date,Time\n")

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        f.write(f"{name},{date_str},{time_str}\n")


# CAPTURE CAMERA FRAMES


def capture_frames():
    global current_frame
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with frame_lock:
            current_frame = frame.copy()

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# FACE RECOGNITION LOGIC


def recognize_faces():
    global current_frame

    known_encodings, known_names = load_known_faces()

    if len(known_encodings) == 0:
        print("No known faces found.")
        return

    print("Starting recognition. Press 'q' to quit.")

    while True:
        with frame_lock:
            if current_frame is None:
                continue

            frame = current_frame.copy()

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # Strict threshold (more accurate)
            if face_distances[best_match_index] < 0.45:
                name = known_names[best_match_index]

            # Scale coordinates back
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if name != "Unknown" and name not in recognized_today:
                print(f"Attendance marked for {name}")
                mark_attendance(name)
                recognized_today.add(name)

        cv2.imshow("Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

    cv2.destroyAllWindows()


# MAIN FUNCTION


def main():
    while True:
        choice = input("Add new face? (yes/no): ").strip().lower()

        if choice == "yes":
            name = input("Enter name: ").strip()
            add_face(name)

        elif choice == "no":
            threading.Thread(target=capture_frames, daemon=True).start()
            threading.Thread(target=recognize_faces, daemon=True).start()

            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            break

        else:
            print("Invalid input. Type yes or no.")


if __name__ == "__main__":

    main()

