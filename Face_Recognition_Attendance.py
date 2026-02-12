import face_recognition
import cv2
import os
import threading
import time

# Define the base directory for known faces and imagesqq
image_path_base = os.path.dirname(os.path.abspath(__file__))
known_faces_dir = os.path.join(image_path_base, 'known_faces')

# Create a directory to store the known faces if it doesn't exist
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

# Global variable to store the current frame for recognition
current_frame = None
frame_lock = threading.Lock()

def add_face(name):
    """Capture an image and save the face encoding."""
    image_path = os.path.join(known_faces_dir, f"{name}.jpg")

    # Open the camera and capture an image
    cap = cv2.VideoCapture(0)
    print("Press 's' to take a picture...")
    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            break
    cap.release()
    cv2.destroyAllWindows()

    # Load the image and encode it
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]

    # Save the encoding and name in a file
    with open(f"{known_faces_dir}/{name}.txt", "w") as f:
        f.write(",".join(map(str, encoding)))
    print(f"Face for {name} added successfully.")

def capture_frame():
    """Capture frames from the camera."""
    global current_frame
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        with frame_lock:
            current_frame = frame  # Store the current frame
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_faces():
    """Recognize faces in the captured frames."""
    global current_frame
    print("Starting face recognition. Press 'q' to quit.")
    
    known_encodings = {}
    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(known_faces_dir, filename), "r") as f:
                encoding = list(map(float, f.read().strip().split(',')))
                known_encodings[filename[:-4]] = encoding  # Store name without .txt extension

    frame_skip = 5  # Process every 5th frame (increase if needed)
    frame_count = 0

    last_recognized_name = None  # Variable to track the last recognized name

    while True:
        with frame_lock:
            if current_frame is None:
                continue  # Skip if no frame is available

            # Resize the frame for faster processing (adjust this as needed)
            small_frame = cv2.resize(current_frame, (0, 0), fx=0.5, fy=0.5)  # Reduce to 50% of original size
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Process every nth frame
            frame_count += 1
            if frame_count % frame_skip == 0:
                # Find all face locations and encodings in the current frame
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Compare with known faces
                    matches = face_recognition.compare_faces(list(known_encodings.values()), face_encoding)
                    name = "Not Recognized"

                    # If a match was found, use the first one
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = list(known_encodings.keys())[first_match_index]

                    # Draw a box around the face and label it
                    # Scale back up the face locations since the frame we detected in was scaled down
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    cv2.rectangle(current_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(current_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Check if the recognized name has changed
                    if name != last_recognized_name:
                        print(f"Face recognized: {name}")  # Notify in console
                        last_recognized_name = name  # Update the last recognized name

        time.sleep(0.05)  # Slight delay to control processing rate

def main():
    while True:
        action = input("Would you like to add a new face? (yes/no): ").strip().lower()
        if action == 'yes':
            name = input("Enter the name of the person: ")
            add_face(name)
        elif action == 'no':
            print("Starting camera capture and recognition...")
            # Start threads for capturing frames and recognizing faces
            threading.Thread(target=capture_frame, daemon=True).start()
            threading.Thread(target=recognize_faces, daemon=True).start()
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Exit the recognition loop
            break
        else:
            print("Invalid input. Please type 'yes' or 'no'.")

if __name__ == "__main__":

    main()
