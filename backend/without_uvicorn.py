import cv2
from face_detection import FaceDetector
from face_recognition import FaceRecognition


def draw_bounding_boxes(frame, results):
    """
    Draw bounding boxes with labels on the frame.
    :param frame: Input frame.
    :param results: List of results with bounding boxes, labels, and confidence.
    """
    for result in results:
        bbox = result[0][:4]  # Extract the first four values (x1, y1, x2, y2)
        label = result[1]
        confidence = result[2]

        x1, y1, x2, y2 = map(int, bbox)  # Ensure coordinates are integers
        color = (0, 255, 0) if label != "Unknown" else (0, 255, 255)  # Green for known, Yellow for unknown
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def register_person(face_recognition, face_detector):
    """
    Register a new person using an image provided by the user.
    :param face_recognition: Instance of FaceRecognition.
    :param face_detector: Instance of FaceDetector.
    """
    image_path = input("Enter the path to the registration image: ").strip()
    label = input("Enter the name of the person to register: ").strip()

    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image. Please check the path.")
        return

    # Register the person
    face_recognition.register_person(frame, face_detector, label)
    print(f"'{label}' has been successfully registered.")


def main():
    # Initialize components
    face_detector = FaceDetector()
    face_recognition = FaceRecognition()

    print("Classroom Face Recognition System")
    print("1. Detect and recognize faces in a classroom image.")
    print("2. Register a new person using an image.")
    print("3. Exit.")

    while True:
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            # Input classroom image
            image_path = input("Enter the path to the classroom image: ").strip()
            frame = cv2.imread(image_path)

            if frame is None:
                print("Failed to load image. Please check the path.")
                continue

            # Detect and recognize faces
            detections, embeddings = face_detector.detect_faces(frame)
            results = face_recognition.recognize_faces(detections, embeddings)

            # Draw bounding boxes
            draw_bounding_boxes(frame, results)

            # Display the image
            display_frame = cv2.resize(frame, (1024, 768))
            cv2.imshow("Face Detection", display_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif choice == "2":
            # Register a new person
            register_person(face_recognition, face_detector)

        elif choice == "3":
            print("Exiting the system. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
