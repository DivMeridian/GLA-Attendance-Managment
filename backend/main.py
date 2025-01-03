from fastapi import FastAPI, UploadFile, Form
import uvicorn
import cv2
import os
from face_detection import FaceDetector
from face_recognition import FaceRecognition
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()

# Allow CORS for React to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your React app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face detector and recognition
face_detector = FaceDetector()
face_recognition = FaceRecognition()

# Create Results directory if it doesn't exist
os.makedirs("Results", exist_ok=True)

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

@app.post("/detect_and_recognize/")
async def detect_and_recognize(file: UploadFile,section:str=Form()):
    """
    Endpoint to detect and recognize faces in a classroom image.
    :param file: Uploaded classroom image file.
    :param section: Section of the classroom.
    :return: Path to the processed image with bounding boxes.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Failed to process the image. Ensure the file is a valid image."}

    # Detect and recognize faces
    detections, embeddings = face_detector.detect_faces(frame)
    results = face_recognition.recognize_faces(detections, embeddings)

    # Draw bounding boxes
    draw_bounding_boxes(frame, results)

    # Save the processed image
    result_path = os.path.join("Results", f"processed_{file.filename}")
    cv2.imwrite(result_path, frame)
    
    # Encode the image as a base64 string
    _, buffer = cv2.imencode(".jpg", frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    # Extract identified names from results
    identified_names = [result[1] for result in results]

    return {
        "message": "Detection and recognition complete.",
        "result_path": result_path,
        "image_base64": base64_image,
        "identified_names": identified_names,
    }

@app.post("/register_person/")
async def register_person(file: UploadFile, label: str = Form(...),Contact: int = Form(...),section:str=Form()):
    """
    Endpoint to register a new person using an image.
    :param file: Uploaded registration image file.
    :param label: Name of the person to register.
    :param Contact: Contact of the person to register.
    :param section: Section of the person to register.
    :return: Success or error message.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Failed to process the image. Ensure the file is a valid image."}

    # Register the person
    face_recognition.register_person(frame, face_detector, label,Contact,section)
    return {"message": f"'{label}' has been successfully registered."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
