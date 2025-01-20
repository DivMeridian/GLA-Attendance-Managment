from fastapi import FastAPI, UploadFile, Form, HTTPException
import uvicorn
import cv2
import os
from face_detection import FaceDetector
from face_recognition import FaceRecognition
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import base64
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv
from email_utils import send_attendance_email


app = FastAPI()
load_dotenv()

# Azure Blob Storage configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

# Initialize Azure Blob Storage client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

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
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

@app.post("/detect_and_recognize/")
async def detect_and_recognize(file: UploadFile,section:str=Form()):
    """
    Endpoint to detect and recognize faces in a classroom image.
    :param file: Uploaded classroom image file.
    :param section: Section of the classroom.
    :return: Path to the processed image with bounding boxes.
    """
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to process the image. Ensure the file is a valid image.")

        # Detect and recognize faces
        detections, embeddings = face_detector.detect_faces(frame)

        if not detections:
            raise HTTPException(status_code=400, detail="No faces detected in the image.")
        
        results = face_recognition.recognize_faces(detections, embeddings,section)

        # Draw bounding boxes
        draw_bounding_boxes(frame, results)

        # Save the processed image
        local_result_path = os.path.join("Results", f"processed_{file.filename}")
        cv2.imwrite(local_result_path, frame)

        blob_name = f"{section}/recognized_images/processed_{file.filename}"
        
        # Encode the image as a base64 string
        _, buffer = cv2.imencode(".jpg", frame)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        container_client.upload_blob(
            name=blob_name,
            data=buffer.tobytes(),
            content_settings=ContentSettings(content_type="image/jpeg"),
            overwrite=True
        )

        # Extract identified names from results
        identified_names = [result[1] for result in results]

        # Delete the local image file
        os.remove(local_result_path)

        return {
            "message": "Detection and recognition complete.",
            "result_path": local_result_path,
            "image_base64": base64_image,
            "identified_names": identified_names,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/register_person/")
async def register_person(file: UploadFile, label: str = Form(...),Contact: int = Form(...),section:str=Form(),email:str=Form(),rollNumber:str=Form()):
    """
    Endpoint to register a new person using an image.
    :param file: Uploaded registration image file.
    :param label: Name of the person to register.
    :param Contact: Contact of the person to register.
    :param section: Section of the person to register.
    :return: Success or error message.
    """
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to process the image. Ensure the file is a valid image.")

        # Register the person
        message = face_recognition.register_person(frame, face_detector, label,Contact,section,email,rollNumber)

        if message == "No face detected. Please try again.":  # No face detected
            raise HTTPException(status_code=400, detail="No face detected. Please try again.")
        else:
            return {"message": f"'{label}' has been successfully registered."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/get_registered_users/{section}")
async def get_registered_users(section: str):
    """
    Get all registered users for a given section.
    """
    try:
        collection_name = f"Embeddings_{section}"
        collection = face_recognition.db[collection_name]
        users = collection.find({}, {"_id": 0, "label": 1})  # Fetch only the labels
        registered_users = [user["label"] for user in users]
        return {"registered_users": registered_users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/submit_attendance/")
async def submit_attendance(data: dict):
    """
    Endpoint to receive and save final attendance data.
    :param data: JSON data containing attendance information.
    """
    try:
        section = data.get("section")
        attendance = data.get("attendance", [])
        if not section or not attendance:
            raise HTTPException(status_code=400, detail="Section or attendance data is missing.")
        
        # Save or log attendance data (this example prints it)
        print(f"Attendance for section {section}: {attendance}")

        # Retrieve the corresponding collection for the section
        collection_name = f"Embeddings_{section}"
        collection = face_recognition.db[collection_name]
        for entry in attendance:
            name = entry.get("name")
            present = entry.get("present")
            # Find the person's contact in the database
            person = collection.find_one({"label": name})
            if person and "Contact" in person:
                email_address = person["email"]

                # Create a message based on attendance status
                if present:
                    subject = "Attendance Notification: Present"
                    message = f"Dear {name}, your attendance for section {section} has been marked as Present."
                else:
                    subject = "Attendance Notification: Absent"
                    message = f"Dear {name}, your attendance for section {section} has been marked as Absent."

                # Send SMS
                send_attendance_email(to_email=email_address, subject=subject, plain_text_body=message)

        return {"message": "Attendance submitted successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/get_sections/")
async def get_sections():
    """
    Fetch all distinct sections from the database.
    """
    try:
        # Query all collections starting with 'Embeddings_'
        collection_names = face_recognition.db.list_collection_names()
        sections = [name.replace("Embeddings_", "") for name in collection_names if name.startswith("Embeddings_")]
        return {"sections": sections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
