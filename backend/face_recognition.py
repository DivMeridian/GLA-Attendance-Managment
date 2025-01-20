import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from azure.storage.blob import BlobServiceClient
import cv2
from dotenv import load_dotenv
import os

load_dotenv()

class FaceRecognition:
    def __init__(self, mongo_uri=os.getenv('MONGO_URI'), db_name="AttendanceSystem", collection_name="Embeddings"):
        """
        Initialize MongoDB connection and load embeddings.
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.known_labels = []
        self.known_embeddings = []
        self.load_embeddings_from_db()

        # Initialize Azure Blob Storage client
        # Azure Storage Configuration
        AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        self.container_name = os.getenv("CONTAINER_NAME")


    def load_embeddings_from_db(self):
        """
        Load embeddings and labels from MongoDB.
        """
        documents = self.collection.find()
        for doc in documents:
            self.known_embeddings.append(np.array(doc["embedding"]))
            self.known_labels.append(doc["label"])
        print(f"Loaded {len(self.known_embeddings)} embeddings from MongoDB.")
    
    def upload_image_to_azure(self, section, label, image_path):
        """
        Upload the image to Azure Blob Storage in the path gla/Section/Registered/label.jpg.
        """
        try:
            blob_path = f"{section}/Registered/{label}.jpg"
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_path)

            with open(image_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            # Generate the URL
            blob_url = f"https://visiondetect.blob.core.windows.net/{self.container_name}/{blob_path}"
            print(f"Image uploaded successfully. URL: {blob_url}")

            return blob_url
        except Exception as e:
            print(f"Error uploading image to Azure: {e}")
            return False

    def register_person(self, frame, face_detector, label,contact,section,email,rollNumber):
        """
        Register a person by capturing their embedding from the frame.
        :param frame: Input frame containing the face.
        :param face_detector: Instance of FaceDetector.
        :param label: Label (e.g., name or ID).
        :param contact: Contact number of the person.
        :param section: Section to which the person belongs.
        """
        detections, embeddings = face_detector.detect_faces(frame)
        if not embeddings:
            return "No face detected. Please try again."

        # Assuming a single face for registration
        embedding = embeddings[0]

        # using a section based embedding collection
        collection_name = f"Embeddings_{section}"
        collection=self.db[collection_name]

        # Check if the person already exists in the same section 
        existing_person = collection.find_one({"label": label})
        if existing_person:
            return {f"'{label}' already exists in the '{section}' section."}
        
        # Save the face image locally
        x1, y1, x2, y2 = map(int, detections[0][:4])  # Assuming a single detection
        cropped_face = frame[y1:y2, x1:x2]
        local_image_path = f"{label}.jpg"
        cv2.imwrite(local_image_path, cropped_face)

        # Upload the image to Azure Blob Storage
        image_url = self.upload_image_to_azure(section, label, local_image_path)
        if image_url:
            # Insert person data into the database
            collection.insert_one({
                "label": label,
                "embedding": embedding.tolist(),
                "Contact": contact,
                "section": section,
                "email": email,
                "rollNumber": rollNumber,
                "image_url": image_url
            })
        
        self.known_embeddings.append(embedding)
        self.known_labels.append(label)

        # Delete the local image file
        os.remove(local_image_path)
        
        print(f"Registered '{label}' successfully and saved to MongoDB.")
        
        return {"message": f"'{label}' has been successfully registered in section '{section}'."}

    def recognize_faces(self, detections, embeddings,section):
        """
        Recognize faces by comparing embeddings to the database.
        :param detections: List of detected face bounding boxes.
        :param embeddings: List of detected face embeddings.
        :param section: Section to search for matching faces.
        :return: List of results with labels and confidence.
        """
        # Use the section-based collection
        collection_name = f"Embeddings_{section}"
        collection = self.db[collection_name]

        # Load embeddings and labels dynamically for the specified section
        section_embeddings = []
        section_labels = []
        documents = collection.find()

        for doc in documents:
            section_embeddings.append(np.array(doc["embedding"]))
            section_labels.append(doc["label"])
        
        if not section_embeddings:
            print(f"No embeddings found in section {section}.")
            return [(bbox, "Unknown", 0) for bbox in detections]

        # Performing recognition
        results = []
        for bbox, embedding in zip(detections, embeddings):

            # Calculate similarity scores
            similarities = [1 - cosine(embedding, known_emb) for known_emb in section_embeddings]
            best_match_idx = np.argmax(similarities)
            best_match_score = similarities[best_match_idx]

            if best_match_score > 0.57:  # Threshold
                label = section_labels[best_match_idx]
                results.append((bbox, label, best_match_score))
            else:
                results.append((bbox, "Unknown", best_match_score))

        return results
