import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from twilio_utils import TwilioClient


class FaceRecognition:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="FaceRecognitionDB", collection_name="Embeddings"):
        """
        Initialize MongoDB connection and load embeddings.
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.known_labels = []
        self.known_embeddings = []
        self.load_embeddings_from_db()

        # intializing Twilio credentials and client
        account_sid = 'AC6f69c2558fd39d71e1c080175b36f209'  # Replace with your Account SID
        auth_token = '0687d6199b11170e5f47a6f4e7ba3b32'    # Replace with your Auth Token
        twilio_number = '+12317455948'  # Replace with your Twilio number
        self.twilio_client = TwilioClient(account_sid, auth_token, twilio_number)


    def load_embeddings_from_db(self):
        """
        Load embeddings and labels from MongoDB.
        """
        documents = self.collection.find()
        for doc in documents:
            self.known_embeddings.append(np.array(doc["embedding"]))
            self.known_labels.append(doc["label"])
        print(f"Loaded {len(self.known_embeddings)} embeddings from MongoDB.")

    def register_person(self, frame, face_detector, label,contact,section):
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
            print("No face detected. Please try again.")
            return

        # Assuming a single face for registration
        embedding = embeddings[0]

        # using a section based embedding collection
        collection_name = f"Embeddings_{section}"
        collection=self.db[collection_name]

        # Check if the person already exists in the same section 
        existing_person = collection.find_one({"label": label})
        if existing_person:
            return {f"'{label}' already exists in the '{section}' section."}
        
        collection.insert_one({"label": label, "embedding": embedding.tolist(),"Contact":contact,"section":section})
        self.known_embeddings.append(embedding)
        self.known_labels.append(label)
        
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
