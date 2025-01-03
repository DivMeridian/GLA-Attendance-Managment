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
        self.known_embeddings = []
        self.known_labels = []
        self.load_embeddings_from_db()

        # intializing Twilio credentials and client
        account_sid = 'AC6f69c2558fd39d71e1c080175b36f209'  # Replace with your Account SID
        auth_token = 'd87a49650a62b9408d5580060dd12f2c'    # Replace with your Auth Token
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
        """
        detections, embeddings = face_detector.detect_faces(frame)
        if not embeddings:
            print("No face detected. Please try again.")
            return

        # Assuming a single face for registration
        embedding = embeddings[0]
        self.known_embeddings.append(embedding)
        self.known_labels.append(label)

        # Save embedding to MongoDB
        self.collection.insert_one({"label": label, "embedding": embedding.tolist(),"Contact":contact,"section":section})
        print(f"Registered '{label}' successfully and saved to MongoDB.")

    def recognize_faces(self, detections, embeddings):
        """
        Recognize faces by comparing embeddings to the database.
        :param detections: List of detected face bounding boxes.
        :param embeddings: List of detected face embeddings.
        :return: List of results with labels and confidence.
        """
        results = []
        for bbox, embedding in zip(detections, embeddings):
            if not self.known_embeddings:
                results.append((bbox, "Unknown", 0))
                continue

            # Calculate similarity scores
            similarities = [1 - cosine(embedding, known_emb) for known_emb in self.known_embeddings]
            best_match_idx = np.argmax(similarities)
            best_match_score = similarities[best_match_idx]

            if best_match_score > 0.57:  # Threshold
                label = self.known_labels[best_match_idx]

                # Fetch contact number from MongoDB
                person = self.collection.find_one({"label": label})

                if person and "Contact" in person:
                    contact_number = person["Contact"]
                    # Format the number with country code
                    to_number = f"+91{contact_number}"
                    # Send SMS
                    self.twilio_client.send_sms(to_number, "Attendance Marked")
                    
                results.append((bbox, label, best_match_score))
            else:
                results.append((bbox, "Unknown", best_match_score))
        return results
