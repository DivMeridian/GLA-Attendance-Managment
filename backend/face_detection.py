from insightface.app import FaceAnalysis

''' RetinaFace uses feature maps with strides of 8, 16, and 32, 
 the input resolution should ideally be divisible by 32. 
 so thats why it will not work for 1920x1080 and 1080x1080 but it will work for 1920x1088
 that is why 2048x2048 and 1024x1024 works and 1280x1280 works also 
'''

class FaceDetector:
    def __init__(self):
        """
        Initialize the FaceDetector using InsightFace.
        """
        self.face_app = FaceAnalysis()
        self.face_app.prepare(ctx_id=-1, det_size=(1280, 1280))  # Adjust det_size as needed

    def detect_faces(self, frame):
        """
        Detect faces in a given frame.
        :param frame: Input image/frame (numpy array).
        :return: List of detected faces with bounding boxes and embeddings.
        """
        faces = self.face_app.get(frame)  # Detect faces
        detections = []
        embeddings = []

        for face in faces:
            bbox = list(map(int, face.bbox))  # Bounding box (x1, y1, x2, y2)
            confidence = float(face.det_score)  # Detection confidence

            if len(bbox) == 4:  # Valid bbox
                detections.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence])
                embeddings.append(face.normed_embedding)

        return detections, embeddings
