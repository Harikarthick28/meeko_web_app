import cv2
import numpy as np
import os
import tensorflow as tf

class AutoDatasetEmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Exact labels from the dataset
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        # Extended emotions mapped from base emotions
        self.extended_emotions = ['love', 'excited', 'confused', 'bored', 'stressed', 'confident']
        
        self.model = self.load_model()
        
    def load_model(self):
        """Load the MobileNetV2 trained emotion model."""
        model_path = 'mobile_emotion_model.h5'
        
        if os.path.exists(model_path):
            print(f"📦 Loading mobile-optimized emotion model '{model_path}'...")
            try:
                from tensorflow.keras.models import load_model
                return load_model(model_path)
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return None
        else:
            print("⚠️ Trained model not found. Please wait for 'train_mobile_emotion.py' to finish.")
            print("Returning neutral face detection fallback for now.")
            return None

    def detect_emotion_advanced(self, frame):
        """Detect emotion using the Keras MobileNetV2 model trained on FER-2013."""
        if frame is None or frame.size == 0:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        results = []
        for (x, y, w, h) in faces:
            # We predict exactly one face at a time, using the Region of Interest
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                primary_emotion = 'neutral'
                extended_emotion = 'neutral'
                confidence = 0.5
                all_scores = {'neutral': 0.5}
                
                # Use the real model if it exists
                if self.model:
                    # Preprocess for MobileNetV2 (48x48, RGB, normalized)
                    face_resized = cv2.resize(face_roi, (48, 48))
                    face_norm = face_resized.astype('float32') / 255.0
                    face_tensor = np.expand_dims(face_norm, axis=0)  # Shape (1, 48, 48, 3)
                    
                    preds = self.model.predict(face_tensor, verbose=0)[0]
                    
                    # Read classes dynamically if the trainer saved them
                    if os.path.exists('emotion_classes.npy'):
                        class_indices = np.load('emotion_classes.npy', allow_dict=True).item()
                        idx_to_class = {v: k for k, v in class_indices.items()}
                        primary_emotion = idx_to_class[np.argmax(preds)]
                        confidence = float(np.max(preds))
                        all_scores = {idx_to_class[i]: float(preds[i]) for i in range(len(preds))}
                    else:
                        # Fallback mapping if classes file isn't found
                        primary_emotion = self.emotion_labels[np.argmax(preds)]
                        confidence = float(np.max(preds))
                        all_scores = {self.emotion_labels[i]: float(preds[i]) for i in range(len(preds))}
                        
                    extended_emotion = primary_emotion
                    if primary_emotion == "happy" and confidence > 0.8:
                        extended_emotion = "excited"
                    elif primary_emotion == "sad" and confidence > 0.8:
                        extended_emotion = "stressed"
                    
                results.append({
                    'emotion': primary_emotion,
                    'extended_emotion': extended_emotion,
                    'confidence': confidence,
                    'all_scores': all_scores,
                    'box': (x, y, w, h)
                })
                
            except Exception as e:
                print(f"Prediction error: {e}")
                continue
                
        # Sort faces by highest emotional confidence
        if results:
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results
            
        return None