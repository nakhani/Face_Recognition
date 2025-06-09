import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class CreateFaceBank:
    def __init__(self, face_bank_path="./face_bank/"):
        
        self.face_bank_path = face_bank_path

    def update(self, face_bank_path, app):

        face_bank = []
        for person_name in os.listdir(face_bank_path):
            
            folder_path = os.path.join(face_bank_path, person_name)
            
            if os.path.isdir(folder_path):
                
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    result = app.get(image)

                    if len(result) > 1:
                        print(f"Warning: more than one face detected in image: {file_path}")
                        continue

                    embedding = result[0]["embedding"]
                    dict = {"name": person_name, "embedding": embedding}
                    face_bank.append(dict)

        np.save("face_bank.npy", face_bank)