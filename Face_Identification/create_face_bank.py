import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class CreateFaceBank:
    def __init__(self, face_bank_path="face_bank"):
        #self.app = FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
        #self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.face_bank_path = face_bank_path

    def update(self, app):
        face_bank = []
        
        for person_name in os.listdir(self.face_bank_path):
            folder_path = os.path.join(self.face_bank_path, person_name)
            
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    
                    image = cv2.imread(file_path)
                    if image is None:
                        print(f"❌ Error: Could not read image {file_path}. Skipping...")
                        continue

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = app.get(image)

                    if not result:
                        print(f"⚠️ Warning: No face detected in {file_path}. Skipping...")
                        continue

                    if len(result) > 1:
                        print(f"⚠️ Warning: More than one face detected in {file_path}. Skipping...")
                        continue

                    embedding = result[0]["embedding"]
                    face_data = {"name": person_name, "embedding": embedding}
                    face_bank.append(face_data)

        if face_bank:
            np.save("face_bank.npy", face_bank)
            print(f"✅ Face bank saved successfully with {len(face_bank)} persons!")
        else:
            print("❌ No valid faces detected. Face bank not created.")


if __name__ == "__main__":
    face_bank_creator = CreateFaceBank()
    face_bank_creator.update()
