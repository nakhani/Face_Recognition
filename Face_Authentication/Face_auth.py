import cv2
import numpy as np
from face_identification import FaceIdentification
import subprocess  

class FaceAuth:
    def __init__(self, app_path):
        self.face_id = FaceIdentification()
        self.app_path = app_path
        self.cap = cv2.VideoCapture(0)  

    def authenticate(self):
        print("🔒 Face Authentication Started...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

          
            self.face_id.input_image = frame
            self.face_id.results = self.face_id.app.get(frame)
            
            if self.face_id.results:
                self.face_id.load_face_bank()
                for result in self.face_id.results:
                    for person in self.face_id.face_bank:
                        distance = np.sqrt(np.sum((person["embedding"] - result["embedding"])**2))
                        if distance <= self.face_id.threshold:
                            print(f"✅ Access Granted: Welcome {person['name']}!")
                            self.cap.release()
                            cv2.destroyAllWindows()
                            self.launch_application()
                            return
                print("❌ Access Denied: Face not recognized.")

           
            cv2.imshow("Face Authentication", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def launch_application(self):
        """ Launch an application after successful face verification """
        print(f"🚀 Launching application: {self.app_path}")
        subprocess.run(self.app_path, shell=True)

if __name__ == "__main__":
    
    app_path = "calc.exe"  # Windows Calculator (change for other OS/apps)
    auth_system = FaceAuth(app_path)
    auth_system.authenticate()
