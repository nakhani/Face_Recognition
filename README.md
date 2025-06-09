# **Face Recognition System**
A comprehensive **Face Recognition System** using **InsightFace** for:
- **Face Verification** (Confirm if two images belong to the same person)
- **Face Identification** (Recognize individuals from a stored database)
- **Face ID** (Use facial recognition as a password for applications)
---
## 🚀 Features

### ✅ **Face Verification**
- Install **InsightFace** package.
- Create `face_verification.py` file.
- Get two image paths using command-line arguments:
  ```sh
  python face_verification.py --image1 ./img1.jpg --image2 ./img2.jpg
  ```
- Extract **512D embedding vectors** for images using InsightFace.
- Compare **embedding vectors**.
- Print:
   - `Same Person` if embeddings are similar.
   - `Different Persons` otherwise.

---

### ✅ **Face Identification**
- Create `create_face_bank.py` file.
- Extract **512D embedding vectors** for known persons using InsightFace.
- Save **face embeddings** to `face_bank.npy`.
- Create `face_identification.py` file.
- Get an image path using command-line arguments:

  ```bash
  python face_identification.py --image ./img.jpg
  ```
- Recognize and print the **name** of the person in the image.
- Draw a **bounding box** and **name** on the recognized face.
- Add `--update` argument to `face_identification.py`:

  ```bash
  python face_identification.py --image ./img.jpg --update
  ```
  - If `--update` is used, **the face bank** updates automatically before identification.

- Code is **Object-Oriented** using the `FaceIdentification` class.
---

### ✅ **Face ID (Smart Webcam-Based Password)**
- Use **FaceIdentification module** to create a **smart webcam-based password**.
- Utilize face authentication for secure access to applications.
- Face recognition as a **keyless login mechanism**.
---
