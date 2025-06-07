# 🔍 Face Verification  

## 📌 Overview  
This project implements **Face Verification** using `insightface`. It extracts **512D embedding vectors** from two input images and compares them to determine identity similarity. If the embeddings are close, it prints **"Same Person"**; otherwise, it prints **"Different Persons"**.  

---
## 📖 How It Works

- Loads two images.
- Extracts **512D face embedding vectors** using i`nsightface`.
- Computes similarity between embeddings.
- Prints **"Same Person"** or **"Different Persons"** based on similarity score.

---

## How to Run the Code
1. Clone the repository:

   ```
   https://github.com/nakhani/Face_Recognition/tree/ee468150293002d89e719d2cd9c9f03671e1ca7c/Face_Verification
   ```

2. Navigate to the directory:

   ```
   Face_Verification
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the project:
   ```bash
    jupyter notebook Face_Verification.ipynb #For verify 2 Faces
   ```
---
## Dependencies
- argparse
- numpy
- opencv-python
- insightface
- onnxruntime

