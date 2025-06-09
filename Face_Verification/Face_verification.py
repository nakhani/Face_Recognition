import argparse
import numpy as np
import cv2
from insightface.app import FaceAnalysis


parser = argparse.ArgumentParser(description="Face comparison using InsightFace")


parser.add_argument('--image1', type=str, default="/content/input/image1.jpg", help="Path to first image")
parser.add_argument('--image2', type=str, default="/content/input/image2.jpg", help="Path to second image")


opt, unknown = parser.parse_known_args()


app = FaceAnalysis(name="buffalo_s", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


image_1 = cv2.imread(opt.image1)
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
image_2 = cv2.imread(opt.image2)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)


result_1 = app.get(image_1)
result_2 = app.get(image_2)

if result_1 and result_2:
    embedding_1 = result_1[0]["embedding"]
    embedding_2 = result_2[0]["embedding"]


    diff = np.sqrt(np.sum((embedding_1 - embedding_2) ** 2))


    print(f"Difference Score: {diff:.2f}")
    if diff <= 25:
        print("\n*Same Person*")
    else:
        print("\n*Different Persons*")
else:
    print("No faces detected in one or both images.")
