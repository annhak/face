from PIL import Image, ImageDraw
import face_recognition
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("ae.jpg")
# Create an empty (white) array of the same size
(h, w, d) = image.shape
empty = np.ones((h, w, d)).astype(np.uint8)*255

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

# Prepare the drawing board
pil_image = Image.fromarray(empty)
d = ImageDraw.Draw(pil_image, 'RGBA')

for face_landmarks in face_landmarks_list:
    # Draw all landmarks as lines. Notice that eyes will have a gap as the last and first points are not connected
    for i in face_landmarks.keys():
        d.line(face_landmarks[i], fill=(0, 0, 0, 110), width=6)

    # Gloss the lips
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))

plt.figure()
plt.imshow(pil_image)
