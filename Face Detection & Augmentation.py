def detect_face(img):
    result = detector.detect_faces(img)
    if not result:
        return None, None
    x, y, w, h = result[0]['box']
    x = abs(x)
    y = abs(y)
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))  # FaceNet requires 160x160
    return face, (x, y, w, h)

# Data Augmentation Generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)