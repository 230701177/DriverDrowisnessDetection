import cv2
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")
hog = cv2.HOGDescriptor()

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 3D model points (important for SolvePnP)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0),# Left eye
    (225.0, 170.0, -135.0), # Right eye
    (-150.0, -150.0, -125.0),# Left mouth
    (150.0, -150.0, -125.0) # Right mouth
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = frame.shape

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        # HOG Classification (old part)
        face = gray[y:y+h, x:x+w]
        try:
            face_resized = cv2.resize(face, (224,224))
        except:
            continue

        features = hog.compute(face_resized).flatten().reshape(1, -1)
        pred = model.predict(features)

        # Approximate 2D image points
        image_points = np.array([
            (x+w/2, y+h/2),           # Nose
            (x+w/2, y+h),             # Chin
            (x, y+h/3),               # Left eye
            (x+w, y+h/3),             # Right eye
            (x+w/3, y+2*h/3),         # Left mouth
            (x+2*w/3, y+2*h/3)        # Right mouth
        ], dtype="double")

        # Camera matrix
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)

        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4,1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        # Convert rotation vector to angles
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]
        yaw = angles[1]

        # Decision logic
        if pitch > 20:
            status = "DROWSY (HEAD DOWN)"
            color = (0,0,255)
        elif abs(yaw) > 20:
            status = "DISTRACTED"
            color = (0,255,255)
        else:
            status = "ALERT"
            color = (0,255,0)

        # Draw
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame, status, (x,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        cv2.putText(frame, f"Pitch: {round(pitch,2)}",
                    (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.putText(frame, f"Yaw: {round(yaw,2)}",
                    (x,y+h+40), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()