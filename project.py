import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

dataset_dir = r"C:\Users\lenovo\Downloads\dataset (1)\dataset\faces"

X = []  
y = [] 

mp_face_mesh = mp.solutions.face_mesh
print(" Loading images and extracting face landmarks...")

for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue  # skip non-folder files

    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_path, img_name)
        try:
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:
                results = face_mesh.process(image_rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                    X.append(features)
                    y.append(person_name)
        except Exception as e:
            print(f" Skipped {img_name}: {e}")

X = np.array(X)
y = np.array(y)

print(f" Extracted features from {len(X)} faces of {len(set(y))} people")

if len(X) < 2:
    raise ValueError(" Not enough images found. Please add more data per person.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print("\n Training SVM classifier...")
clf = SVC(C=1, kernel='linear', probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

print(f"\n Model training complete! Accuracy: {acc:.2f}%")
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

model_path = os.path.join(dataset_dir, "face_recognizer_model_mediapipe.pkl")
joblib.dump(clf, model_path)
print(f"\n Model saved at: {model_path}")

print("\n Face Recognition Project Completed Successfully!")
import cv2
import numpy as np
import mediapipe as mp
import joblib
import os
model_path = r"C:\Users\lenovo\Downloads\dataset (1)\dataset\faces\face_recognizer_model_mediapipe.pkl"
clf = joblib.load(model_path)

mp_face_mesh = mp.solutions.face_mesh

test_image_path = r"C:\Users\lenovo\Downloads\dataset (1)\dataset\faces\Ileana\face_16.jpeg" 
image = cv2.imread(test_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

  
        prediction = clf.predict([features])[0]
        probabilities = clf.predict_proba([features])[0]

        print(f" Predicted Person: {prediction}")
        print(" Probabilities for all classes:")
        for person, prob in zip(clf.classes_, probabilities):
            print(f"   {person}: {prob:.2f}")

    else:
        print(" No face detected in the image.")
