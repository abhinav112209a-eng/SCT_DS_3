import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

dataset_path = "PetImages"

for category in ["Cat", "Dog"]:
    path = os.path.join(dataset_path, category)
    label = 0 if category == "Cat" else 1

    for img in os.listdir(path)[:500]:
        try:
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (64, 64))

            data.append(image.flatten())
            labels.append(label)

        except:
            pass

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))