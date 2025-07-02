import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Mengatur seed untuk reproduktifitas
np.random.seed(42)
# Membuat fitur
X = np.random.rand(1000, 3)  # 1000 sampel, 3 fitur
# Membuat label (0 atau 1)
y = np.random.randint(0, 2, 1000)  # 1000 label 

# Memisahkan data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Membuat model
model = LogisticRegression()
# Melatih model
model.fit(X_train, y_train) 

# Memprediksi
y_pred = model.predict(X_test)
# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy:.2f}')
# Menampilkan confusion matrix dan classification report
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred)) 