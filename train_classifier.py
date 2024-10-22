import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
with open('dataset.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Extract data and labels from the loaded dataset
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
# `stratify=labels` ensures each label is evenly represented in train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the RandomForest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model's performance
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
