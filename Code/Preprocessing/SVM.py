import os
from glob import glob

from sklearn import svm
from sklearn.model_selection import train_test_split

from Code.Preprocessing.SignalQualityPlatform import WINDOWS_DIR, load_windows

path_dir = '/host/media/tuvalgelvan@staff.technion.ac.il/HD34/Estimated-Blood-Pressure-Project/test_data/windows'
windows = load_windows(path_dir)
print(windows)


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

