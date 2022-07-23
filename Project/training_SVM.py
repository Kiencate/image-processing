from skimage.feature import hog
import joblib,glob,os,cv2

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, metrics
from sklearn.metrics import classification_report
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from hog1 import hog_feature
train_data = []
train_labels = []
pos_im_path = 'DATAIMAGE/positive/'
neg_im_path = 'DATAIMAGE/negative/'
model_path = 'models/my_model.dat'
# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path,"*.png")):
    fd = cv2.imread(filename,0)
    fd = cv2.resize(fd,(64,128))
    # fd = hog(fd,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(2,2))
    fd = hog_feature(fd).reshape(1,-1)
    train_data.append(fd)
    train_labels.append(1)

# Load the negative features
for filename in glob.glob(os.path.join(neg_im_path,"*.jpg")):
    fd = cv2.imread(filename,0)
    fd = cv2.resize(fd,(64,128))
    # fd = hog(fd,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(2,2))
    fd = hog_feature(fd).reshape(1,-1)
    train_data.append(fd)
    train_labels.append(0)
# split data
(trainData, testData, trainLabels, testLabels) = train_test_split(
    np.array(train_data), train_labels, test_size=0.20, random_state=0)
trainData = np.float32(trainData)
trainLabels = np.array(trainLabels)
testData = np.float32(testData)
testLabels = np.array(testLabels)
print('Data Prepared........')
print('Train Data:',len(trainData))
print('Train Labels (1,0)',len(trainLabels))
print('Test Data:',len(testData))
print('Test Labels (1,0)',len(testLabels))
print("""
Classification with SVM

""")

model = LinearSVC()
print('Training...... Support Vector Machine')
nsamples, nx, ny = trainData.shape
d2_train_dataset = trainData.reshape((nsamples,nx*ny))
model.fit(d2_train_dataset,trainLabels)

nsamples, nx, ny = testData.shape
d2_test_dataset = testData.reshape((nsamples,nx*ny))
predictions = model.predict(d2_test_dataset)
print(classification_report(testLabels, predictions))


