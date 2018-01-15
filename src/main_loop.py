from PIL import Image
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing

# Constants for file paths

HRF_ROOT_PATH = "/home/victor/Pictures/rvs-mwt/HRF/"
DR_MANUAL = "diabetic_retinopathy_manual/"
HEALTHY_MANUAL = "healthy_manual/"
WAVES_HRF_OUTPUT = "waves_hrf_output/"
RESULT = "-result"
PNG_EXTENSION = ".png"

# Constants for result names.

ANS_HEALTHY = "Healty Eye"
ANS_UNHEALTHY = "Unhealty Eye"

#
# List of classes to be recognized, considering the supervised learning technique.
#

healthy = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
unhealthy = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

training_classes = np.concatenate([healthy, unhealthy])

#
# Instantiation of a Binarizer object, in order to normalize
# all pixel values between 0 and 1.
#

binarizer = preprocessing.Binarizer()

#
# Instantiation of a LinearDiscriminantAnalysis object, in order
# to apply the LDA method and classify each sample image.
#

lda_classifier = LinearDiscriminantAnalysis()

#
# Read the training set of segmented images into a Python list,
# convert them to numpy arrays, normalize them with the binarizer
# object and turn them into one dimension arrays.
#
training_set = []


# Healthy images
for i in range(0, 15):
	img_name = str(i + 1) + PNG_EXTENSION
	img = Image.open(HRF_ROOT_PATH + HEALTHY_MANUAL + img_name)
	
	np_arr = np.array(img)
	np_arr = np_arr.reshape(-1, 1)
	np_arr = binarizer.transform(np_arr)
	np_arr = np.ravel(np_arr)
	
	training_set.append(np_arr)

# Unhealthy images
for i in range(0, 15):
	img_name = str(i + 1) + PNG_EXTENSION
	img = Image.open(HRF_ROOT_PATH + DR_MANUAL + img_name)
	
	np_arr = np.array(img)
	np_arr = np_arr.reshape(-1, 1)
	np_arr = binarizer.transform(np_arr)
	np_arr = np.ravel(np_arr)
	
	training_set.append(np_arr)
	
lda_classifier.fit(training_set, training_classes)

#
# Read the test set of segmented images into a Python list,
# convert them to numpy arrays, normalize them with the binarizer
# object, turn them into one dimension arrays and print each
# prediction of classes.
#
	
for i in range(10, 15):
	img_name = str(i + 1) + RESULT + PNG_EXTENSION
	img = Image.open(HRF_ROOT_PATH + WAVES_HRF_OUTPUT + img_name)
	
	np_arr = np.array(img)
	np_arr = np_arr.reshape(-1, 1)
	np_arr = binarizer.transform(np_arr)
	np_arr = np.ravel(np_arr)

	klass = lda_classifier.predict([np_arr])
	ans = None

	if klass == [1]:
		ans = ANS_HEALTHY
	if klass == [2]:
		ans = ANS_UNHEALTHY
	
	print('Image: ' + HRF_ROOT_PATH + WAVES_HRF_OUTPUT + img_name)
	print(ans + '\n')


