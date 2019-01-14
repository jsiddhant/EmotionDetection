

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import emotionScripts as em
import dataloader as DL
import PCA as p
import numpy as np

from PIL import Image

data_img, labels = DL.load_data()
data_img = np.asarray(data_img)  # Convert from List to ND-Array

# Convert to Float and Reshape for PCA Operations
data_img = data_img.astype(float)
numImgs, height, width = data_img.shape
data_pca = data_img.reshape(numImgs, height*width)
cafe = em.DataSet(data_img, labels)

trainTotal = list(set(cafe.trainHappy + cafe.trainSad))

trainImages = np.asarray([t.image for t in trainTotal])
trainImgPCA = trainImages.reshape(len(trainTotal), height * width)

############################################################
n_components = len(trainTotal)
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(trainImgPCA)
eigFace, _, _ = p.PCA(trainImages, 16)
norm_eig = np.linalg.norm(eigFace, 2,axis=0)

eigFace = eigFace/norm_eig

eigFace = (eigFace.T).reshape(n_components,height,width)


eigenfaces = pca.components_.reshape((n_components, height, width))


print("Projecting the input data on the eigenfaces orthonormal basis")
I = eigenfaces[0]
I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)

# img = Image.fromarray(I8)
DL.display_face(I8)

I = eigFace[0]
I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)

# img = Image.fromarray(I8)

DL.display_face(I8)

print(p.test_PCA(eigFace))

# # Prep Training Data
#
# trainTotal = list(set(cafe.trainHappy + cafe.trainSad))
# trainImages = []  # Total Training Set
# trainLabels = []  # Total Training Image Set
# for t in trainTotal:
#     trainImages.append(t.image)
#     trainLabels.append(1.0 if t.emotion == 'h' else 0.0)
#
# trainImages = np.asarray(trainImages)
# trainLabels = np.asarray(trainLabels)
#
# # Prep HoldOut Data
#
# holdTotal = list(set(cafe.holdHappy + cafe.holdSad))
# holdImages = []
# holdLabels = []
# for t in holdTotal:
#     holdImages.append(t.image)
#     holdLabels.append(1.0 if t.emotion == 'h' else 0.0)
#
# holdImages = np.asarray(holdImages)
# holdLabels = np.asarray(holdLabels)
#
# # Prep Test Data
#
# testTotal = list(set(cafe.testHappy + cafe.testSad))
# testImages = []
# testLabels = []
# for t in testTotal:
#     testImages.append(t.image)
#     testLabels.append(1.0 if t.emotion == 'h' else 0.0)
#
# testImages = np.asarray(testImages)
# testLabels = np.asarray(testLabels)

# # PCA
# n_components = 16
# eigComps, _, _ = PCA.PCA(trainImages, n_components)

# # Uncomment to show EigenFaces
# # eigFace = eigComps.reshape(n_components, height, width)
# # for i in range(0, 16):
# #     visualize_image(eigFace[i])
#
# #########################################################
# #########################################################
#
# #########################################################
# # PA:1 -- Q.2
# #########################################################
# parameter_num = 10
# eigComps = eigComps[0:parameter_num]

# # trainImageVec = trainImages.reshape(len(trainTotal), height*width)
# trainImageVec = trainImages.reshape(len(trainTotal), height*width)
# # trainImageVec = (trainImageVec.T / np.linalg.norm(trainImageVec, 2, axis=1)).T
# trainImageVec = norm_vec(trainImageVec)
#
# holdImageVec = holdImages.reshape(len(holdTotal), height * width)
# holdImageVec = norm_vec(holdImageVec)

# x_proj = np.dot(eigComps, trainImageVec.T)
# x_h_proj = np.dot(eigComps, holdImageVec.T)
# w = np.random.rand(parameter_num)
# alpha = 0.1  # 0.01
#
# for epoch in range(0, 10):
#     y = eval_logistic(w, x_proj)
#     yH = eval_logistic(w, x_h_proj)
#     cross_entropy_loss(trainLabels, y)
#     print("Epoch: " + str(epoch) + " SSE Training Error: " + str(eval_error(trainLabels, y)) + " SSE Val Error: " + str(eval_error(holdLabels, yH)))
#     print("Epoch: " + str(epoch) + " Training Error: " + str(cross_entropy_loss(trainLabels, y)) + " Val Error: " + str(cross_entropy_loss(holdLabels, yH)))
#     w = update_w_logistic(trainLabels, y, x_proj, alpha, w)
# y = eval_logistic(w, x_proj)
# yH = eval_logistic(w, x_h_proj)
# print(eigComps.shape)
#
# ## Update Loss function to CrossEntropy
