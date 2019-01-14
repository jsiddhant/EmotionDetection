################################################################################
# CSE 253: Programming Assignment 1
# Code snippet by Siddhant Jain
# Winter 2019
################################################################################

import dataloader as DL
import numpy as np
import random
import re
import PCA


class DataSet:

    def __init__(self, data, data_labels):

        # Create a Map from Label to Data
        dataMap={}
        for i in range(0,len(data_labels)):
            dataMap[data_labels[i]] = data[i]

        # Create a List of DataItem to track all attributes of each Image
        regex = re.compile(r'(\d+|\s+|_)')
        self.dataList=[]
        for lbl in data_labels:
            label = regex.split(lbl)
            self.dataList.append(DataItem(label[1], label[4], label[5], dataMap[lbl]))

        # Store List of all Subjects and Emotions
        emotionCount = dict.fromkeys([data.emotion for data in self.dataList])
        personCount = dict.fromkeys([data.subject for data in self.dataList])
        self.emotions = list(emotionCount.keys())
        self.people = list(personCount.keys())

        # Get No. of Images by Subject and Emotion
        self.get_data_stats()

        # Generate Train, Hold and Test Set for h and m emotions
        self.data_split()

    def get_data_stats(self):
        # Get Number of Images for each emotion and each person.
        emotionCount = dict.fromkeys(self.emotions)
        personCount = dict.fromkeys(self.people)

        for data in self.dataList:
            # Init Map
            if emotionCount[data.emotion] is None:
                emotionCount[data.emotion] = 1
            else:
                emotionCount[data.emotion] += 1

            if personCount[data.subject] is None:
                personCount[data.subject] = 1
            else:
                personCount[data.subject] += 1

        print("---- Number of People: " + str(len(list(personCount.keys()))))
        for key in personCount.keys():
            print("Subject: " + key + " has " + str(personCount[key]) + " pictures.")

        print("--------------------------")

        print("---- Number of Emotions: " + str(len(list(emotionCount.keys()))))
        for key in emotionCount.keys():
            print("Emotion: " + key + " has " + str(emotionCount[key]) + " pictures.")

    def get_filtered(self, person, emotion):
        return [data for data in self.dataList if data.subject in person and data.emotion in emotion]

    def data_split(self):

        people_list = self.people
        random.shuffle(people_list)
        print('------------------')

        self.trainHappy = self.get_filtered(people_list[:8], ['h'])
        print("Length Train Happy: ")
        print(len(self.trainHappy))

        self.trainSad = self.get_filtered(people_list[:8], ['m'])
        print("Length Train Sad: ")
        print(len(self.trainSad))

        self.holdHappy = self.get_filtered(people_list[8], ['h'])
        print("Length Hold Happy: ")
        print(len(self.holdHappy))

        self.holdSad = self.get_filtered(people_list[8], ['m'])
        print("Length Hold Sad: ")
        print(len(self.holdSad))

        self.testHappy = self.get_filtered(people_list[9], ['h'])
        print("Length Test Happy: ")
        print(len(self.testHappy))

        self.testSad = self.get_filtered(people_list[9], ['m'])
        print("Length Test Sad: ")
        print(len(self.testSad))


class DataItem:
    def __init__(self, subject, emotion, digit, image):
        self.subject = subject
        self.emotion = emotion
        self.digit = digit
        self.image = image


def visualize_image(image):
    I = image
    I_uint8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
    DL.display_face(I_uint8)


def eval_logistic(w,x):
    return 1 / (1 + np.exp(-np.dot(w, x)))


def update_w_logistic(t, y, x, al, w):
    update = al * np.dot(t-y, x.T)
    return w + update


def eval_error(t, y):
    return np.dot(t-y, t-y)


def norm_vec(img):
    img = img.T - img.mean(axis=1)
    img = (img.T / np.linalg.norm(img, 2, axis=1))
    return img


if __name__ == "__main__":

    #########################################################
    # PA:1 -- Q.1
    #########################################################
    # Load the images
    data_img, labels = DL.load_data()
    data_img = np.asarray(data_img)  # Convert from List to ND-Array

    # Convert to Float and Create DataSet Object
    data_img = data_img.astype(float)
    numImgs, height, width = data_img.shape
    cafe = DataSet(data_img, labels)

    # Prep Training Data

    trainTotal = list(set(cafe.trainHappy + cafe.trainSad))
    trainImages = []  # Total Training Set
    trainLabels = []  # Total Training Image Set
    for t in trainTotal:
        trainImages.append(t.image)
        trainLabels.append(1.0 if t.emotion == 'h' else 0.0)

    trainImages = np.asarray(trainImages)
    trainLabels = np.asarray(trainLabels)

    # Prep HoldOut Data

    holdTotal = list(set(cafe.holdHappy + cafe.holdSad))
    holdImages = []
    holdLabels = []
    for t in holdTotal:
        holdImages.append(t.image)
        holdLabels.append(1.0 if t.emotion == 'h' else 0.0)

    holdImages = np.asarray(holdImages)
    holdLabels = np.asarray(holdLabels)

    # Prep Test Data

    testTotal = list(set(cafe.testHappy + cafe.testSad))
    testImages = []
    testLabels = []
    for t in testTotal:
        testImages.append(t.image)
        testLabels.append(1.0 if t.emotion == 'h' else 0.0)

    testImages = np.asarray(testImages)
    testLabels = np.asarray(testLabels)

    # PCA
    n_components = 16
    eigComps, _, _ = PCA.PCA(trainImages, n_components)

    # Uncomment to show EigenFaces
    # eigFace = eigComps.reshape(n_components, height, width)
    # for i in range(0, 16):
    #     visualize_image(eigFace[i])

    #########################################################
    #########################################################

    #########################################################
    # PA:1 -- Q.2
    #########################################################
    eigComps = eigComps[0:10]

    # trainImageVec = trainImages.reshape(len(trainTotal), height*width)
    trainImageVec = trainImages.reshape(len(trainTotal), height*width)
    # trainImageVec = (trainImageVec.T / np.linalg.norm(trainImageVec, 2, axis=1)).T
    trainImageVec = norm_vec(trainImageVec)

    holdImageVec = holdImages.reshape(len(holdTotal), height * width)
    holdImageVec = norm_vec(holdImageVec)

    x_proj = np.dot(eigComps, trainImageVec.T)
    x_h_proj = np.dot(eigComps, holdImageVec.T)
    w = np.random.rand(10)*0.01
    alpha = 0.01

    for epoch in range(0, 10):
        y = eval_logistic(w, x_proj)
        yH = eval_logistic(w, x_h_proj)
        print("Epoch: " + str(epoch) + " Training Error: " + str(eval_error(trainLabels, y)) + " Val Error: " + str(eval_error(holdLabels, yH)))
        w = update_w_logistic(trainLabels, y, x_proj, alpha, w)
    y = eval_logistic(w, x_proj)
    yH = eval_logistic(w, x_h_proj)
    print(eigComps.shape)
