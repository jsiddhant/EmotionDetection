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

    def __init__(self, data, data_labels, height=380, width=240, testPeople= None, holdPeople = None, emotionList=['h','m']):

        self.height = height
        self.width = width
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

        # Create a set of training data by first selecting people in the 80:10:10 ratio.
        # TestPeople / HoldPeople can be used optionally specifying people to keep for respective set
        # Otherwise one is chosen randomly.

        people_list_train = self.people

        testPeople = random.choice(people_list_train) if testPeople is None else testPeople
        people_list_train.remove(testPeople)
        holdPeople = random.choice(people_list_train) if holdPeople is None else holdPeople
        people_list_train.remove(holdPeople)

        self.trainSet = self.get_filtered(people_list_train, emotionList)
        self.testSet = self.get_filtered(testPeople, emotionList)
        self.holdSet = self.get_filtered(holdPeople, emotionList)

        # # Get No. of Images by Subject and Emotion
        # self.get_data_stats()
        #
        # # Generate Train, Hold and Test Set for h and m emotions
        # self.data_split()

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

    # def createSets(self, testPeople= None, holdPeople = None, emotionList=['h','m']):
    #
    #     # Create a set of training data by first selecting people in the 80:10:10 ratio.
    #     # TestPeople / HoldPeople can be used optionally specifying people to keep for respective set
    #     # Otherwise one is chosen randomly.
    #
    #     people_list_train = self.people()
    #
    #     testPeople = random.choice(people_list_train) if testPeople is None else testPeople
    #     people_list_train.remove(testPeople)
    #     holdPeople = random.choice(people_list_train) if holdPeople is None else holdPeople
    #     people_list_train.remove(holdPeople)
    #
    #     self.trainSet = self.get_filtered(people_list_train, emotionList)
    #     self.testSet = self.get_filtered(testPeople, emotionList)
    #     self.holdSet = self.get_filtered(holdPeople, emotionList)
    #
    # def data_split(self, testPeople = None, testEmotions = ['h','m']):
    #
    #     people_list = self.people
    #     random.shuffle(people_list)
    #     print('------------------')
    #
    #     self.trainHappy = self.get_filtered(people_list[:8], ['h'])
    #     print("Length Train Happy: ")
    #     print(len(self.trainHappy))
    #
    #     self.trainSad = self.get_filtered(people_list[:8], ['m'])
    #     print("Length Train Sad: ")
    #     print(len(self.trainSad))
    #
    #     self.holdHappy = self.get_filtered(people_list[8], ['h'])
    #     print("Length Hold Happy: ")
    #     print(len(self.holdHappy))
    #
    #     self.holdSad = self.get_filtered(people_list[8], ['m'])
    #     print("Length Hold Sad: ")
    #     print(len(self.holdSad))
    #
    #     self.testHappy = self.get_filtered(people_list[9], ['h'])
    #     print("Length Test Happy: ")
    #     print(len(self.testHappy))
    #
    #     self.testSad = self.get_filtered(people_list[9], ['m'])
    #     print("Length Test Sad: ")
    #     print(len(self.testSad))


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


def eval_error(t, y):
    return np.dot(t-y, t-y)


def norm_vec(img):
    img = img.T - img.mean(axis=1)
    img = (img.T / np.linalg.norm(img, 2, axis=1))
    return img


def cross_entropy_loss(label,prediction, eps = 1e-12):
    prediction = np.clip(prediction, eps, 1. - eps)
    n = prediction.shape[0]
    ce = -np.sum(label*np.log(prediction+1e-9))/n
    return ce


class TrainInstance:

    def __init__(self, dataset, parameter_num= 10):

        self.parameter_num = parameter_num
        self.trainError = np.Inf
        self.holdError = np.Inf
        self.testError = np.Inf

        h = dataset.height
        w = dataset.width

        # Prep Training Data
        trainTotal = list(set(dataset.trainSet))
        trainImages = []  # Total Training Set
        trainLabels = []  # Total Training Image Set

        for t in trainTotal:
            trainImages.append(t.image)
            trainLabels.append(1.0 if t.emotion == 'h' else 0.0)

        trainImages = np.asarray(trainImages)
        self.trainLabels = np.asarray(trainLabels)

        # Prep HoldOut Data

        holdTotal = list(set(dataset.holdSet))
        holdImages = []
        holdLabels = []
        for t in holdTotal:
            holdImages.append(t.image)
            holdLabels.append(1.0 if t.emotion == 'h' else 0.0)

        holdImages = np.asarray(holdImages)
        self.holdLabels = np.asarray(holdLabels)

        # Prep Test Data

        testTotal = list(set(dataset.testSet))
        testImages = []
        testLabels = []
        for t in testTotal:
            testImages.append(t.image)
            testLabels.append(1.0 if t.emotion == 'h' else 0.0)

        testImages = np.asarray(testImages)
        self.testLabels = np.asarray(testLabels)

        eigComps, _, _ = PCA.PCA(trainImages, parameter_num)
        self.eigComps = eigComps[0:parameter_num]

        trainImageVec = trainImages.reshape(len(trainTotal), h * w)
        self.trainImageVec = norm_vec(trainImageVec)

        holdImageVec = holdImages.reshape(len(holdTotal), h * w)
        self.holdImageVec = norm_vec(holdImageVec)

        testImageVec = testImages.reshape(len(testTotal), h * w)
        self.testImageVec = norm_vec(testImageVec)

        self.inputTrain = np.dot(self.eigComps, self.trainImageVec.T)
        self.inputHold = np.dot(self.eigComps, self.holdImageVec.T)
        self.inputTest = np.dot(self.eigComps, self.testImageVec.T)

    def batch_gradient_descent(self, model, epochs):

        x_proj = self.inputTrain
        x_h_proj = self.inputHold

        for epoch in range(0, epochs):

            y = model.eval(x_proj)
            yH = model.eval(x_h_proj)
            cross_entropy_loss(self.trainLabels, y)

            self.trainError = cross_entropy_loss(self.trainLabels, y)
            self.holdError = cross_entropy_loss(self.holdLabels, yH)
            print("=========================================================================================")
            print("Epoch: " + str(epoch) + " SSE Training Error: " + str(
                eval_error(self.trainLabels, y)) + " SSE Val Error: " + str(eval_error(self.holdLabels, yH)))

            print("Epoch: " + str(epoch) + " Training Error: " + str(
                self.trainError) + " Val Error: " + str(self.holdError))
            print("=========================================================================================")
            model.early_stopping(cross_entropy_loss(self.holdLabels, yH))
            model.update_w(self.trainLabels, y, x_proj)

            # Implement Graphing of Training and Hold Out Error

    def get_test_error(self, model):
        # count = 0
        # correct = 0
        # for i in range(0, len(self.holdLabels)):
        #     count = count+1
        #     label_test = 1 if model.eval(self.inputHold[:, i]) > 0.5 else 0
        #     correct = correct +1 if self.holdLabels[i] == label_test else correct
        #
        # self.testError = 1- np.float(correct/count)

        count = 0
        correct = 0
        for i in range(0, len(self.testLabels)):
            count = count+1
            label_test = 1 if model.eval(self.inputTest[:, i]) > 0.5 else 0
            correct = correct +1 if self.testLabels[i] == label_test else correct

        self.testError = 1 - np.float(correct/count)

class LogisticModel:

    def __init__(self, learningRate, parameter_num):
        self.w = np.random.rand(parameter_num)*0.1
        self.w_final = self.w
        self.alpha = learningRate
        self.minHoldError = np.Inf

    def eval(self, input):
        return 1 / (1 + np.exp(-np.dot(self.w, input)))

    def update_w(self, target, y, x):
        update = self.alpha * np.dot(target - y, x.T)
        self.w = self.w + update

    def early_stopping(self, hold_error):

        self.minHoldError = hold_error if hold_error < self.minHoldError else self.minHoldError
        self.w_final = self.w if hold_error < self.minHoldError else self.w_final



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

    param_num = 10
    logisticTrain = TrainInstance(cafe, param_num)
    logisticRegression = LogisticModel(0.1, param_num)
    logisticTrain.batch_gradient_descent(logisticRegression, 10)

    logisticTrain.get_test_error(logisticRegression)

    print("===================================================")
    print("Test Error is: " + str(logisticTrain.testError))
    print("===================================================")

