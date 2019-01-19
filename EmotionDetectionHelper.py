################################################################################
# Code snippet by Siddhant Jain
# Winter 2019
# Description: Support File to train Logistic and Softmax Regression for Emotion Detection
#              Required to run Jupyter Notebook for demo
################################################################################

import dataloader as DL
import numpy as np
import random
import re
import PCA
import matplotlib.pyplot as plt
import copy


class DataSet:

    def __init__(self, data, data_labels, testPeople= None, holdPeople = None, emotionList=['h','m'], height=380, width=240):

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

        people_list_train = self.people[:]

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


class DataItem:
    def __init__(self, subject, emotion, digit, image):
        self.subject = subject
        self.emotion = emotion
        self.digit = digit
        self.image = image


class TrainInstance:

    def __init__(self, dataset, parameter_num=10, emotionList=['h','s'], softmax=False):

        self.parameter_num = parameter_num
        self.trainError = np.Inf
        self.holdError = np.Inf
        self.testError = np.Inf
        self.trainErrorList = []
        self.holdErrorList = []
        h = dataset.height
        w = dataset.width
        numEmotions = len(emotionList)
        emotion_dict = {emotionList[i]:i for i in range(0, numEmotions)}
        self.softmax = softmax
        self.conf_mat = np.zeros((numEmotions, numEmotions))

        # Prep Training Data
        trainTotal = list(set(dataset.trainSet))
        trainImages = []  # Total Training Set
        trainLabels = []  # Total Training Image Set

        if not softmax:

            for t in trainTotal:
                trainImages.append(t.image)
                trainLabels.append(1.0 if t.emotion == emotionList[0] else 0.0)

        else:
            for t in trainTotal:
                trainImages.append(t.image)
                l = [0 for i in range(0,numEmotions)]
                l[emotion_dict[t.emotion]] = 1
                trainLabels.append(l)

        trainImages = np.asarray(trainImages)
        self.trainLabels = np.asarray(trainLabels)

        # Prep HoldOut Data

        holdTotal = list(set(dataset.holdSet))
        holdImages = []
        holdLabels = []

        if not softmax:

            for t in holdTotal:
                holdImages.append(t.image)
                holdLabels.append(1.0 if t.emotion == emotionList[0] else 0.0)

        else:
            for t in holdTotal:
                holdImages.append(t.image)
                l = [0 for i in range(0,numEmotions)]
                l[emotion_dict[t.emotion]] = 1
                holdLabels.append(l)

        holdImages = np.asarray(holdImages)
        self.holdLabels = np.asarray(holdLabels)

        # Prep Test Data

        testTotal = list(set(dataset.testSet))
        testImages = []
        testLabels = []
        if not softmax:

            for t in testTotal:
                testImages.append(t.image)
                testLabels.append(1.0 if t.emotion == emotionList[0] else 0.0)

        else:
            for t in testTotal:
                testImages.append(t.image)
                l = [0 for i in range(0,numEmotions)]
                l[emotion_dict[t.emotion]] = 1
                testLabels.append(l)

        testImages = np.asarray(testImages)
        self.testLabels = np.asarray(testLabels)

        eigComps, _, _ = PCA.PCA(trainImages, parameter_num)
        self.eigComps = eigComps[0:parameter_num]

        trainImageVec = trainImages.reshape(len(trainTotal), h * w)
        # self.trainImageVec = norm_vec(trainImageVec)
        self.trainImageVec = trainImageVec

        holdImageVec = holdImages.reshape(len(holdTotal), h * w)
        # self.holdImageVec = norm_vec(holdImageVec)
        self.holdImageVec = holdImageVec

        testImageVec = testImages.reshape(len(testTotal), h * w)
        # self.testImageVec = norm_vec(testImageVec)
        self.testImageVec = testImageVec

        # self.inputTrain = np.dot(self.eigComps, self.trainImageVec.T)
        self.inputTrain = norm_vec(np.dot(self.eigComps, self.trainImageVec.T))
        # self.inputHold = np.dot(self.eigComps, self.holdImageVec.T)
        self.inputHold = norm_vec(np.dot(self.eigComps, self.holdImageVec.T))
        # self.inputTest = np.dot(self.eigComps, self.testImageVec.T)
        self.inputTest = norm_vec(np.dot(self.eigComps, self.testImageVec.T))

    def batch_gradient_descent(self, model, epochs):

        x_proj = self.inputTrain
        x_h_proj = self.inputHold

        for epoch in range(0, epochs):

            y = model.eval(x_proj)
            yH = model.eval(x_h_proj)
            cross_entropy_loss(self.trainLabels, y)

            self.trainError = cross_entropy_loss(self.trainLabels, y)
            self.holdError = cross_entropy_loss(self.holdLabels, yH)
            self.trainErrorList.append(self.trainError)
            self.holdErrorList.append(self.holdError)
            print("=========================================================================================")
            print("Epoch: " + str(epoch) + " Training Error: " + str(
                self.trainError) + " Val Error: " + str(self.holdError))
            print("=========================================================================================")
            model.early_stopping(cross_entropy_loss(self.holdLabels, yH))
            model.update_w(self.trainLabels, y, x_proj)

            # Implement Graphing of Training and Hold Out Error

    def stochastic_gradient_descent(self, model, epochs):

        x_proj = self.inputTrain
        x_h_proj = self.inputHold

        idx = [num for num in range(0, x_proj.shape[1])]

        for epoch in range(0, epochs):

            idx = np.random.permutation(idx)

            for ind in idx:

                y = model.eval(x_proj)
                yH = model.eval(x_h_proj)
                self.trainError = cross_entropy_loss(self.trainLabels[ind], y[ind])
                self.holdError = cross_entropy_loss(self.holdLabels, yH)
                cross_entropy_loss(self.trainLabels[ind], y[ind])
                print("=========================================================================================")
                # print("Epoch: " + str(epoch) + " SSE Training Error: " + str(
                #     eval_error(self.trainLabels, y)) + " SSE Val Error: " + str(eval_error(self.holdLabels, yH)))

                print("Epoch: " + str(epoch) + " Training Error: " + str(
                    self.trainError) + " Val Error: " + str(self.holdError))
                print("=========================================================================================")
                model.early_stopping(cross_entropy_loss(self.holdLabels, yH))
                model.update_w(self.trainLabels[ind], y[ind], x_proj[:, ind])

            self.trainError = cross_entropy_loss(self.trainLabels, y)
            self.holdError = cross_entropy_loss(self.holdLabels, yH)
            self.trainErrorList.append(self.trainError)
            self.holdErrorList.append(self.holdError)

    def get_test_error(self, model, softmax =False):
        if softmax:
            count =0.0
            correct=0.0
            for i in range(0, len(self.testLabels)):
                count=count+1
                lbl_test = np.round(model.eval(self.inputTest, w=model.w_final))
                gtruth = np.argwhere(self.testLabels[i] == 1)[0, 0]
                # pred = np.argwhere(lbl_test[i] == 1)[0, 0]
                pred = np.argmax(lbl_test[i])
                correct = correct +1 if gtruth == np.argmax(lbl_test[i]) else correct
        else:
            count = 0.0
            correct = 0.0
            for i in range(0, len(self.testLabels)):
                count = count+1
                label_test = 1 if model.eval(self.inputTest[:, i], w=model.w_final) > 0.5 else 0
                correct = correct +1 if self.testLabels[i] == label_test else correct

        self.testError = 1 - np.float(correct/count)

    def gen_conf_mat(self, model):

        for i in range(0, len(self.testLabels)):
            lbl_test = np.round(model.eval(self.inputTest, w=model.w_final))
            gtruth = np.argwhere(self.testLabels[i] == 1)[0, 0]
            pred = np.argmax(lbl_test[i])
            self.conf_mat[gtruth, pred] = self.conf_mat[gtruth, pred]+1


class LogisticModel:

    def __init__(self, learningRate, parameter_num):
        self.w = np.random.rand(parameter_num)
        self.w_final = self.w
        self.alpha = learningRate
        self.minHoldError = np.Inf

    def eval(self, input, w=None):
        weight = self.w if w is None else w
        return 1 / (1 + np.exp(-np.dot(weight, input)))

    def update_w(self, target, y, x):
        update = self.alpha * np.dot(target - y, x.T)
        self.w = self.w + update

    def early_stopping(self, hold_error):

        self.minHoldError = hold_error if hold_error < self.minHoldError else self.minHoldError
        self.w_final = self.w if hold_error <= self.minHoldError else self.w_final


class SoftmaxModel:

    def __init__(self, learningRate, numparams=20, numemotions=6):

        self.w = np.random.rand(numparams, numemotions)
        self.w_final = self.w
        self.alpha = learningRate
        self.minHoldError = np.Inf

    def eval(self, input, w=None):
        weight = self.w if w is None else self.w
        ans = np.dot(weight.T, input)
        return (np.exp(ans) / np.sum(np.exp(ans), axis=0)).T

    def update_w(self, target, y, x):
        # update = self.alpha * np.dot(x, (target-y))
        update = self.alpha * np.outer(x, target-y) if y.ndim == 1 else self.alpha * np.dot(x, (target-y))
        self.w = self.w + update

    def early_stopping(self, hold_error):
        self.w_final = self.w if hold_error < self.minHoldError else self.w_final
        self.minHoldError = hold_error if hold_error < self.minHoldError else self.minHoldError


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


def cross_entropy_loss(label, prediction, eps = 1e-12):

    prediction = np.clip(prediction, eps, 1. - eps)
    n = prediction.shape[0]
    ce = -np.sum(label*np.log(prediction+1e-9))/n
    return ce


def create_train_plot(epochs, train_err, hold_err, title):

    fig = plt.figure()
    fig.suptitle(title)
    ep = [i+1 for i in range(0, epochs)]
    train_err = np.asarray(train_err)
    hold_err = np.asarray(hold_err)

    train_avg = np.mean(train_err, axis=0)
    hold_avg = np.mean(hold_err, axis=0)
    plt.plot(ep, train_avg, '-b', label='Avg Train Loss')
    plt.plot(ep, hold_avg, '--r', label='Avg Hold Loss')

    # epL = [i*2 for i in range(0, int(epochs/2))]
    epL =[i*2 for i in range(0, int(epochs/2))] if epochs < 50 else [10, 20, 30, 40, 50]
    train_err = np.asarray(train_err)
    hold_err = np.asarray(hold_err)

    for i in epL:
        x = i
        y = np.mean(train_err[:, i-1])
        e = np.std(train_err[:, i-1])
        plt.errorbar(x, y, e, linestyle='None', marker='o', color='blue', elinewidth=2.0)
        y = np.mean(hold_err[:, i-1])
        e = np.std(hold_err[:, i-1])
        plt.errorbar(x, y, e, linestyle='None', marker='o', color='red', elinewidth=1.0)

    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend(loc='upper right')
    plt.show()


def show_conf_mat(conf_mat, emotion_list):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, interpolation='nearest')
    fig.colorbar(cax)
    fig.suptitle('Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    ax.set_xticklabels([''] + emotion_list)
    ax.set_yticklabels([''] + emotion_list)

    plt.show()


def generate_eig_face(emotion_list  ):
    data_img, labels = DL.load_data()
    data_img = np.asarray(data_img)

    # Convert to Float and Create DataSet Object
    data_img = data_img.astype(float)
    _, height_img, width_img = data_img.shape
    eigen_data = DataSet(data_img, labels, emotionList=emotion_list)
    trainPCA = TrainInstance(eigen_data, 48, emotionList=emotion_list)

    eigFace = trainPCA.eigComps.reshape(48, height_img, width_img)
    fig = plt.figure(figsize=(2, 3))
    plt.suptitle("Top 6 Eigen Faces after PCA")
    for i in range(0, 6):
        # visualize_image(eigFace[i])
        ax = fig.add_subplot(2, 3, i + 1)
        plt.imshow(eigFace[i], cmap='gray')
        ax.set_title(str(i+1))
        plt.axis('off')
    plt.show()


def visualize_weights(eig_comps, weights, emotion_list, height=380, width=240):

    emotion_vis = np.dot(weights.T, eig_comps)
    fig = plt.figure(figsize=(2, 3))
    for i in range(0, len(emotion_list)):
        ax = fig.add_subplot(2, 3, i+1)
        eigFace = emotion_vis[i].reshape(height, width)
        plt.imshow(eigFace, cmap='gray')
        ax.set_title("Emotion: " + emotion_list[i])
        plt.axis('off')
    plt.show()


if __name__ == "__main__":

    data_img, labels = DL.load_data()
    data_img = np.asarray(data_img)  # Convert from List to ND-Array
    #
    # # # Convert to Float and Create DataSet Object
    data_img = data_img.astype(float)
    numImgs, height, width = data_img.shape
    cafe_data = DataSet(data_img, labels)

    peopleList = cafe_data.people[:]

    errorList = []
    trainingErrorList = []
    holdErrorList = []
    emotionList = ['h', 'a', 's', 'f', 'd', 'm']
    conf_mat = np.zeros((len(emotionList), len(emotionList)))

    for person in peopleList:
        cafe = DataSet(data_img, labels, person, emotionList=emotionList)  # Select Emotions to run for.
        param_num = 30
        epochs = 50
        learning_rate = 10e-1
        softmaxTrain = TrainInstance(cafe, param_num, emotionList=emotionList,
                                            softmax=True)  # Select Emotions to run for.
        softMaxRegress = SoftmaxModel(learning_rate, param_num, len(emotionList))
        # softmaxTrain.batch_gradient_descent(softMaxRegress, epochs)
        softmaxTrain.stochastic_gradient_descent(softMaxRegress, epochs)
        trainingErrorList.append(softmaxTrain.trainErrorList)
        holdErrorList.append(softmaxTrain.holdErrorList)

        softmaxTrain.get_test_error(softMaxRegress, softmax=True)

        print("===================================================")
        print("Test Error is: " + str(softmaxTrain.testError))
        print("===================================================")
        errorList.append(softmaxTrain.testError)
        softmaxTrain.gen_conf_mat(softMaxRegress)
        conf_mat = conf_mat + softmaxTrain.conf_mat

    print("AVG ERROR: " + str(np.asarray(errorList).mean()) + ' (' + str(np.std(errorList)) + ') ')
