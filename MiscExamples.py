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
#     w = update_w_logistic(trainLabels, y, x_proj, alpha, w)
# y = eval_logistic(w, x_proj)
# yH = eval_logistic(w, x_h_proj)
# print(eigComps.shape)
#
# ## Update Loss function to CrossEntropy


# # generate_eig_face()
# #########################################################
# # PA:1 -- Q.1
# # #########################################################
# # Load the images
# data_img, labels = DL.load_data()
# data_img = np.asarray(data_img)  # Convert from List to ND-Array
#
# # Convert to Float and Create DataSet Object
# data_img = data_img.astype(float)
# numImgs, height, width = data_img.shape
# cafe_data = DataSet(data_img, labels)
#
# peopleList = cafe_data.people[:]
#
# errorList = []
# trainingErrorList = []
# holdErrorList = []
#
# for person in peopleList:
#     cafe = DataSet(data_img, labels, person, emotionList=['h', 'm'])  # Select Emotions to run for.
#     param_num = 10
#     epoch = 10
#     logisticTrain = TrainInstance(cafe, param_num, emotionList=['h', 'm'])  # Select Emotions to run for.
#     logisticRegression = LogisticModel(10e-2, param_num)
#     logisticTrain.batch_gradient_descent(logisticRegression, epoch)
#
#     trainingErrorList.append(logisticTrain.trainErrorList)
#     holdErrorList.append(logisticTrain.holdErrorList)
#
#     logisticTrain.get_test_error(logisticRegression)
#
#     print("===================================================")
#     print("Test Error is: " + str(logisticTrain.testError))
#     print("===================================================")
#     errorList.append(logisticTrain.testError)
#
# print('Average Test Error over 10 runs is: ' + str(np.mean(errorList)))
#
# # # create_train_plot(epoch, trainingErrorList, holdErrorList)    # Load the images
# data_img, labels = DL.load_data()
# data_img = np.asarray(data_img)  # Convert from List to ND-Array
# #
# # # Convert to Float and Create DataSet Object
# data_img = data_img.astype(float)
# numImgs, height, width = data_img.shape
# cafe_data = DataSet(data_img, labels)
#
# peopleList = cafe_data.people[:]
#
# errorList = []
# trainingErrorList = []
# holdErrorList = []
#
# for person in peopleList:
#     cafe = DataSet(data_img, labels, person, emotionList=['a', 's'])  # Select Emotions to run for.
#     param_num = 2
#     alpha = 10e-1
#     logisticTrain = TrainInstance(cafe, param_num, emotionList=['a', 's'])  # Select Emotions to run for.
#     logisticRegression = LogisticModel(alpha, param_num)
#     logisticTrain.batch_gradient_descent(logisticRegression, 10)
#
#     trainingErrorList.append(logisticTrain.trainErrorList)
#     holdErrorList.append(logisticTrain.holdErrorList)
#
#     logisticTrain.get_test_error(logisticRegression)
#
#     print("===================================================")
#     print("Test Error is: " + str(logisticTrain.testError))
#     print("===================================================")
#     errorList.append(logisticTrain.testError)
#
# print('Average Test Error over 10 runs is: ' + str(np.mean(errorList)) + ' (' + str(np.std(errorList)) + ') ')
# create_train_plot(10, trainingErrorList, holdErrorList, 'Train & Hold Loss alpha: ' + str(alpha) + ' Params: ' + str(param_num))

# # #########################################################
# # # PA:1 -- END
# # #########################################################

# # # # #########################################################
# # # # # Stoc vs. Batch
# # # # #########################################################
# # #
# # # Load the images
# data_img, labels = DL.load_data()
# data_img = np.asarray(data_img)  # Convert from List to ND-Array
# #
# # # # Convert to Float and Create DataSet Object
# data_img = data_img.astype(float)
# numImgs, height, width = data_img.shape
# cafe_data = DataSet(data_img, labels)
#
# peopleList = cafe_data.people[:]
#
# errorList = []
# trainingErrorList = []
# holdErrorList = []
# errorList_sg = []
# trainingErrorList_sg = []
# holdErrorList_sg = []
#
# emotionList = ['h', 'a', 's', 'f', 'd', 'm']
# conf_mat = np.zeros((len(emotionList), len(emotionList)))
#
# for person in peopleList:
#     cafe = DataSet(data_img, labels, person, emotionList=emotionList)  # Select Emotions to run for.
#     param_num = 20
#     epochs = 50
#     learning_rate = 1
#
#     softmaxTrain = TrainInstance(cafe, param_num, emotionList=emotionList, softmax=True)  # Select Emotions to run for.
#     softMaxRegress = SoftmaxModel(learning_rate, param_num, len(emotionList))
#     softMaxTrain_sg = copy.deepcopy(softmaxTrain)
#     softMaxRegress_sg = copy.deepcopy(softMaxRegress)
#
#     softmaxTrain.batch_gradient_descent(softMaxRegress, epochs)
#     softMaxTrain_sg.stochastic_gradient_descent(softMaxRegress_sg, epochs)
#
#     trainingErrorList.append(softmaxTrain.trainErrorList)
#     holdErrorList.append(softmaxTrain.holdErrorList)
#
#     trainingErrorList_sg.append(softMaxTrain_sg.trainErrorList)
#     holdErrorList_sg.append(softMaxTrain_sg.holdErrorList)
#
#     softmaxTrain.get_test_error(softMaxRegress, softmax=True)
#     softMaxTrain_sg.get_test_error(softMaxRegress_sg, softmax=True)
#     # print("===================================================")
#     # print("Test Error is: " + str(softmaxTrain.testError))
#     # print("===================================================")
#     errorList.append(softmaxTrain.testError)
#     errorList_sg.append(softMaxTrain_sg.testError)
#
#     # softmaxTrain.gen_conf_mat(softMaxRegress)
#     # conf_mat = conf_mat + softmaxTrain.conf_mat
#
# print("AVG ERROR: " + str(np.asarray(errorList).mean()) + ' (' + str(np.std(errorList)) + ') ')
#
# fig = plt.figure()
# fig.suptitle('Stochastic GD vs Batch GD')
# ep = [i+1 for i in range(0, epochs)]
# train_err = np.asarray(trainingErrorList)
# hold_err = np.asarray(holdErrorList)
#
# train_avg = np.mean(train_err, axis=0)
# hold_avg = np.mean(hold_err, axis=0)
# p1, =plt.plot(ep, train_avg, '-b', label='Batch Train Loss')
# p2, =plt.plot(ep, hold_avg, '--r', label='Batch Hold Loss')
#
# train_err = np.asarray(trainingErrorList_sg)
# hold_err = np.asarray(holdErrorList_sg)
#
# train_avg = np.mean(train_err, axis=0)
# hold_avg = np.mean(hold_err, axis=0)
# p3, =plt.plot(ep, train_avg, '-g', label='SGD Train Loss')
# p4, =plt.plot(ep, hold_avg, '--y', label='SGD Hold Loss')
#
# plt.legend([p1,p2,p3,p4],["Batch Train Loss", "Batch Hold Loss", "SGD Train Loss", "SGD Hold Loss"])
# plt.show()
#
#
# # create_train_plot(epochs, trainingErrorList, holdErrorList, 'Train & Hold Loss alpha: ' + str(learning_rate) + ' Params: ' + str(param_num))
# # show_conf_mat(conf_mat / len(peopleList), emotionList)
#
# # visualize_weights(softmaxTrain.eigComps, softMaxRegress.w_final, emotionList)
#
#
# # # Uncomment below to show EigenFaces
# # # show_eigen_faces(logisticTrain.eigComps)