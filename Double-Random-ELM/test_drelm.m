% Call function double_random_elm

[TrainingT, TestingT, TrainingAccuracyNum, TestingAccuracyNum] = double_random_elm('../Dataset/housing1_train.txt', '../Dataset/housing1_test.txt', 1000, 0, 0, 1, 1);


% Usage: elm(TrainingData_File, TestingData_File, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, NumberofHiddenNeurons, c1, c2, option_f1, option_f2)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% c1			- Tikhonov regulation c1
% c2			- Tikhonov regulation c2
% option_f1		- Option for choosing activation function f1. The option range: [1 5]
%                         1: 'sig' for Sigmoidal function
%                         2: 'sin' for Sine function
%                         3: 'hardlim' for Hardlim function
%                         4: 'tribas' for Triangular basis function
%                         5: 'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
% option_f2		- Option for choosing function f2. The option range: [1 2]
% 	                  1: 'f2(x) = x'
%			  2: 'f2(x) = x ^ n'
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification
%
