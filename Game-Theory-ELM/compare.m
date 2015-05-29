%%%%%%%%%% This code compares the performance of our proposed regulation methods of ELM with the original one, and the code is written by Ruofan Kong.

%%%%%%%%%% Set up the regulation vector c containing all possible values of c.
index_c = 1;	% Initialize the index of vector c.
start_c = -20;	% Initialize the start index of exponential function 2^i with i = start_c.
end_c = 20;	% Initialize the end index of exponential function 2^i with i = end_c.
c = 2 .^ [start_c : 1 : end_c];	% Generate all possible values of c using exponential function 2^i.

%%%%%%%%%% Initialize the remaining parameters: training dataset, testing dataset, Elm-type, NumberofHiddenNeurons, option and random seed.
% Training Data File
TrainingDataFile = '../Dataset/australian1_train.txt';
% Testing Data File
TestingDataFile = '../Dataset/australian1_test.txt';
% Elm-type: 0 - Regression; 1 - Classification
Elm_Type = 1;
% NumberofHiddenNeurons
NumberofHiddenNeurons = 1000;
% option: choosing activation function f. The option range: [1 5].
% 1 - sigmoid function
% 2 - sin function
% 3 - hardlim function
% 4 - tribas function
% 5 - radbas function
option = 1;
% The seed value to control the start value random generated matrix.
seed = 30;	

%%%%%%%%%% Compare Parameters
% Implement the game theory based ELM
p_rate = 100000;
[time_gt_elm, test_accuracy_gt_elm] = elm_gt(TrainingDataFile, TestingDataFile, Elm_Type, NumberofHiddenNeurons, option, c, seed, p_rate);
% Usage: [TotalTime, TestingAccuracy] = elm_gt(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, option, c, seed, p_rate)



% Implement the original ELM
test_accuracy_origin_elm = zeros(1, size(c,2));
time_origin_elm = 0;
for i = 1 : size(c, 2)
	[t, accuracy] = elm(TrainingDataFile, TestingDataFile, Elm_Type, NumberofHiddenNeurons, option, c(1,i), seed);
	% Usage: [TotalTime, TestingAccuracy, B] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, option, c, seed)
	time_origin_elm += t;
	test_accuracy_origin_elm(1, i) = accuracy;
end

time_gt_elm
time_origin_elm
test_accuracy_gt_elm
test_accuracy_origin_elm
plot(log(c)/log(2), test_accuracy_gt_elm, '-r', log(c)/log(2), test_accuracy_origin_elm, '-b');
