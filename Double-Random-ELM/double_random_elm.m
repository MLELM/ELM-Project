function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = double_random_elm(TrainingData_File, TestingData_File, NumberofHiddenNeurons, c1, c2, option_f1, option_f2)

% Usage: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = double_random_elm(TrainingData_File, TestingData_File, NumberofHiddenNeurons, c1, c2, option_f1, option_f2)
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
%
    %%%%    The code is revised by Ruofan (Ryan) Kong based on the original version.
    %%%%    Original Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004

%%%%%%%%%%% Load training dataset
train_data=load(TrainingData_File);
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=load(TestingData_File);
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);
NumberofOutputNeurons=1;	%   The number of output neurons in regression

%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

%%%%%%%%%%% Step 1: Initialize the InputWeight A1, BiasofHiddenNeurons b and feed-forward training input X in paper.
InputWeight = rand(NumberofHiddenNeurons, NumberofInputNeurons) * 2 - 1;	%   Randomly select matrix A1 in paper
BiasofHiddenNeurons = rand(NumberofHiddenNeurons, 1); %		Vector b in paper
tempH = InputWeight * P;			      % 	A1 * X in paper
ind = ones(1, NumberofTrainingData);		      % 	Vector 1^T in paper
BiasMatrix = BiasofHiddenNeurons(:, ind);             % 	(1^T) ** b where ** means the kronecker product in paper
tempH=tempH+BiasMatrix;                               %		A1 * X + (1^T) ** b in paper
% Variable option is for choosing activation function f1. The option range: [1 5].
% option 1:	sigmoid function
% option 2:	sin function
% option 3:	hardlim function
% option 4:	tribas function
% option 5:	radbas function
f1 = {@(x)1 ./ (1 + exp(-x)); @(x)sin(x); @(x)double(hardlim(x)); @(x)tribas(x); @(x)radbas(x)};
H = f1{option_f1}(tempH);	% Choose activation function based on the option. 
clear tempH;	% Release tempH

%%%%%%%%%%% Step 2: Initialize the OutputWeight A3, and feed-backward training output Z in paper.
OutputWeight = rand(NumberofOutputNeurons, NumberofHiddenNeurons) * 2 - 1;      %   Randomly select matrix A3 in paper
K1 = pinv(OutputWeight' * OutputWeight + c1 * eye(size(OutputWeight, 2))) * OutputWeight';	%  K1 = (A3^T * A3 + c1 * I) * A3^T in paper
Y = K1 * T;	%  Y = K1 * T in paper

%%%%%%%%%%% Step 3: Estimate A2 with H and Y using the formula in paper
% Option for choosing function f2. The option range: [1 2]
% option 1:	f2(x) = a0 * x
% option 2:	f2(x) = a0 * x ^ n
a0 = 10; n = 5;
f2 = {@(x)a0*x; @(x)a0*sign(x).*abs(x.^n)};	%  f2 in paper
f2_inv = {@(x)(1/a0)*x; @(x)sign(x).*abs((x/a0).^(1/n))};	% The inverse function of f2 in paper
K2 = H' * pinv(H * H' + c2 * eye(size(H, 1))); %  H^T*(H * H^T + c2 * I)^-1 in paper
A2 = f2_inv{option_f2}(Y) * K2;	%  A2 in paper


%%%%%%%%%%% The actual training output Z, MSE and CPU time in paper.
Z = OutputWeight * f2{option_f2}(A2 * H);	%  Z in paper
eta = sum((T - Z) .^ 2) / NumberofTrainingData;	% eta: mean square error in paper.

end_time_train=cputime;
TrainingTime=end_time_train-start_time_train        %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy for Regression.
TrainingAccuracy = eta               %   Calculate training accuracy (MSE) for regression case
clear H;

%%%%%%%%%%% Calculate the output of testing input as well as the testing accuracy.
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;		    %   A * X + (1^T) ** b in paper
H_test = f1{option_f1}(tempH_test);		    %   H = f1(A1*X + (1^T) ** b) in paper
TZ=OutputWeight * f2{option_f2}(A2 * H_test);       %   TZ: the actual output of the testing data. TZ = A3 * f2(A2 * H) in paper
end_time_test=cputime;
TestingTime=end_time_test-start_time_test           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
TestingAccuracy = sum((TV.T - TZ) .^ 2) / NumberofTestingData            %   Calculate testing accuracy (MSE) for regression case

end
