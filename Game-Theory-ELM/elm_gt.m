% This code implements the game theory based ELM algorithm, which highly reduces the regulation time.
% The implementations are developed based on the original ELM which was proposed by MR QIN-YU ZHU AND DR GUANG-BIN HUANG in NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE.
% The code is written by Ruofan Kong.

function [TotalTime, TestingAccuracy] = elm_gt(TrainingDataFile, TestingDataFile, Elm_Type, NumberofHiddenNeurons, option, c, seed, p_rate)

% Usage: [TotalTime, TestingAccuracy] = elm_gt(TrainingDataFile, TestingDataFile, Elm_Type, NumberofHiddenNeurons, option, c, seed, p_rate)
%
% Input:
% TrainingDataFile     - Filename of training data set
% TestingDataFile      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% option	        - Option of activation function:
%                           1 for Sigmoidal function
%                           2 for Sine function
%                           3 for Hardlim function
%                           4 for Triangular basis function
%                           5 for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
% seed			- Control the random number generator 
% c			- c is a vector containing all possible regulation values.
% p_rate		- p_rate means the ratio between maximum eigenvalue of H * H' and the minimum one that we want to maintain. 
%			  For example, if p_rate = 100, then we keep the eigenvalues in the range [max(eig(H * H')) / 100, max(eig(H * H'))], and discard all other eigenvalues. 
%
% Output: 
% TotalTime          	- Time (seconds) spent on both training ELM and adjusting regulation factor c.	    
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification.
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class

%%%%%%%%%%% Begin timing
start_time = cputime;

%%%%%%%%%%% Compute the required information for regulation in advance to avoid re-using the information, and initialize all required parameters.
% The required information are: H, H', Y, Y' in paper.
[H, H_test, T, V] = elm_gt_predefine(TrainingDataFile, TestingDataFile, Elm_Type, NumberofHiddenNeurons, option, seed);
NumberofTrainingData = size(T,2);	% Number of Training data instances
NumberofTestingData = size(V,2);	% Number of Testing data instances
c_n = size(c, 2);	% c_n is the total trial times for regulating c. For example, if we tried 30 different c, then c_n = 30
TestingAccuracy = zeros(1, c_n);	% The TestingAccuracy for each regulator c.

threshold = 170;	% The eigenvalue threshold, meaning that eigen values less than threshold will be discarded. 

%%%%%%%%%%% Compute the testing accuracy for each regulator c.
% Diagonal matrix lambda contains all eigenvalues of H * H^T, and matrix u contains all respective eigenvectors.
[u lambda] = eig(H * H');	% u is the eigen vector of H * H ^T, and lambda is the corresponding eigen values
lambda = diag(lambda);		% Convert lambda to a vector
threshold = max(lambda) / p_rate; 	% Threshold of principal components of H * H'.

% Delete all eigenvalues with their eigenvectors which are useless.
index = find(lambda < threshold);	
lambda(index) = [];	
u(:,index) = []; 

% Initialize the required parameters: u_size, the dimension of the eigenvector and E_i in paper. 
u_size = size(u, 1);	
temp_sum = zeros(u_size, c_n * u_size);	% Initialize E_i in paper

% Compute ||E_0 - sum((1/ c - 1/(lambda_j + c)) * E_i) - Y'|| in paper.
E_0 = T * H' * H_test;	% E_0  = Y * H^T * H' (not including c) in paper.
res = kron(1 ./ c, E_0);	% res takes into account c, and extend the matrix with row size(E_0, 1) and column c_n * size(E_0, 2). res = 1/c * E_0 for each c.
for j = 1 : length(lambda)	% Compute sum((1/ c - 1/(lambda_j + c)) * E_i). 
	temp_sum += kron(1 ./ c - 1 ./ (lambda(j) + c), u(:,j) * u(:,j)'); 
end
H_test_exd = kron(eye(c_n, c_n), H_test);	% Extend H_test to be compatible with temp_sum.
res -= T * H' * temp_sum * H_test_exd;	% Compute 1/c * Y * H^T * H' - Y * H^T * sum((1/c -1/(lambda_i+c)) * u_i * u_i^T) for all c in paper
end_time = cputime;		% End timing
TotalTime = end_time - start_time;

%%%%%%%%%%% Compute the testing accuracy
res_test = kron(ones(1, c_n), V);	% res_test extends the output of testing dataset to be compatible with the actual output res
index = 1;	% index of accuracy matrix
if Elm_Type == 0
	for i = 1 : NumberofTestingData : size(res_test, 2)
    		TestingAccuracy(1, index++) = sum((res(:,i:i+NumberofTestingData-1) - res_test(:,i:i+NumberofTestingData-1)) .^ 2) / NumberofTestingData;                %   Calculate testing accuracy (RMSE) for regression case
	end
else	
	for i = 1 : NumberofTestingData : size(res_test, 2)	%   Calculate testing accuracy for classification case		
        	[temp, label_index_expected] = max(res_test(:, i : i + NumberofTestingData - 1));
		[temp, label_index_actual] = max(res(:, i : i + NumberofTestingData - 1));
		MissClassificationRate_Testing = length(find(label_index_actual ~= label_index_expected));
   		TestingAccuracy(1,index++) = 1 - MissClassificationRate_Testing/NumberofTestingData;
    	end 	
end	

end
