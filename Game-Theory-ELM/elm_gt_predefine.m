function [H, H_test, T, V] = elm_gt_predefine(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, option, seed)

% Usage: [H, H_test, T, V] = elm_gt_predefine(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, option, seed)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% option	        - Option of activation function:
%                           1 for Sigmoidal function
%                           2 for Sine function
%                           3 for Hardlim function
%                           4 for Triangular basis function
%                           5 for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
% seed			- Control the random number generator 
%
% Output: 
% H          		- H matrix in paper.
% H_test       		- H' matrix in paper.
% T			- The output of training dataset.
% V			- The output of testing dataset.
%
% The code computes the required information (H, H', Y and Y' in paper) to avoid re-using information.
% The code is implemented based on the original ELM proposed by MR QIN-YU ZHU AND DR GUANG-BIN HUANG in NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE.
% The code is updated by Ruofan Kong.

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
train_data=load(TrainingData_File);
T=train_data(:,1)';				    % Y in paper
P=train_data(:,2:size(train_data,2))';		    % X in paper
clear train_data;                                   % Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=load(TestingData_File);		   
TV.T=test_data(:,1)';                               % Y' in paper					    % Y' in paper
TV.P=test_data(:,2:size(test_data,2))';		    % X' in paper
clear test_data;                                    % Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData) 
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class; % Get the total number of class
       
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
    V = TV.T;

end                                                 %   end if of Elm_Type

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
rand("seed", seed);				    %   Set the first element of the random number
InputWeight = rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons = rand(NumberofHiddenNeurons,1);
tempH = InputWeight * P;
clear P;                                            %   Release input of training data 
ind = ones(1,NumberofTrainingData);
BiasMatrix = BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH += BiasMatrix;
% Variable option is for choosing activation function f. The option range: [1 5].
% option 1:	sigmoid function
% option 2:	sin function
% option 3:	hardlim function
% option 4:	tribas function
% option 5:	radbas function
f = {@(x)1 ./ (1 + exp(-x)); @(x)sin(x); @(x)double(hardlim(x)); @(x)tribas(x); @(x)radbas(x)};
H = f{option}(tempH);	% Compute the H matrix in paper.
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Compute H_test which is H' in paper.
tempH_test = InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind = ones(1,NumberofTestingData);
BiasMatrix = BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test += BiasMatrix;
H_test = f{option}(tempH_test);	% H' in paper

end
