%% Synthetic Data Generation (SDG) by Group Method of Data Handling (GMDH)
% Developed by Seyed Muhammad Hossein Mousavi - July 2023
% The "Group Method of Data Handling" (GMDH) is an algorithmic approach used for modeling and pattern
% recognition in complex systems. It was proposed by the Ukrainian mathematician
% Aleksandr G. Ivakhnenko in 1968. The primary goal of GMDH is to find functional 
% relationships and patterns within a given dataset without making any prior assumptions about 
% the underlying model structure. I used it for SDG. GMDH could be used for
% time series forecasting and more applications. GMDH part is developed by
% Yarpiz. I just modify it for the SDG application. 
%-------------------------------------------------------------------------------
clc;
clear;
close all;

%% Input
load fisheriris.mat;
Input=reshape(meas,1,[]); % Preprocessing - convert matrix to vector
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target'; % Original labels
classes= 3;
% More value of "Variation" means more synthetic data.
Variation = 6;
for i=1:Variation
Delays = [1 2 3 4];
[Inputs, Targets] = CreateTimeSeriesData(Input,Delays);
nData = size(Inputs,2);
Perm = randperm(nData);
% Create GMDH Network
params.MaxLayerNeurons = 30;   % Maximum Number of Neurons in a Layer
params.MaxLayers = 9;          % Maximum Number of Layers
params.alpha = 0.3;            % Selection Pressure
params.pTrain = 0.9;           % Train Ratio
% Train GMDH
gmdh = GMDH(params, Inputs, Targets);
%  GMDH Model on Train 
Outputs{i} = ApplyGMDH(gmdh, Inputs);
end
% Converting cell to matrix
for i = 1 : Variation
Generated(:,i)=Outputs{i};
end
% Converting matrix to cell
P = size(Input); P = P (1,2);
S = size(Outputs{i});
SO = size (meas);
SF = SO (1,2);
SO = SO (1,1);
SS = S (1,2); 
R = SS*SO/P;
for i = 1 : Variation
Generated1{i}=reshape(Generated(:,i),[R,SF]);
end
% Converting cell to matrix (the last time)
Synthetic = cell2mat(Generated1');
% K-means clustering to get the labels
[idx,C] = kmeans(Synthetic,classes);

%% Plot data and classes
Feature1=1;
Feature2=3;
f1=meas(:,Feature1); % feature1
f2=meas(:,Feature2); % feature 2
ff1=Synthetic(:,Feature1); % feature1
ff2=Synthetic(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
plot(meas, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,2)
plot(Synthetic, 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,3)
gscatter(f1,f2,Target,'rkgb','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,4)
gscatter(ff1,ff2,idx,'rkgb','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;

%% Train and Test
% Training Synthetic dataset by SVM
Mdlsvm  = fitcecoc(Synthetic,idx); CVMdlsvm = crossval(Mdlsvm); 
SVMError = kfoldLoss(CVMdlsvm); SVMAccAugTrain = (1 - SVMError)*100;
% Predict new samples (the whole original dataset)
[label5,score5,cost5] = predict(Mdlsvm,meas);
sizlbl=R;
% Test error and accuracy calculations
a=0;b=0;c=0;
for i=1:R
if label5(i)== 1
a=a+1;
elseif label5(i)==2
b=b+1;
else
label5(i)==3
c=c+1;
end;end;
erra=abs(a-50);errb=abs(b-50);errc=abs(c-50);
err=erra+errb+errc;TestErr=err*100/R;SVMAccAugTest=100-TestErr; % Test Accuracy
% Result SVM
AugResSVM = [' Synthetic Train SVM "',num2str(SVMAccAugTrain),'" Synthetic Test SVM"', num2str(SVMAccAugTest),'"'];
disp(AugResSVM);
