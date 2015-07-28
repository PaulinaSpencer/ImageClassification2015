close all ; clear all; 
%Need Quartiles_data.m function and the BagOfFeaturesExtractor.m function
%to run this script. 
% CHANGE WORKING DIRECTORY IN THIS CODE ONLY

load('img_filenameNOGIF.mat'); load('img_scoreNOGIF.mat') ;
%load img file data

%sort the imgs into classes based quartiles.  Our assumption is anything
%below the 3rd quartile is 'bad' and anything about the 3rd quartile is
%'good' 


img_score = img_scoreNOGIF ; 
img_filename = img_filenameNOGIF ; 

[Q1, Q2, Q3] = Quartiles_data(img_score) ;

Q3
%class 1: below Q3
%class 2: above Q3
class1 = find(img_score<Q3) ; 
class2 = find(img_score>Q3) ; 

%retrieve the filenames for image in each class
class1_imgNAME = img_filename(class1) ; 
class2_imgNAME = img_filename(class2) ; 


%CHANGE THIS TO BE YOUR CURRENT WORKING DIRECTORY!!!! (or else where your
%img_file folder is stored
rootFolder = fullfile('C:\Users\Paulina\Desktop\Desktop Folders\REU 2015\SURF', 'img_files');


%gets each individual image location for each class
Class1 = [] ;
for i = 1:length(class1_imgNAME)
Class1 = [Class1, strcat(rootFolder, '\',class1_imgNAME(i))] ;  
end

Class2 = [] ;
for i = 1:length(class2_imgNAME)
Class2 = [Class2, strcat(rootFolder, '\',class2_imgNAME(i))] ;  
end


%creates an image set with each class
imgset = [imageSet(Class1), ...
          imageSet(Class2)] ; 


%sets the description for each image set to be the class number (this is
%the 'label')
for i = 1:length(imgset) 
    Class_Num = int2str(i) ; 
    imgset(i) = setfield(imgset(i),'Description', strcat('class',Class_Num)) ; 
end

%and now we use the other SURF code....

{ imgset.Description } % display all labels on one line
[imgset.Count]         % show the corresponding count of images



minSetCount = min([imgset.Count]); % determine the smallest amount of images in a category (if we want all imgsets to be the same size)
% 
% % Use partition method to trim the set. (set when we want our code to run
% fast -- otherwise comment out.  
 imgset = partition(imgset, 10, 'randomize');

%30% random img for training data 70% validation data -- now we're using
%50%
[trainingSets, validationSets] = partition(imgset, 0.5, 'randomize');

extractor = @BagOfFeaturesExtractor; %set incase we decide to use .gifs.
bag = bagOfFeatures(trainingSets,'CustomExtractor', extractor);

categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

confMatrix = evaluate(categoryClassifier, trainingSets);

confMatrix = evaluate(categoryClassifier, validationSets);

% Compute average accuracy
mean(diag(confMatrix));


