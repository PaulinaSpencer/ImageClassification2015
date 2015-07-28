close all ; clear all; 
load('img_filenameNOGIF.mat'); load('img_scoreNOGIF.mat') ;
%load img file data

%sort the imgs into classes based on score.... 4 classes
%Class 1: 0
%Class 2: 1-929, 
%Class 3: 930-1859, 
%Class 4: 1860-2789,
%Class 5: 2790-3720
img_score = img_scoreNOGIF ; 
img_filename = img_filenameNOGIF ; 

class1 = find(img_score==0) ; 
class2 = find(img_score<930 & img_score>0) ; 
class3 = find(img_score>= 930 & img_score<1860) ; 
class4 = find(img_score>=1860 & img_score<2790) ; 
class5 = find(img_score>=2790) ; 

%retrieve the filenames for image in each class
class1_imgNAME = img_filename(class1) ; 
class2_imgNAME = img_filename(class2) ; 
class3_imgNAME = img_filename(class3) ; 
class4_imgNAME = img_filename(class4) ; 
class5_imgNAME = img_filename(class5) ; 

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

Class3 = [] ;
for i = 1:length(class3_imgNAME)
Class3 = [Class3, strcat(rootFolder, '\',class3_imgNAME(i))] ;  
end

Class4 = [] ;
for i = 1:length(class4_imgNAME)
Class4 = [Class4, strcat(rootFolder, '\',class4_imgNAME(i))] ;  
end

Class5 = [] ;
for i = 1:length(class5_imgNAME)
Class5 = [Class5, strcat(rootFolder, '\',class5_imgNAME(i))] ;  
end

%creates an image set with each class
imgset = [imageSet(Class1), ...
         % imageSet(Class2),...
          %imageSet(Class3),...
          %imageSet(Class4), ...
          imageSet(Class5) ] ;

%sets the description for each image set to be the class number (this is
%the 'label')
for i = 1:length(imgset) 
    Class_Num = int2str(i) ; 
    imgset(i) = setfield(imgset(i),'Description', strcat('class',Class_Num)) ; 
end

%and now we use the other SURF code....

{ imgset.Description } % display all labels on one line
[imgset.Count]         % show the corresponding count of images



minSetCount = min([imgset.Count]); % determine the smallest amount of images in a category
% 
% % Use partition method to trim the set.
 imgset = partition(imgset, 2, 'randomize');

%30% random img for training data 70% validation data -- now we're using
%50%
[trainingSets, validationSets] = partition(imgset, 0.5, 'randomize');

extractor = @BagOfFeaturesExtractor;
bag = bagOfFeatures(trainingSets,'CustomExtractor', extractor);

categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

confMatrix = evaluate(categoryClassifier, trainingSets);

confMatrix = evaluate(categoryClassifier, validationSets);

% Compute average accuracy
mean(diag(confMatrix));


