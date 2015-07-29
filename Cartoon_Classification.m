close all ; clear all; 

%sort the imgs into classes based on score.... 3 classes
%Class 1: Family Guy
%Class 2: Futurama
%Class 3: The Simpsons 

%CHANGE THIS TO BE YOUR CURRENT WORKING DIRECTORY!!!! (or else where your
%img folders are stored folder is stored
rootFolder = fullfile('C:\Users\Paulina\Desktop\Desktop Folders\REU 2015\SURF\Data_Image_sets', 'TVSubreddit') ; 

imgset = [ imageSet(fullfile(rootFolder, 'familyguy')), ...
           imageSet(fullfile(rootFolder, 'futurama')), ...
           imageSet(fullfile(rootFolder, 'thesimpsons'))] ;


%and now we use the other SURF code....

{ imgset.Description } % display all labels on one line
[imgset.Count]         % show the corresponding count of images


[trainingSets, validationSets] = partition(imgset, 0.5, 'randomize');

extractor = @BagOfFeaturesExtractor;
bag = bagOfFeatures(trainingSets,'CustomExtractor', extractor);

categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

confMatrix = evaluate(categoryClassifier, trainingSets);

confMatrix = evaluate(categoryClassifier, validationSets);

% Compute average accuracy
mean(diag(confMatrix));


