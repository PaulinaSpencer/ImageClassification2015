%trainImageCategoryClassifier Train bag of features based image category classifier
%   Use of this function requires that you have the Statistics and Machine Learning Toolbox.
%
%   classifier = trainImageCategoryClassifier(imgSets, bag) creates an
%   image category classifier. imgSets must be an array of imageSet objects
%   and bag is a bagOfFeatures object. Each element of the imgSets defines
%   an image category. A linear SVM classifier using error correcting
%   output codes (ECOC) is then trained to classify amongst each of the
%   image categories.
%
%   [...] = trainImageCategoryClassifier(..., Name, Value) specifies
%   additional name-value pairs described below:
%
%   'Verbose'        Set to true to display training progress information. 
%
%                    Default: true
%
%   'LearnerOptions' A learner template constructed by calling templateSVM
%                    function in the Statistics and Machine Learning Toolbox.
%                    This lets you change characteristics of the SVM classifier.
%                    See help for templateSVM function for more details.
%                    For example, to adjust the regularization parameter 
%                    and to set a custom kernel function, use 
%                    the following syntax:
%
%                    opts = templateSVM('BoxConstraint', 1.1, ...
%                                       'KernelFunction', 'gaussian');
%                    classifier = trainImageCategoryClassifier(imgSets,...
%                                       bag, 'LearnerOptions', opts);
%
%                    Default: defaults used by templateSVM function
%
%   Notes:
%   ------   
%   - trainImageCategoryClassifier supports parallel computing using
%     multiple MATLAB workers. Enable parallel computing using the 
%     <a href="matlab:preferences('Computer Vision System Toolbox')">preferences dialog</a>.
%
%   Example
%   -------
%   % Load two image categories
%   setDir  = fullfile(toolboxdir('vision'),'visiondata','imageSets');
%   imgSets = imageSet(setDir, 'recursive');
%
%   % Separate the two sets into training and test data. Pick 30% of images
%   % from each set for the training data and the remainder (70%) for the 
%   % test data.
%   [trainingSets, testSets] = partition(imgSets, 0.3, 'randomize'); 
%
%   % Create bag of visual words
%   bag = bagOfFeatures(trainingSets);
%
%   % Train a classifier
%   categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);
% 
%   % Evaluate the classifier on test images and display the confusion matrix
%   confMatrix = evaluate(categoryClassifier, testSets)
%
%   % Average accuracy
%   mean(diag(confMatrix))
%
%   % You can apply the newly trained classifier to categorize new images
%   img = imread(fullfile(setDir, 'cups', 'bigMug.jpg'));
%   [labelIdx, score] = predict(categoryClassifier, img);
%   % Display the string label
%   categoryClassifier.Labels(labelIdx)
%
%   See also bagOfFeatures, imageCategoryClassifier, imageSet, fitcecoc, 
%      templateSVM

% Copyright 2014 MathWorks, Inc.

% References:
%    Gabriella Csurka, Christopher R. Dance, Lixin Fan, Jutta Willamowski,
%    Cedric Bray "Visual Categorization with Bag of Keypoints", 
%    Workshop on Statistical Learning in Computer Vision, ECCV

%NOTHING TO CHANGE HEERE 
function classifier = trainImageCategoryClassifier_EDIT(imgSet, bag, varargin)

vision.internal.requiresStatisticsToolbox(mfilename);

classifier = imageCategoryClassifier_EDIT.create(imgSet, bag, varargin{:});

