close all; close all; clc

% Reload the image from the previous exercise and run K-Means on it
% For this to work, you need to complete the K-Means assignment first
A = double(imread('compressed.png'));

% If imread does not work for you, you can try instead
%   load ('bird_small.mat');

A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);

t1=clock();% start to count run time

K = 32; 
max_iters = 32;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

minutes = etime(clock(), t1) / 60 ; % end up with counting time
sprintf("runKmeans cost time %f minutes\n", minutes); % print the running time

% Find closest cluster members
idx = findClosestCentroids(X, centroids);

% Essentially, now we have represented the image X as in terms of the
% indices in idx. 

% We can now recover the image from the indices (idx) by mapping each pixel
% (specified by its index in idx) to the centroid value
X_cpmpressed = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_cpmpressed = reshape(X_cpmpressed, img_size(1), img_size(2), 3);

% Display the original image 
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_cpmpressed)
title(sprintf('Kmeans Compressed, with %d colors.', K));


fprintf('Program paused. Press enter to continue.\n');
pause;


fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this might take a minute or two ...)\n\n']);

%% =========== Part 2: PCA on Face Data: Eigenfaces  ===================
%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature

X = reshape(A, img_size(1) * img_size(2), 3);

[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

K = 100;
Z = projectData(X_norm, U, K);
