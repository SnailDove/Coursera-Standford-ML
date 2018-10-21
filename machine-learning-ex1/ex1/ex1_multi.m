%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = [0.01;2];
theta = zeros(3, 1);
num_iters = 50;

% Choose some alpha value
% To calculate the optimized alpha by dichotomy( binary division method )
% this for loop can be a function : optimized_alpha = find_optimized_alpha(alpha, 1, X, y, num_iters);
for i = 1 : 20,
	min_alpha = alpha(1,1);
	max_alpha = alpha(2,1);
	theta = zeros(3, 1);
	[theta, min_J_history] = gradientDescentMulti(X, y, theta, min_alpha, num_iters);
	theta = zeros(3, 1);
	[theta, max_J_history] = gradientDescentMulti(X, y, theta, max_alpha, num_iters);
	
	compare_start = ceil((num_iters*0.50));
	threshold = ceil((num_iters*0.50)) * 0.8;
	compareFlag =  sum(max_J_history(compare_start:num_iters,:) > min_J_history(compare_start:num_iters,:)) >= threshold;	
	
	%debug code : 
	%disp(sprintf("compare %d \n", compareFlag));
	%save max.dat max_J_history -ascii;
	%save min.dat min_J_history -ascii;
	
	if 1 == compareFlag, 
		alpha(2,1) = (min_alpha + max_alpha) / 2;
		optimized_alpha = min_alpha;
	else
		alpha(1,1) = (min_alpha + max_alpha) / 2;
		optimized_alpha = max_alpha;
	end
end

% debug code : 
% disp(sprintf("optimized alpha : %f \n", optimized_alpha));
% Init Theta and Run Gradient Descent 

% Plot the convergence graph
figure;

theta = zeros(3, 1);

[theta, J_history] = miniBatchGradientDescent(X, y, theta, optimized_alpha, 1000);
fprintf('cost of mini batch GD : % f\n', J_history(end));
save miniBatchGradientDescent.dat J_history -ascii;
plot(1:numel(J_history), J_history, '-y', 'LineWidth', 2);
hold on;

theta = zeros(3, 1);
[theta, J_history] = SGD(X, y, theta, optimized_alpha, 1000);
fprintf('SGD mini cost: % f\n', J_history(end));
save SGD.dat J_history -ascii;
plot(1:numel(J_history), J_history, '-r', 'LineWidth', 2);
hold on;

theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, optimized_alpha, 25);
save GradientDescent.dat J_history -ascii;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);

fprintf('cost of GD: % f\n', J_history(end));

legend('mini Batch gradient descent', 'Stochastic gradient descent','Batch gradient descent');
xlabel('Number of iterations');
ylabel('Cost J');


% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.

price = [1, (1650 - mu(1,1)) / sigma(1,1), (3 - mu(1,2)) / sigma(1,2)] * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = [1, 1650, 3] * theta;


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);


