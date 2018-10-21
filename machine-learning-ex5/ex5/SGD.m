function [error_train, error_val] = RandSammplelearningCurve(X, y, Xval, yval, lambda);

% learning 


for i = 1 : m,
	error_train(i) = 0;
	error_val(i) = 0;
	for j = 1 : 50
		random_ith = randperm(12)(1:i);
		X = X(random_ith, :);
		y = y(random_ith, :);
		
		% Initialize Theta
		initial_theta = zeros(size(X, 2), 1); 
		% Create "short hand" for the cost function to be minimized
		costFunction = @(t) linearRegCostFunction(X, y, t, lambda);

		% Now, costFunction is a function that takes in only one argument
		options = optimset('MaxIter', 100, 'GradObj', 'on');

		% Minimize using fmincg
		theta = fmincg(costFunction, initial_theta, options);

		theta = trainLinearReg(X, y, lambda);
		error_train(i) =  error_train(i) + linearRegCostFunction(X, y, theta, 0); 
		error_val(i) = error_val(i) + linearRegCostFunction(Xval, yval, theta, 0);		
	end
	error_train(i) = error_train / 50;
	error_val(i) = error_val / 50;
end








end