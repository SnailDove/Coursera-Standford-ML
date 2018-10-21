function [error_train, error_val] = RandSampleLearningCurve(X, y, Xval, yval, lambda);

% learning 
m = size(X, 1);
error_train = zeros(m, 1);
error_val = zeros(m, 1);

for i = 1 : m,
	error_train(i) = 0;
	error_val(i) = 0;
	for j = 1 : 50
		random_ith = [randperm(12)](1 : i);
		Xtrain_selected = X(random_ith, :);
		ytrain_selected = y(random_ith, :);
		Xval_selected = Xval(random_ith, :);
		yval_selected = yval(random_ith, :);
		
		theta = trainLinearReg(Xtrain_selected, ytrain_selected, lambda);
		error_train(i) =  error_train(i) + linearRegCostFunction(Xtrain_selected, ytrain_selected, theta, 0); 
		error_val(i) = error_val(i) + linearRegCostFunction(Xval_selected, yval_selected, theta, 0);		
	end
	error_train(i) = error_train(i) / 50;
	error_val(i) = error_val(i) / 50;
end

end