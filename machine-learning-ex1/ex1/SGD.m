function [theta, J_history] = SGD(X, y, theta, alpha, num_iters)
	%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
	%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
	%   taking num_iters gradient steps with learning rate alpha

	% Initialize some useful values
	m = length(y); % number of training examples
	J_history = zeros(num_iters, 1);

	mini_J = Inf;
	mini_theta = theta;
	for iter = 1 : num_iters,

		% ====================== YOUR CODE HERE ======================
		% Instructions: Perform a single gradient step on the parameter vector
		%               theta. 
		%
		% Hint: While debugging, it can be useful to print out the values
		%       of the cost function (computeCostMulti) and gradient here.
		%
		
		r = randperm(m)(:,end);
		theta = theta - alpha / m * X(r, :)' * (X(r,: ) * theta - y(r, :));

		% ============================================================

		% Save the cost J in every iteration    
		J_history(iter) = computeCostMulti(X, y, theta);
		
		if(J_history(iter) < mini_J),
			mini_J = J_history(iter);
			mini_theta = theta;
		else
			alpha = alpha * 0.99998;
		end;
		
		if(J_history(iter) < 2043280051),
			disp("mini batch gd break\n");
			break;
		end;
	end;
	J_history(iter) = mini_J;
	theta = mini_theta;
end
