
% To calculate the optimized alpha by dichotomy( binary division method )
function [optimized_alpha, iterNum] = findAlpha(alpha,numIter, X, y, num_iters)
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
end