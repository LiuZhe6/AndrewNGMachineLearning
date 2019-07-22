function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    %for循环（效率低，而且精度不符合，不推荐）
    % n = length(X(1));
    % for j = 1 : n+1,
    %     sum = 0;
    %     for i = 1 : m,
    %         sum += (theta(1) * X(i,1) + theta(2) * X(i,2) - y(i)) * X(i,j);
    %     end;
    %     theta(j) -= alpha / m * sum;
    % end


    %向量化 (推荐)
    theta -= alpha / m * X' * (X * theta - y);




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
