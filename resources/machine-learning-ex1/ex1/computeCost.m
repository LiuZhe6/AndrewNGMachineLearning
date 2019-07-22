function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%使用for循环(这样不好，效率太低)
% for i=1:m
%    hx = X(i,:) * theta;
%    J += (hx - y(i)) ^ 2;
% end;
% J = J / (2*m);


%向量化方法(推荐使用)
hx = X * theta -y;
J = sum(hx .^ 2)/(2*m);

% =========================================================================

end
