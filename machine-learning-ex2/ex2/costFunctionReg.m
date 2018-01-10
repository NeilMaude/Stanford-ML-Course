function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% hypothesis theta is sigmoid theta-transpose * x
% seems to be swapped in our case to X * theta
h = sigmoid(X * theta);

y_ones = log(h) .* -1 .* y;
y_zeros = log(1 .- h) .* -1 .* (1 .- y);
J = (sum(y_ones) + sum(y_zeros)) / m;
% reg term to add on to the cost function
J = J + (sum(theta(2:size(theta)) .* theta(2:size(theta)))  * (lambda / (2 * m)));   


% gradients 
grad = ((sigmoid(X * theta) - y)' * X)' ./ m ;
grad = grad .+ (theta .* (lambda / m));         % reg term 

% grad(1) = theta(0) term, different from the rest
grad(1) = sum(sigmoid(X * theta) - y) * X(1) / m;



% =============================================================

end
