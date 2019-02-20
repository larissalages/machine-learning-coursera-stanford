function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n_features = size(X,2);
temp = zeros(n_features,0);

for iter = 1:num_iters


    hip = X*theta;
    
    for j =1:n_features
        theta(j) = theta(j) -(alpha/m)*(hip-y).'*X(:,j);

    end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    disp(computeCostMulti(X, y, theta))

end

end
