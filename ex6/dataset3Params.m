function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_values = [0.01, 0,03, 0.1, 0.3, 1, 3, 10, 30];
sigma_values = [0.01, 0,03, 0.1, 0.3, 1, 3, 10, 30];
small_error = Inf;


for i=1:length(C_values)
    for j=1:length(sigma_values)
        
        C_test = C_values(i);
        sigma_test = sigma_values(j);
        model = svmTrain(X, y, C_test, @(x1, x2)gaussianKernel(x1, x2, sigma_test));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        
        if (error < small_error)
            C = C_test;
            sigma = sigma_test;
            small_error = error;
        end
        
    end
    
end



end

% =========================================================================


