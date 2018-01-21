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
sigma = 0.1;      % found 0.1 as optimal, was 0.3 default;

testing = false;    % set true to run the whole process

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

if testing == true
  values = [0.01 0.03 0.1 0.3 1 3 10 30];
  error_min = inf;

  total_runs = size(values)(2) ^ 2;
  run_count = 1;
  
  fprintf('Searching for optimal C and sigma..\n\n');
  for C = values
    for sigma = values
      fprintf('Run %i of %i.\n', run_count, total_runs);
      model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
      err   = mean(double(svmPredict(model, Xval) ~= yval));
      if( err <= error_min )
        C_final     = C;
        sigma_final = sigma;
        error_min   = err;
        fprintf('New min found: C, sigma = %f, %f with error = %f\n\n', C_final, sigma_final, error_min)
      end
      run_count = run_count + 1;
    end
  end
  C     = C_final;
  sigma = sigma_final;

  fprintf('Best values: C, sigma = [%f %f] with prediction error = %f\n', C, sigma, error_min);
else
  fprintf('Not running any tests this time around...');
end




% =========================================================================

end
