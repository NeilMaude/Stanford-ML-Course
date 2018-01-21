function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% loop over the centroids
for k = 1:size(centroids,1)     % K centroids
  
  % find the mean of the points assigned to this centroid
  point_count = 0;
  
  for i = 1:m   % we have m points = size(X, 1)
    if idx(i) == k   
      % point is assigned to this centroid
      centroids(k,:) = centroids(k,:) .+ X(i,:);     % accumulate the values
      point_count = point_count + 1;
    end
  end
  if point_count > 0
    % have some points
    centroids(k,:) = centroids(k,:) ./ point_count;  % average out the values
  end
  
end
   
    






% =============================================================


end

