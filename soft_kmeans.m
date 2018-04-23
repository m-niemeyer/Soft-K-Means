% This script illustrates how the soft-k-means algorithm can find cluster
% centers for appropriate data. 
% 100 draws from a Gaussian mixture distribution with means (0 0) and (7 7)
% and the covanriance matrix of [4 1; 1 4]. The soft k-means-algorithm is
% able to discover these centers fairly well which can be seen in the
% figure as well as the final cluster centers afer 10 iterations.

mu = [0 0; 7 7];
sigma = [4 1; 1 4];

gm = gmdistribution(mu, sigma);
data = random(gm, 120);

cen = [-2 -2; -1 -1];

y = calc_respon(data, cen);
centers_1 = assign(data, cen, y);
cen = centers_1;

for i=1:10
    y = calc_respon(data, cen);
    cen = assign(data, cen, y);
end
    

figure(1);cla;
hold on
plot(data(:, 1), data(:, 2), '*')
plot(cen(:, 1), cen(:, 2), '+')
display('The cluster centers after 10 iterations are:');
display(cen);

function res=calc_respon(data, cen)
% Calculates the 'soft' responsibilities for each data point with regard to
% each cluster.
[k_cluster, ~] = size(cen);
[num_points, ~] = size(data);
b = -1; % The beta value

res = zeros(k_cluster, num_points);
for i=1:k_cluster
    res(i, :) = exp(b*vecnorm((data-cen(i, :))'));
end
res = res./sum(res); % Column-wise normalisation
end

function center=assign(data, cen, responsibility)
% Assigns the new cluster centers in the 'soft' manner. The new centers are
% not just averages of the assigned data points, but weighted averages that
% take into account how far away a data point is.
[k_cluster, dim] = size(cen);
[num_points, ~] = size(data);
center = zeros(k_cluster, dim);

total_res = sum(responsibility');

for i=1:k_cluster
    new_cen = zeros(1, dim);
    for j=1:num_points
        new_cen = new_cen + data(j, :).*responsibility(i, j);
    end
    center(i, :) = new_cen./total_res(1, i);
end
end