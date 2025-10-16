%% PART A
% Load input data and Scatter Plot of Training Data 
% Load data from 'cw1a.mat'
load('cw1a.mat');  % Assumes 'cw1a.mat' contains variables 'x' and 'y'

% Generate scatter plot
figure;
scatter(x, y, 15, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c');
% Axis labels and titles
xlabel('Input x', 'FontSize', 12);
ylabel('Output y', 'FontSize', 12);
title('Training Data', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);

% Fitting Gaussian Process Function 
meanfunc = @meanZero;           
covfunc = @covSEiso;             % Squared exponential isotropic covariance
likfunc = @likGauss;             % Gaussian likelihood

hyp.mean = [];                    
hyp.cov = [-1; 0];                % Initial log hyperparameters for covSEiso
hyp.lik = 0;                      % Initial log noise standard deviation

% Inference method for exact Gaussian likelihood
inf = @infGaussLik;

% Optimising hyperparameters 
hyp2 = minimize(hyp, @gp, -100, inf, meanfunc, covfunc, likfunc, x, y);
hyp2.cov
hyp2.lik

% Test inputs (xs)
xs = linspace(min(x), max(x), 500)';  

% Predictions using optimised hyperparameters
[mu, s2] = gp(hyp2, inf, meanfunc, covfunc, likfunc, x, y, xs);
[nlml, dnlmlz] = gp(hyp2, inf, meanfunc, covfunc, likfunc, x, y);
nlml

% 95% confidence intervals
upper = mu + 2*sqrt(s2);
lower = mu - 2*sqrt(s2);

figure;
hold on;
% Plot training data
scatter(x, y, 15, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c');
% Plot predictive mean
plot(xs, mu, 'b-', 'LineWidth', 1.5);
% Fill the confidence interval
fill([xs; flipud(xs)], [upper; flipud(lower)], [0.65 0.65 0.65], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold off;
xlabel('Input x');
ylabel('Output y');
title('GP Regression with 95% Predictive Error Bars');
legend('Training Data', 'Predictive Mean', '95% Confidence Interval');

% Heat Maps (helps understand behaviour of kernel/ covariance function)
% Input points
input = linspace(-5, 5, 200)'; 

% Define a set of length-scale parameters to visualise
length_scales = [0.5, 1, 2, 5];  
sigma_f = 1;  % signal standard deviation

% Plotting covariance heatmaps
figure;
for i = 1:length(length_scales)
    ell = length_scales(i);
    
    % Covariance function parameters
    covfunc = @covSEiso;
    hyp.cov = [log(ell); log(sigma_f)]; 
    
    % Covariance matrix with jitter
    K = feval(covfunc, hyp.cov, input) + (1e-6 * eye(length(input))); 
    
    % Plot covariance matrix as a heatmap
    subplot(2, 2, i);  % Arrange plots in a 2x2 grid
    imagesc(input, input, K);  % Display the covariance matrix as an image
    axis xy;  % Correct the y-axis orientation
    axis square;
    colormap(jet);
    colorbar;  % color scale
    xlabel('Input x', 'FontSize', 10, 'FontWeight', 'bold');  
    ylabel('Input x', 'FontSize', 10, 'FontWeight', 'bold');  
    title(['l = ' num2str(ell)], 'FontSize', 12, 'FontWeight', 'bold');
end

% Overall title for the figure
sgtitle('Squared Exponential Covariance Heatmaps', 'FontSize', 16, 'FontWeight', 'bold');

%% PART B
% Finding hyperparameter optima via random restarts 
meanfunc = @meanZero;            
covfunc = @covSEiso;             % Squared Exponential Isotropic covariance
likfunc = @likGauss;             % Gaussian likelihood
inf = @infGaussLik;

% Number of random restarts
num_restarts = 10;

% Preallocate storage for optimised hyperparameters and NLMLs
hyp_opts = cell(num_restarts,1);
nlmls = zeros(num_restarts,1);            
log_hyperparams = zeros(num_restarts,3);  % [log_ell, log_sigma_f, log_sigma_n]

% Perform Multiple Random restarts
for i = 1:num_restarts
    % Randomly initialise hyperparameters within specified ranges
    initial_length_scale = log(0.1 + (2 - 0.1) * rand());    % log(ell) between log(0.1) and log(2)
    initial_signal_std = log(0.5 + (2 - 0.5) * rand());      % log(sigma_f) between log(0.5) and log(2)
    initial_noise_std = log(0.05 + (1 - 0.05) * rand());     % log(sigma_n) between log(0.05) and log(1)
    
    % Set initial hyperparameters
    initial_hyp.mean = [];                              
    initial_hyp.cov = [initial_length_scale; initial_signal_std]; % [log(ell); log(sigma_f)]
    initial_hyp.lik = initial_noise_std; % log(sigma_n)
    
    % Optimise hyperparameters
    hyp_opts{i} = minimize(initial_hyp, @gp, -100, inf, meanfunc, covfunc, likfunc, x, y);
    
    % Compute NLML for the optimised hyperparameters
    [nlmls(i), ~] = gp(hyp_opts{i}, inf, meanfunc, covfunc, likfunc, x, y);
    
    % Store the log hyperparameters
    log_hyperparams(i,1) = hyp_opts{i}.cov(1);   % log(ell)
    log_hyperparams(i,2) = hyp_opts{i}.cov(2);   % log(sigma_f)
    log_hyperparams(i,3) = hyp_opts{i}.lik;      % log(sigma_n)
end

% Create a table to display the results
Restart = (1:num_restarts)';
Log_Ell = log_hyperparams(:,1);
Log_Sigma_f = log_hyperparams(:,2);
Log_Sigma_n = log_hyperparams(:,3);
NLML = nlmls;

% Convert log hyperparameters to natural domain
Ell = exp(Log_Ell);
Sigma_f = exp(Log_Sigma_f);
Sigma_n = exp(Log_Sigma_n);

% Combine Data into a Table 
Results_Table = table(Restart, Log_Ell, Log_Sigma_f, Log_Sigma_n, Ell, Sigma_f, Sigma_n, NLML, ...
    'VariableNames', {'Restart', 'Log_Ell', 'Log_Sigma_f', 'Log_Sigma_n', 'Ell', 'Sigma_f', 'Sigma_n', 'NLML'});

% Display the table
disp(Results_Table);

% Best hyperparameters (lowest NLML)
[best_nlml, best_idx] = min(NLML);
best_hyp = hyp_opts{best_idx};

fprintf('\nBest NLML: %.4f from Restart #%d\n', best_nlml, best_idx);
fprintf('Optimized Hyperparameters (Natural Domain):\n');
fprintf('  Length-scale (ell): %.5f\n', exp(best_hyp.cov(1)));
fprintf('  Signal Std Dev (sigma_f): %.5f\n', exp(best_hyp.cov(2)));
fprintf('  Noise Std Dev (sigma_n): %.5f\n', exp(best_hyp.lik));

%% PART C
meanfunc = @meanZero;            
covfunc = @covPeriodic;       % Periodic covariance function
likfunc = @likGauss;          % Gaussian likelihood

hyp.mean = [];               
hyp.cov = [-1; 0; log(1)];
hyp.lik = 0;                 
inf = @infGaussLik;

% Optimise hyperparameters
hyp2 = minimize(hyp, @gp, -100, inf, meanfunc, covfunc, likfunc, x, y);
hyp2.cov
hyp2.lik

% Test inputs (xs)
xs = linspace(min(x), max(x), 500)';  
% Perform prediction using the optimised hyperparameters
[mu, s2] = gp(hyp2, inf, meanfunc, covfunc, likfunc, x, y, xs);

% Calculate 95% confidence intervals
upper = mu + 2*sqrt(s2);
lower = mu - 2*sqrt(s2);

% Plot the results
figure;
hold on;
% Plot training data
scatter(x, y, 15, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c');
% Plot predictive mean
plot(xs, mu, 'b-', 'LineWidth', 1.5);
% Fill the confidence interval
fill([xs; flipud(xs)], [upper; flipud(lower)], [0.65 0.65 0.65], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold off;
xlabel('Input x');
ylabel('Output y');
title('GP Regression with 95% Predictive Error Bars');
legend('Training Data', 'Predictive Mean', '95% Confidence Interval');

%% PART D
% Define input points
input_1 = linspace(-5, 5, 200)';  
n = size(input_1,1);

meanfunc = @meanZero;            
covfunc = {@covProd, {@covPeriodic, @covSEiso}};
likfunc = @likGauss; % Gaussian likelihood
hyp.mean = [];                  
hyp.cov = [-0.5; 0; 0; 2; 0]; 
hyp.lik = [];              

% Covariance matrix with jitter
K = feval(covfunc{:}, hyp.cov, input_1) + (1e-5 * eye(length(input_1)));
% Cholesky decomposition (lower-triangular)
L = chol(K, 'lower');


% Generate sample functions and store them
rng(1);
num_samples = 3; % Number of functions to sample

colors = {
    [102/255, 194/255, 165/255],   % Soft Green
    [252/255, 141/255, 98/255],    % Soft Orange
    [141/255, 160/255, 203/255]    % Soft Purple
};

f_samples = cell(num_samples, 1);

% Generate and plot each sample function separately
for i = 1:num_samples
    f = L * randn(n,1); 
    f_samples{i} = f;
    
    % Plot the sample function
    figure;
    plot(input_1, f, 'LineWidth', 1.5, 'Color', colors{i});    
    % Customise the plot
    xlabel('Input x', 'FontSize', 12);
    ylabel('Function f(x)', 'FontSize', 12);
    title(sprintf('Sample Function %d from GP Prior', i), 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 12);
end

% Plot all sample functions together
figure;
hold on;

for i = 1:num_samples
    plot(input_1, f_samples{i}, 'LineWidth', 1.5, 'Color', colors{i}, 'DisplayName', sprintf('Sample %d', i));
end

% Customize the plot
xlabel('Input x', 'FontSize', 12);
ylabel('Function f(x)', 'FontSize', 12);
title('Sample Functions from GP Prior', 'FontSize', 14, 'FontWeight', 'bold');  % Main title in bold
legend('Location', 'best');
grid on;
set(gca, 'FontSize', 12);
hold off;

%% PART E
clear all; close all; clc;

% Load 'cw1e.mat'
load('cw1e.mat'); 

disp('Size of x:'); disp(size(x));  
disp('Size of y:'); disp(size(y));  

% Reshape data for plotting 
X1 = reshape(x(:,1), 11, 11);
X2 = reshape(x(:,2), 11, 11);
Y = reshape(y, 11, 11);

figure;
mesh(X1, X2, Y);
xlabel('x1');
ylabel('x2');
zlabel('y');
title('Data Visualization');
grid on;
rotate3d on;

%% GP Model with Sum of Two covSEard
meanfunc = @meanZero;  
hyp2.mean = [];   
covfunc2 = {@covSum, {@covSEard, @covSEard}};
likfunc = @likGauss;   

% Initialise hyperparameters
hyp2.cov = 0.1 * randn(6, 1);  % random values to break symmetry
hyp2.lik = log(0.1); 

% Number of iterations for optimisation
num_iters = 100;
% Optimising hyperparameters
hyp2_opt = minimize(hyp2, @gp, -num_iters, @infGaussLik, meanfunc, covfunc2, likfunc, x, y);

% Compute the NLML with optimised hyperparameters
[nlZ2, ~] = gp(hyp2_opt, @infGaussLik, meanfunc, covfunc2, likfunc, x, y);

% Make predictions at training inputs 
[mu2_train, s2_2_train] = gp(hyp2_opt, @infGaussLik, meanfunc, covfunc2, likfunc, x, y, x);

% Compute MSE on training data
mse2_train = mean((mu2_train - y).^2);

% Display the optimised hyperparameters
disp('Optimised hyperparameters (Model 2):');

disp('First covSEard:');
disp('Length scales (ell):'); disp(exp(hyp2_opt.cov(1:2)));
disp('Signal variance (sf):'); disp(exp(hyp2_opt.cov(3)));

disp('Second covSEard:');
disp('Length scales (ell):'); disp(exp(hyp2_opt.cov(4:5)));
disp('Signal variance (sf):'); disp(exp(hyp2_opt.cov(6)));
disp('Noise standard deviation (sn):'); disp(exp(hyp2_opt.lik));

% NLML and MSE
fprintf('Model 2 NLML: %.4f\n', nlZ2);
fprintf('Model 2 Training MSE: %.4f\n', mse2_train);

% Range for each input dimension
x1_min = min(x(:,1));
x1_max = max(x(:,1));
x2_min = min(x(:,2));
x2_max = max(x(:,2));

% Number of points along each dimension for the finer grid
n_fine = 100;  

% Generate evenly spaced points within the range
x1_fine = linspace(x1_min, x1_max, n_fine);
x2_fine = linspace(x2_min, x2_max, n_fine);

[X1_fine, X2_fine] = meshgrid(x1_fine, x2_fine);
x_test = [X1_fine(:), X2_fine(:)];

% Predictions with Model 2 on the finer grid
[mu2_fine, s2_2_fine] = gp(hyp2_opt, @infGaussLik, meanfunc, covfunc2, likfunc, x, y, x_test);
MU2_fine = reshape(mu2_fine, n_fine, n_fine);
S2_2_fine = reshape(s2_2_fine, n_fine, n_fine);

% Plot predictive mean as a surface
figure;
surf(X1_fine, X2_fine, MU2_fine, 'EdgeColor', 'none');
xlabel('x1');
ylabel('x2');
zlabel('Predicted y');
title('GP Predictions (Model 2)');
grid on;
rotate3d on;
colormap jet;
colorbar;
hold on;
plot3(x(:,1), x(:,2), y, 'k.', 'MarkerSize', 10);
hold off;

%% Predictive Uncertainty
% Compute the predictive standard deviation
STD2_fine = sqrt(S2_2_fine);
% Predictive standard deviation as a surface
figure;
surf(X1_fine, X2_fine, STD2_fine, 'EdgeColor', 'none');
xlabel('x1');
ylabel('x2');
zlabel('Predictive Std Dev');
title('Predictive Uncertainty');
grid on;
rotate3d on;
colormap jet;
colorbar;

% Contour plot of predictive mean
figure;
contourf(X1_fine, X2_fine, MU2_fine, 20, 'LineColor', 'none');
xlabel('x1');
ylabel('x2');
title('Predictive Mean Contour');
colormap jet;
colorbar;

% Heatmap of the predictive mean
figure;
imagesc(x1_fine, x2_fine, MU2_fine);
axis xy;  % Correct the axis direction
xlabel('x1');
ylabel('x2');
title('Predictive Mean Heatmap');
colormap jet;
colorbar;
