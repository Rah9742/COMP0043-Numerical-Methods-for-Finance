%% Plot and sample the non-central chi-square distribution

% Define the parameters
clear variables % clear all variables from workspace
d = 5; % degrees of freedom
lambda = 2; % non-centrality parameter
a = 0; % left truncation
b = 20; % right truncation
ngrid = 200; % number of grid intervals

% Define the grid with linspace
%x = linspace(a,b,ngrid+1);
%deltax = x(2)-x(1) % grid step 

% Define the grid with the colon operator
deltax = (b-a)/ngrid; % grid step
x = a:deltax:b;

%% Compute and plot the PDF and CDF
f = pdf('ncx2',x,d,lambda);
F = cdf('ncx2',x,d,lambda);

close all
figure(1) % open a plot window
plot(x,f,'r',x,F,'b')
xlabel('x')
title('Non-central chi-square PDF and CDF with d=5 and \lambda=2')
legend('PDF','CDF')
print('-dpng','ncchisq.png')

%% Sample the non-central chi-square distribution
% Add this as an exercise
