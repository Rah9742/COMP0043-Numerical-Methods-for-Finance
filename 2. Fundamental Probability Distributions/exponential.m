%% Plot and sample the exponential distribution

% Define the parameters
clear variables % clear all variables from workspace
mu = 2/3; % mean
nsample = 10^6; % number of random samples

% Define the grid
%x = linspace(0,5,101);
x = 0:0.05:5;

% Compute the PDF and the CDF
%f = 1/mu*exp(-x/mu);
%F = 1-exp(-x/mu);
f = pdf('Exponential',x,mu);
F = cdf('Exponential',x,mu);

% Open a plot window
close all
figure(1)
plot(x,f,'r',x,F,'b');

% Label the x axis
xlabel('x')

% Insert the legend
legend('PDF','CDF')

% Set the title
title('Exponential distribution with \mu = 2/3')

% Print the figure to a file
print('-dpdf','exponential.pdf')

%% Sample an exponential distribution
U = rand(nsample,1); % standard uniform random numbers
X = -mu*log(U); % transformation to exponential random numbers

% Bin the random variables in a histogram and normalise it
dx = x(2)-x(1); % bin width
[h,xx] = hist(X,x+dx/2);
h = h/(nsample*dx); % normalisation

figure(2)
plot(xx,h,'b',x,f,'r')
xlim([0 5])
xlabel('x')
ylabel('f')
legend('Sampled','Theory')
title('Exponential distribution with \mu = 0.2')

figure(3)
bar(xx,h)
hold on
plot(x,f,'r')
xlabel('x')
ylabel('f')
legend('Sampled','Theory')
title('Exponential distribution with \mu = 0.2')
