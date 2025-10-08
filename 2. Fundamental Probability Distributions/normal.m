%% Plot and sample the normal distribution

% Define the parameters
clear variables % clear all variables from workspace
mu = 0.2; % mean
sigma = 0.1; % standard deviation
a = -0.4; % left truncation
b = 0.8; % right truncation
ngrid = 200; % number of grid intervals
nsample = 10^6; % number of random samples

% Define the grid with linspace
%x = linspace(a,b,ngrid+1);
%deltax = x(2)-x(1) % grid step

% Define the grid with the colon operator
deltax = (b-a)/ngrid; % grid step
x = a:deltax:b;

%% Compute and plot the PDF and CDF
f1 = 1/(sqrt(2*pi)*sigma)*exp(-((x-mu)/sigma).^2/2);
%f = normpdf(x,mu,sigma);
%f = normcdf(x,mu,sigma);
f = pdf('Normal',x,mu,sigma);
F = cdf('Normal',x,mu,sigma);

close all
figure(1) % open a plot window
plot(x,f1,'r',x,f,'bo',x,F,'g');
xlim([a b]) % limits of the x-axis
xlabel('x') % label of the x axis
legend('PDF','PDF, Matlab','CDF')
title('Normal distribution with \mu = 0.2 and \sigma = 0.1')
print('-dpng','normal.png') % print the figure to a file

%% Sample the normal distribution
%U = rand(nsample,1); % method 1: start from standard uniform random numbers
%X = mu + sigma*norminv(U); % method 1a1: scale and shift
%X = norminv(U,mu,sigma); % method 1a2
%X = mu + sigma*icdf('Normal',U,0,1); % method 1b1: scale and shift
%X = icdf('Normal',U,mu,sigma); % method 1b2 
X = mu + sigma*randn(nsample,1); % method 2

figure(2)
histfit(X,ngrid)
xlim([a b])
xlabel('x')
ylabel('f')
legend('Sampled','Normal fit')
title('Normal distribution with \mu = 0.2 and \sigma = 0.1')

figure(3)
h = hist(X,x)/(nsample*deltax);
plot(x,h,'b.',x,f,'r',x,h,'b.','LineWidth',2)
xlim([a b])
xlabel('x')
ylabel('f')
legend('Sampled','Theory')
title('Normal distribution with \mu = 0.2 and \sigma = 0.1')

figure(4)
bar(x,h)
hold on
plot(x,f,'r','LineWidth',2)
xlim([a b])
xlabel('x')
ylabel('f')
legend('Sampled','Theory')
title('Normal distribution with \mu = 0.2 and \sigma = 0.1')

figure(5)
x2 = [x-deltax/2 x(end)+deltax/2]; % bin edges
ho = histogram(X,x2,'normalization','pdf'); % histogram object
hold on
plot(x,f,'r','LineWidth',2)
xlim([a b])
xlabel('x')
ylabel('f')
legend('Sampled','Theory')
title('Normal distribution with \mu = 0.2 and \sigma = 0.1')

figure(6)
h2 = ho.Values;
plot(x,h2,'b.',x,f,'r',x,h2,'b.')
xlim([a b])
xlabel('x')
ylabel('f')
legend('Sampled','Theory')
title('Normal distribution with \mu = 0.2 and \sigma = 0.1')
