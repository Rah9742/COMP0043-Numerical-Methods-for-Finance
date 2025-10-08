%% Plot and sample the lognormal distribution

% Define the parameters
mu = 0.2;
sigma = 0.1;
nsample = 10^6;

% Define the grid
%x = linspace(0,2,101);
x = 0:0.02:2;

%% Compute the PDF and CDF
%f = 1/(sqrt(2*pi)*sigma)*exp(-((log(x)-mu)/sigma).^2/2)./x;
f = pdf('lognorm',x,mu,sigma);
F = cdf('lognorm',x,mu,sigma);

% Open a plot window
close all
figure(1)
plot(x,f,'r',x,F,'b');

% Label the x axis
xlabel('x')

% Insert the legend
legend('PDF','CDF')

% Set the title
title('Lognormal distribution with \mu = 0.2 and \sigma = 0.1')

% Print the figure to a file
print('-dpdf','lognormal.pdf')

%% Sample a normal distribution
%U = rand(nsample,1); % method 1
%X = mu + sigma*norminv(U,0,1); % method 1a
%X = norminv(U,mu,sigma); % method 1b
X = mu + sigma*randn(nsample,1); % method 2

% Sample the lognormal distribution by exponential transformation
Y = exp(X);

figure(2)
histfit(Y,100,'logn')
legend('Sampled','Lognormal fit')
title('Lognormal random variables with \mu = 0.2 and \sigma = 0.1')

% Bin the random variables in a histogram
[h,y] = hist(Y,x);

figure(3)
plot(y,h)
title('Lognormal random variables with \mu = 0.2 and \sigma = 0.1')

figure(4)
bar(y,h)
title('Lognormal random variables with \mu = 0.2 and \sigma = 0.1')
