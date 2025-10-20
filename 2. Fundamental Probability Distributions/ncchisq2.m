%% Plot and sample the non-central chi-square distribution

% Define the parameters
clear variables % clear all variables from workspace
d = 5; % degrees of freedom
lambda = 2; % non-centrality parameter
a = 0; % left truncation
b = 20; % right truncation
ngrid = 200; % number of grid intervals
nsample = 10^6; % number of random samples

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
plot(x,f,'r',x,F,'b','LineWidth',2)
xlabel('x')
title('Non-central chi-square PDF and CDF with d=5 and \lambda=2')
legend('PDF','CDF')
print('-dpng','ncchisq.png')

%% Sample the non-central chi-square distribution
tic
%X = icdf('ncx2',rand(1,nsample),d,lambda); % 50 times slower
X = ncx2rnd(d,lambda,1,nsample);
toc

% Bar histogram
figure(2)
x2 = [x-deltax/2 x(end)+deltax/2]; % bin edges
ho = histogram(X,x2,'normalization','pdf'); % histogram object
hold on
plot(x,f,'r','LineWidth',2)
xlim([a b])
xlabel('x')
ylabel('f_X')
legend('Sampled','Theory')
title('Non-central chi-square PDF with d=5 and \lambda=2')

% Line histogram
figure(3)
h2 = ho.Values;
plot(x,h2,'b',x,f,'r--','LineWidth',2)
xlim([a b])
xlabel('x')
ylabel('f_X')
legend('Sampled','Theory')
title('Non-central chi-square PDF with d=5 and \lambda=2')

% Scatter plot
figure(4)
U = rand(1,1000);
plot(X(1:1000),U.*pdf('ncx2',X(1:1000),d,lambda),'.', ...
    x,pdf('ncx2',x,d,lambda),'LineWidth',2)
xlabel('x')
ylabel('f_X')
legend('Sampled','Theory')
title('Non-central chi-square PDF with d=5 and \lambda=2')
