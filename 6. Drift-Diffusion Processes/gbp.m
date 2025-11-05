%% Monte Carlo simulation of geometric Brownian motion
% dS = mu*S*dt + sigma*S*dW

% Define parameters and time grid
clear variables % clear all variables from workspace
npaths = 20000; % number of paths
T = 1; % time horizon
nsteps = 200; % number of time steps
dt = T/nsteps; % time step
t = 0:dt:T; % observation times
mu = 0.2; sigma = 0.4; % model parameters
S0 = 1; % initial stock price

%% Monte Carlo

% Compute the increments of the arithmetic Brownian motion X = log(S/S0)
dX = (mu-0.5*sigma^2)*dt + sigma*randn(npaths,nsteps)*sqrt(dt);

% Accumulate the increments
X = [zeros(npaths,1) cumsum(dX,2)];

% Transform to geometric Brownian motion
S = S0*exp(X);

%% Expected, mean and sample paths
close all
figure(1)
EX = exp(mu*t); % expected path
plot(t,EX,'k',t,mean(S),':k',t,S(1:1000:end,:),t,EX,'k',t,mean(S),':k')
legend('Expected path','Mean path')
xlabel('t')
ylabel('X')
ylim([0,2.5])
title('Geometric Brownian motion dS = \muSdt + \sigmaSdW')
print('-dpng','gbppaths.png')

%% Probability density function at different times
figure(2)

subplot(3,1,1)
histogram(S(:,20),0:0.035:3.5,'normalization','pdf');
ylabel('f_X(x,0.15)')
xlim([0,3.5])
ylim([0,3.5])
title('Geometric Brownian motion: PDF at different times')

subplot(3,1,2)
histogram(S(:,80),0:0.035:3.5,'normalization','pdf');
xlim([0,3.5])
ylim([0,3.5])
ylabel('f_X(x,0.4)')

subplot(3,1,3)
histogram(S(:,end),0:0.035:3.5,'normalization','pdf');
xlim([0,3.5])
ylim([0,3.5])
xlabel('x')
ylabel('f_X(x,1)')
print('-dpng','gbpdensities.png')
