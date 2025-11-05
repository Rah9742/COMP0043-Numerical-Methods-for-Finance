%% Monte Carlo simulation of the Brownian bridge
% dX = (b-X)/(T-t)*dt + sigma*dW

% Define parameters and time grid
clear variables % clear all variables from workspace
npaths = 20000; % number of paths
T = 1; % time horizon
nsteps = 200; % number of time steps
dt = T/nsteps; % time step
t = 0:dt:T; % observation times
sigma = 0.3; % volatility
a = 0.8; % initial value
b = 1; % final value

%% Monte Carlo method 1

%Allocate and initialise all paths
X = [a*ones(1,npaths); zeros(nsteps-1,npaths); b*ones(1,npaths)];

% Compute the Brownian bridge with Euler-Maruyama
for i = 1:nsteps-1
     X(i+1,:) = X(i,:) + (b-X(i,:))/(nsteps-i+1) + sigma*randn(1,npaths)*sqrt(dt);
end

%% Monte Carlo method 2

% Compute the increments of driftless arithmetic Brownian motion
%dW = sigma*randn(nsteps,npaths)*sqrt(dt);

% Accumulate the increments of arithmetic Brownian motion
%W = cumsum([a*ones(1,npaths); dW]);

% Compute the Brownian bridge with X(t) = W(t) + (b-W(T))/T*t
%X = W + repmat(b-W(end,:),nsteps+1,1)/T.*repmat(t',1,npaths);

%% Expected, mean and sample paths
figure(1)
close all
EX = a + (b-a)/T*t; % expected path
plot(t,EX,'k',t,mean(X,2),':k',t,X(:,1:1000:end),t,EX,'k',t,mean(X,2),':k')
legend('Expected path','Mean path')
xlabel('t')
ylabel('X')
%sdevmax = sigma*sqrt(T)/2;
%ylim([(a+b)/2-4*sdevmax (a+b)/2+4*sdevmax])
title('Brownian bridge dX = ((b-X)/(T-t))dt + \sigmadW')
print('-dpng','bbpaths.png')
