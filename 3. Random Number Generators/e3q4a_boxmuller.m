%% Standard normal random numbers with the
%% Box-Muller algorithm and its Marsaglia variant

% Define the parameters
clear variables % clear all variables from workspace
a = -4; % left truncation
b = 4; % right truncation
ngrid = 200; % number of grid steps
nsample = 2000000; % number of random samples

% Define the grid using the colon operator
deltax = (b-a)/ngrid; % grid step
x = a:deltax:b; % grid

% Box-Muller
U1 = rand(1,nsample/2);
U2 = rand(1,nsample/2);
tic
rho = sqrt(-2*log(U1));
theta = 2*pi*U2;
N = [rho.*cos(theta) rho.*sin(theta)]; % standard normal numbers
CPUtime1 = toc

% Scatter plot
close all
U = rand(1,nsample);
figure(1)
plot(N,U.*pdf('Normal',N,0,1),'.',x,pdf('Normal',x,0,1))
xlabel('x')
ylabel('f_N')

% Variant of Marsaglia
tic
V1 = 2*U1-1;
V2 = 2*U2-1;
W = V1.^2+V2.^2;
I = W<=1;
WI = W(I);
rho2 = sqrt(-2*log(WI)./WI);
N1 = [rho2.*V1(I) rho2.*V2(I)]; % standard normal numbers
CPUtime2 = toc
L = length(N1);
L/nsample
CPUtime2/CPUtime1

% Scatter plot
figure(2)
plot(N1,U(1:L).*pdf('Normal',N1,0,1),'.',x,pdf('Normal',x,0,1))
xlabel('x')
ylabel('f_N')