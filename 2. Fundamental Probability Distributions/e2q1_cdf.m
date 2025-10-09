%% Plot the normal distribution

clear variables
close all

% Define the parameters
mu = 0.2; % mean
sigma = 0.1; % standard deviation
a = -0.4; % left truncation
b = 0.8; % right truncation
nsteps = 120; % number of grid steps

% Define the grid with linspace
%x = linspace(a,b,ngrid+1); % grid step
%deltax = x(2)-x(1) % grid

% Define the grid with the colon operator
deltax = (b-a)/nsteps; % grid step
x = a:deltax:b; % grid

%% Compute the PDF and the CDF
%f = 1/(sqrt(2*pi)*sigma)*exp(-((x-mu)/sigma).^2/2);
f = pdf('Normal',x,mu,sigma);
F = cdf('Normal',x,mu,sigma);

%% (a)
Fa = cumsum(f)*deltax;

%% (b)
Fb = f;
for i = 2:length(f)
    Fb(i) = sum(f(1:i));
end
Fb = Fb*deltax;

%% (c)
Fc= f;
for i = 2:length(f)
    Fc(i) = trapz(f(1:i));
end
Fc = Fc*deltax;

%% (d)
Fd = f;
Fd(2) = 0.5*(f(1)+f(2));
for i = 3:length(f)
    Fd(i) = Fd(i-1)+0.5*(f(i-1)+f(i));
end
Fd = Fd*deltax;

%% (e)
Fe = cumsum(f)-0.5*(f(1)+f);
Fe(1) = f(1);
Fe = Fe*deltax;

%% (f)
F(1:10)
Fa(1:10)
Fb(1:10)
Fc(1:10)
Fd(1:10)
Fe(1:10)

F (nsteps/2-4:nsteps/2+5)
Fa(nsteps/2-4:nsteps/2+5)
Fb(nsteps/2-4:nsteps/2+5)
Fc(nsteps/2-4:nsteps/2+5)
Fd(nsteps/2-4:nsteps/2+5)
Fe(nsteps/2-4:nsteps/2+5)

%% (g)
figure(1) % open a plot window
plot(x,F,'g',x,Fa,'b',x,Fc,'b.');
xlim([a b]) % limits of the x-axis
xlabel('x') % label of the x axis
ylabel('F') % label of the y axis
legend('cdf','cumsum','trapz')
title('Normal distribution with \mu = 0.2 and \sigma = 0.1')
print('-dpdf','normal.pdf') % print the figure to a file
