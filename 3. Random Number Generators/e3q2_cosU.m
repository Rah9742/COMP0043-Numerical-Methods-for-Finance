%% Plot and sample the distribution of X = cos U

clear variables
close all

ngrid = 200;
nsample = 10000;

% (a) Compute the PDF
dx = 1/ngrid;
x = 0:dx:1;
f = 2./(pi*sqrt(1-x.^2));

% (b) Sample
dy = pi/ngrid;
y = -pi/2:dy:pi/2;
U = pi*(rand(nsample,1)-1/2);
hu = hist(U,y)/(nsample*(y(2)-y(1)));
X = cos(U);
hx = hist(X,x)/(nsample*(x(2)-x(1)));

% Plot the PDF of U
figure(1)
plot(y,hu,'r',y,ones(size(y))/pi,'b');
ylim([0 1])
xlabel('x')
ylabel('f')
legend('Sampled','Theory')
title('Distribution of U')

% Plot the PDF of X
figure(2)
plot(x,hx,'r',x,f,'b');
xlabel('x')
ylabel('f')
legend('Sampled','Theory')
title('Distribution of X = cos U')
