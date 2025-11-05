% Check numerically that Gaussians form a closed set with respect to
% the Fourier transform

clear all
close all

N = 1024;
Dx = 0.1;
x = Dx*(-N/2:N/2-1);
L = N*Dx;
Dxi = 2*pi/L; % Nyquist relation: Dx*Dxi = 2*pi/N
xi = Dxi*(-N/2:N/2-1);
W = N*Dxi; % = 2*pi/Dx

%%
a = 1;
fa = sqrt(a/pi)*exp(-a*x.^2); % Gaussian in x space
Fa = exp(-xi.^2/(4*a)); % Gaussian in xi space

Fn  = fftshift(ifft(ifftshift(fa)))*L;
fn  = fftshift( fft(ifftshift(Fa)))/L;
Fn1 = fftshift( fft(ifftshift(fa)))*Dx;
fn1 = fftshift(ifft(ifftshift(Fa)))/Dx;

figure(1), clf, hold on
plot(x,real(fn),'r')
plot(x,imag(fn),'g')
plot(x,fa,'y:')
axis([-5 5 0 1])
xlabel('x')
ylabel('f')
legend('Re(fn)','Im(fn)','fa')

figure(2), clf, hold on
plot(xi,real(Fn),'b')
plot(xi,imag(Fn),'m')
plot(xi,Fa,'c:')
axis([-10 10 0 1])
xlabel('\xi')
ylabel('F')
legend('Re(Fn)','Im(Fn)','Fa')

%%
figure(3), clf, hold on
plot(x,real(fn1),'r')
plot(x,imag(fn1),'g')
plot(x,fa,'y:')
axis([-5 5 0 1])
xlabel('x')
ylabel('f')
legend('Re(fn1)','Im(fn1)','fa')

figure(4), clf, hold on
plot(xi,real(Fn1),'b')
plot(xi,imag(Fn1),'m')
plot(xi,Fa,'c:')
axis([-10 10 0 1])
xlabel('\xi')
ylabel('F')
legend('Re(Fn1)','Im(Fn1)','Fa')