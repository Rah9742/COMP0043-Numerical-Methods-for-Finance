% Check numerically the Fourier pair Laplace <-> Lorentzian

clear all

N = 1024;
Dt = 0.1;
t = Dt*(-N/2:N/2-1);
T = N*Dt;
% Nyquist relation: Dt*Dnu = 1/N
Dnu = 1/T;
nu = Dnu*(-N/2:N/2-1);
Nu = N*Dnu; % = 1/Dt;

a = 1;
fa = a/2*exp(-a*abs(t)); % Laplace (or double exponential)
Fa = a^2./(a^2+(2*pi*nu).^2); % Lorentzian (or Cauchy)

Fn  = fftshift(ifft(ifftshift(fa)))*T;
fn  = fftshift( fft(ifftshift(Fa)))/T;
Fn1 = fftshift( fft(ifftshift(fa)))*Dt;
fn1 = fftshift(ifft(ifftshift(Fa)))/Dt;

figure(5), clf, hold on
plot(t,real(fn),'r')
plot(t,imag(fn),'g')
plot(t,fa,'y:')
axis([-10 10 0 1])
xlabel('t')
ylabel('f')
legend('Re(fn)','Im(fn)','fa')

figure(6), clf, hold on
plot(nu,real(Fn),'b')
plot(nu,imag(Fn),'m')
plot(nu,Fa,'c:')
axis([-10/pi 10/pi 0 1])
xlabel('\nu')
ylabel('F')
legend('Re(Fn)','Im(Fn)','Fa')

figure(7), clf, hold on
plot(t,real(fn1),'r')
plot(t,imag(fn1),'g')
plot(t,fa,'y:')
axis([-10 10 0 1])
xlabel('t')
ylabel('f')
legend('Re(fn1)','Im(fn1)','fa')

figure(8), clf, hold on
plot(nu,real(Fn1),'b')
plot(nu,imag(Fn1),'m')
plot(nu,Fa,'c:')
axis([-10/pi 10/pi 0 1])
xlabel('\omega')
ylabel('F')
legend('Re(Fn1)','Im(Fn1)','Fa')