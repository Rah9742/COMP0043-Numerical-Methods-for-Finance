%% Compute the machine precision

% Using a for loop
epsm = 1;
for i = 1:100
    epsm = epsm/2;
    if 1+epsm == 1
        break
    end
end
epsm = 2*epsm
i = i-1
epsm = 1/2^i

% Using a while loop
epsm = 1;
while 1+epsm ~= 1
    epsm = epsm/2;
end
epsm = 2*epsm

% Built-in value
eps