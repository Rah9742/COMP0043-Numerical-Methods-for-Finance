%% Matrix multiplication

clear variables

n = 3
A = rand(n)
B = rand(n)

% (a) MATLAB's built-in operator
disp('(a)')
C = A*B

% (b) Using three for loops
disp('(b)')
C1 = zeros(n);
for i = 1:n
    for j = 1:n
        for k = 1:n
            C1(i,j) = C1(i,j) + A(i,k)*B(k,j);
        end
    end
end
C1
diff = C-C1
maxdiff = max(max(abs(diff)))

% (c) As (b) swapping the two outer loops over i and j
disp('(c)')
C1 = zeros(n);
for j = 1:n
    for i = 1:n
        for k = 1:n
            C(i,j) = C(i,j) + A(i,k)*B(k,j);
        end
    end
end
C1
maxdiff = max(max(abs(C-C1)))

% (d) Using two for loops over i and j and sum()
disp('(d)')
C1 = zeros(n);
for i = 1:n
    for j = 1:n
        C(i,j) = sum(A(i,:).'.*B(:,j));
    end
end
C1
maxdiff = max(max(abs(C-C1)))

% (e) As (d) transposing A
disp('(e)')
AT = A.';
C1= zeros(n);
for i = 1:n
    for j = 1:n
        C(i,j) = sum(AT(:,i).*B(:,j));
    end
end
C1
maxdiff = max(max(abs(C-C1)))

% (f) As (d) transposing B
disp('(f)')
BT = B.';
C1 = zeros(n);
for i = 1:n
    for j = 1:n
        C(i,j) = sum(A(i,:).*BT(j,:));
    end
end
C1
maxdiff = max(max(abs(C-C1)))

%% (g) As (a-f) with increasing matrix size
disp('(g)')
for n = [10 100 1000]
    n
    A = rand(n);
    B = rand(n);
    
    % (a)
    tic
    C = A*B;
    toc
    
    % (b)
    tic
    C = zeros(n);
    for i = 1:n
        for j = 1:n
            for k = 1:n
                C(i,j) = C(i,j) + A(i,k)*B(k,j);
            end
        end
    end
    toc
    
    % (c)
    tic
    C = zeros(n);
    for j = 1:n
        for i = 1:n
            for k = 1:n
                C(i,j) = C(i,j) + A(i,k)*B(k,j);
            end
        end
    end
    toc
    
    % (d)
    tic
    C = zeros(n);
    for i = 1:n
        for j = 1:n
            C(i,j) = sum(A(i,:).'.*B(:,j));
        end
    end
    toc
    
    % (e)
    tic
    AT = A.';
    C = zeros(n);
    for i = 1:n
        for j = 1:n
            C(i,j) = sum(AT(:,i).*B(:,j));
        end
    end
    toc
    
    % (f)
    tic
    BT = B.';
    C = zeros(n);
    for i = 1:n
        for j = 1:n
            C(i,j) = sum(A(i,:).*BT(j,:));
        end
    end
    toc
end

tic
AT = A.';
toc

%% (g) Find how the CPU time of matrix multiplication scales with matrix size
nvec = [10 20 50 100 200 500 1000 2000 5000 10000];
CPUt1 = zeros(size(nvec));
CPUt2 = zeros(1,7);
CPUt3 = zeros(1,7);
for l = 1:length(nvec)
    n = nvec(l)
    A = rand(n);
    B = rand(n);
    
    tic
    C = A*B;
    CPUt1(l) = toc;

    if n <= 1000
        tic
        C = zeros(n);
        for i = 1:n
            for j = 1:n
                for k = 1:n
                   C(i,j) = C(i,j) + A(i,k)*B(k,j);
                end
            end
        end
        CPUt2(l) = toc;

        tic
        AT = A.';
        C = zeros(n);
        for i = 1:n
            for j = 1:n
                C(i,j) = sum(AT(:,i).*B(:,j));
            end
        end
        CPUt3(l) = toc;
    end
end
CPUt1.'
fitobj0 = fit(log10(nvec(2:4).'),log10(CPUt1(2:4).'),'poly1')
fitobj1 = fit(log10(nvec(5:end).'),log10(CPUt1(5:end).'),'poly1')
CPUt2.'
fitobj2 = fit(log10(nvec(2:7).'),log10(CPUt2(2:end).'),'poly1')
CPUt3.'
fitobj3 = fit(log10(nvec(2:7).'),log10(CPUt3(2:end).'),'poly1')

close all 
figure(1)
loglog(nvec(1:7),CPUt2,'ro',nvec(2:7),10^fitobj2.p2*nvec(2:7).^fitobj2.p1,'r')
hold on
loglog(nvec(1:7),CPUt3,'go',nvec(2:7),10^fitobj3.p2*nvec(2:7).^fitobj3.p1,'g')
loglog(nvec,CPUt1,'bo',nvec(2:4),10^fitobj0.p2*nvec(2:4).^fitobj0.p1,'b')
loglog(nvec(5:end),10^fitobj1.p2*nvec(5:end).^fitobj1.p1,'b')
legend('Triple loop',sprintf('a*n^{%.3f}',fitobj2.p1), ...
    'Double loop with sum()',sprintf('b*n^{%.3f}',fitobj3.p1), ...
    'Matlab',sprintf('c*n^{%.3f}',fitobj0.p1), ...
    sprintf('d*n^{%.3f}',fitobj1.p1),'Location','southeast')
xlabel('n')
ylabel('CPU t/s')
ylim([3E-5 20])
print('-dpng','matmul.png')