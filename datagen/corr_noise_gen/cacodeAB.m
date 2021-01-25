function CA = cacodeAB(n, chiprate, fs)
% The funtion CA = cacode(n, chiprate, fs)
% returns the Gold code for GPS satellite ID n :
% n = 1..32
% chiprate = number of chips per second.
% fs = sampling frequency
% The code is represented at levels :
%   -1 for bit = 0
%   +1 for bit = 1

% Phase assignments
phase = [2 6; 3 7; 4 8; 5 9; 1 9; 2 10; 1 8; 2 9; 3 10;
    2 3; 3 4; 5 6; 6 7; 7 8; 8 9; 9 10; 1 4; 2 5;
    3 6; 4 7; 5 8; 6 9; 1 3; 4 6; 5 7; 6 8; 7 9;
    8 10; 1 6; 2 7; 3 8; 4 9]	;

% Initial state - all ones
G1 = -1*ones(1,10);
G2 = G1;
% Select taps for G2 delay
s1 = phase(n,1);
s2 = phase(n,2);
tmp = 0;
for k = 1:1023;
    % Gold-code
    G(k) = G2(s1)*G2(s2)*G1(10);
    % Generator 1 - shift reg 1
    tmp = G1(1);
    G1(1)=G1(3)*G1(10);
    G1(2:10) = [tmp G1(2:9)];
    % Generator 2 - shift reg 2
    %disp(G2);
    %disp("Hello wolrd1 cacodeAB");
    
    tmp = G2(1);
    G2(1) = G2(2)*G2(3)*G2(6)*G2(8)*G2(9)*G2(10);
    G2(2:10)=[tmp G2(2:9)];
end;

% Resample - doesn't work for (k*chiprate/fs) > 1023
% but replica chiprate is constant in this implementation

% k = 1:n_samples;ceil(k*chiprate/fs)
% CA(k) = G(ceil(k*chiprate/fs));

% Version A. Blais
fEch = lcm(chiprate,fs);
nbPointsPerChip = fEch/chiprate;
opEch = ones(1,nbPointsPerChip);
caEch = kron(G,opEch);
CA = caEch(1:fEch/fs:end);
