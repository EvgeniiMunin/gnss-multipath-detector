% Zoom 2-D sur le pic de correlation.

% Attention, on peut toujours tomber sur une partie du signal o�
% il y a un changement de bit d'information. Donc le niveau de 
% correlation ne sera pas optimal dans ce cas l� si N > 1.

% Mode non-interactif, par exemple :
% interactive = false;
% path = './Data/';
% file = 'pluto_2084kS_2MHz_71_dB_GPS_04112019_1306.cfile'
% file_name = [path, file];
% Fs = 2.084e6; Fif = 0; LO_offset = -10e3; sign_Q = +1; file_format = 'cfile';
% if_signal = false;
% satNum = 6; eps_c = 0.130e-3; eps_f = -500;
% zoom_acquisition_GPS_L1_CA

% Mode interactif
interactive = true, choice = 'debug'%, zoom_acquisition_GPS_L1s_CA_no_oscil

close all;

% Mode interactif ou non ?
if exist("interactive","var") != 1 || (interactive == true)
  clear -x choice;
  interactive = true;
  if exist("choice","var") != 1
    choice = 1;
  endif
else
  clear -x file_name Fs Fif LO_offset sign_Q file_format if_signal ...
    satNum eps_c eps_f;
  interactive = false;
  % Une rustine pour la g�n�ration d'image de bruit par batch
  if (strcmp(file_format,"noise") == 1)
    choice = "noise";
    sauv_satNum = satNum;
    satNum = 1;
  else  
    choice = 0;
  endif
endif

%addpath './Sub_Functions/';

% For the detection threshold
%pkg load statistics;

%----LES CONSTANTES GPS--------------------------------------------%

FL1 = 1575.42e6;
Fc = 1.023e6; Tc = 1/Fc;
Nc = 1023;

% L'excursion Doppler maximale
dopp_max = 2000;
% Le pas en fr�quence, en Hz
dopp_step = 10;

% L'excursion maximale en d�lai qui sera visualis�e autour du pic :
tau_max = 2*Tc;    % Donc +- 2 chips autour du pic
% Le nombre de points de correlation sur [0, tau_max]
Np = 60;

if (interactive == true)
  switch choice
    case {1}
      if_signal = true;
      path = 'C:\Users\blaisan\Downloads\'
      % path = 'C:\Users\blais\Downloads\';
      % path = 'C:\Users\Antoine Blais\Downloads\';
      % path = '~/T�l�chargements/';
      file = 'test_real_long.dat'
      file_name = [path,file];
      Fs = 23.104e6; Fif = 4.348e6; LO_offset = 0; sign_Q = 0;
      file_format = 'old';
      
      % 3, -4250 Hz, 0.165 ms, 41 dB/Hz : B_P = 100 Hz
      % 11, -250 Hz, 0.916 ms, 45 dB/Hz : B_P = 100 Hz
      % 14, -2250 Hz, 0.233 ms, 41 dB/Hz : B_P = 100 Hz
      % 17, 2750 Hz, 0.279 ms, 43 dB/Hz : B_P = 100 Hz
      % 19, -3500 Hz, 0.138 ms, 45 dB/Hz : B_P = 100 Hz
      % 20, 1750 Hz, 0.387 ms, 47 dB/Hz : B_P = 10 Hz
      % 23, 3500 Hz, 0.660 ms, 41 dB/Hz
      % 28, -1500 Hz, 0.418 ms, 42 dB/Hz
      % 32, 0000 Hz, 0.515 ms, 47 dB/Hz

      % satNum = 3; B_P = 100; eps_c = 0.165e-3; eps_f = -4250;
      % satNum = 11; B_P = 100; eps_c = 0.916e-3; eps_f = -250;
      % satNum = 14; B_P = 100; eps_c = 0.233e-3; eps_f = -2250;
      % satNum = 17; B_P = 100; eps_c = 0.279e-3; eps_f = 2750
      % satNum = 19; B_P = 100; eps_c = 0.138e-3; eps_f = -3500;
      satNum = 20; B_P = 10; eps_c = 0.387e-3; eps_f = 1750;
    case {2}
      if_signal = true;
      path = './Data/';
      file = 'SwRxData_Band0_FE0_ANT0_f1575420000.stream'
      file_name = [path,file];
      Fs = 20e6; Fif = 5000445.89; LO_offset = 0; sign_Q = +1;
      file_format = 'stream';
      
      satNum = 4; eps_c = 0.117e-3; eps_f = -1500; % IFEN : C/N0 = 46.78 dBHz
      % satNum = 8; % IFEN : C/N0 = 48.12 dBHz
      % satNum = 16; % IFEN : C/N0 = 40.50 dBHz
      % satNum = 27; % IFEN : C/N0 = 50.58 dBHz
    case {3}
      if_signal = false;
      path = './Data/';
      file = 'pluto_2084kS_2MHz_71_dB_GPS_04112019_1306.cfile'
      file_name = [path, file];
      Fs = 2.084e6; Fif = 0; LO_offset = -10e3; sign_Q = +1;
      file_format = 'cfile';
      
      satNum =  6;
    case {4}
      if_signal = true;
      path = './Data/';
      file = 'SwRxData-191113151614_Band0_FE0_ANT0_f1575420000.stream'
      file_name = [path,file];
      Fs = 20e6; Fif = 5000445.89; LO_offset = 0; sign_Q = +1;
      file_format = 'stream';
      
      % satNum = 18;
      satNum = 8; eps_c = 0.993e-3; eps_f = -2500.0;
    case {'test'}
      if_signal = false;
      path = './Data/';
      file_I = 'voieI_121208_16H35.dat';
      file_Q = 'voieQ_121208_16H35.dat';
      file_name_I = [path,file_I];
      file_name_Q = [path,file_Q];
      Fs = 10e6; Fif = 0; LO_offset = 0; sign_Q = +1; file_format = 'dat';
      
      % satNum = 3, 6, 11, ...
      satNum = 11; eps_c = 0.870e-3; eps_f = 3000;
    case {'debug'}
      if_signal = true;
      % Il semble qu'avec if_signal == true la g�n�ration du rapport C/N0 soit
      % trop faible de 3 dB. Ou alors c'est l'estimation de ce rapport qui 
      % n'est pas correcte.
      if (if_signal == true)
        Fs = 23.104e6; Fif = 4.348e6; LO_offset = 0; sign_Q = 0;
        file_format = 'debug';
        
        satNum = 1; eps_c = 0.4e-3; eps_f = 1800;
      else
        Fs = 20e6; Fif = 0; LO_offset = 0; sign_Q = +1;
        file_format = 'debug';
        
        satNum = 1; eps_c = 0.4e-3; eps_f = 1800;
      endif
      C_N0_dB = 100; C_N0 = 10^(C_N0_dB/10);
    case {'noise'}
      if_signal = true;
      if (if_signal == true)
        Fs = 20e6; Fif = 5000445.89; LO_offset = 0; sign_Q = +1;
        file_format = 'noise';
        
        satNum = 1; eps_c = 0.0e-3; eps_f = 0;
      else
        Fs = 20e6; Fif = 0; LO_offset = 0; sign_Q = +1;
        file_format = 'noise';
        
        satNum = 1; eps_c = 0.0e-3; eps_f = 0;
      endif
    otherwise
      error("Invalid choice");
  endswitch
else
  if exist("file_name","var") != 1
    error("File name not specified!");
  endif
  if exist("Fs","var") != 1
    error("Sampling frequency not specified!");
  endif
  if exist("Fif","var") != 1
    warning("Intermediate frequency not specified, defaulting to 0");
    Fif = 0;
    if_signal = false;
  else
    if (Fif != 0)
      if_signal = true;
    else
      if_signal = false;
    endif
  endif
  if exist("LO_offset","var") != 1
    error("Local oscillator offset not specified!");
  endif
  if exist("sign_Q","var") != 1
    error("Sign of Q not specified!");
  endif
  if exist("file_format","var") != 1
    error("File format not specified!");
  endif
  if exist("satNum","var") != 1
    error("Satellite number not specified!");
  endif
  % Code delay, in units of ms
  if exist("eps_c","var") != 1
    error("Code delay not specified!");
  endif
  % Doppler shift, in units of Hz
  if exist("eps_f","var") != 1
    error("Doppler shift not specified!");
  endif
endif

% Code delay, in samples
eps_c = round(eps_c*Fs);

% Sampling period
Ts = 1/Fs;

% Nombre de p�riodes de code d'int�gration coh�rente
N = 1;
% Dur�e d'int�gration coh�rente
Ti = N*Nc*Tc;

% Nombre de sommations non-coh�rentes
M = 1;
% Pour l'�tude du bruit, c'est corrModCos et corrModSin qui sont utilis�s.
% Or ces deux quantit�s ne peuvent pas �tre cumul�es.
if (strcmp(choice,"noise") == 1) M = 1; endif

%----LES DONNEES---------------------------------------------------%

% Number of samples of the signal we need.
Ns = N*M*Nc/Fc*Fs;

% We load the number of samples needed
switch file_format
  case {'old'}
    if exist(file_name,"file") != 2
      error("File does not exist: %s",file_name);
      return;
    endif
    voieI = DataReader_AB(file_name,Ns); voieI = voieI.';
    voieQ = zeros(1,Ns);
  case {'cfile'}
    if exist(file_name,"file") != 2
      error("File does not exist: %s",file_name);
      return;
    endif
    x = read_complex_binary(file_name,Ns);
    voieI = real(x); voieQ = imag(x);
    clear x y;
  case {'stream'}
    if exist(file_name,"file") != 2
      error("File does not exist: %s",file_name);
      return;
    endif
    voieI = reader_IFEN_SX3_AB(file_name,Ns); voieI = voieI.';
    voieQ = zeros(1,Ns);
  case {'dat'}
    if (exist(file_name_I,"file") != 2)
      error("File does not exist: %s",file_name_I);
      if (exist(file_name_Q,"file") != 2)
        error("File does not exist: %s",file_name_Q);
      endif
      return;
    elseif (exist(file_name_Q,"file") != 2)
      error("File does not exist: %s",file_name_Q);
      return;
    endif
    fid = fopen(file_name_I,'r');
    voieI = fread(fid,Ns,'uint16');
    fclose(fid);
    % On passe en entier sign? et en ligne
    voieI = voieI.' - 2^15;

    fid = fopen(file_name_Q,'r');
    voieQ = fread(fid,Ns,'uint16');
    fclose(fid);
    % On passe en entier sign? et en ligne
    voieQ = voieQ.' - 2^15;
  case {'debug'}
    % Le code PRN du signal
    code_PRN1 = cacodeAB(satNum,Fc,Fs);
    % Code p�riodis� N*M fois
    code_debug = kron(ones(1,N*M),code_PRN1);
    codeDelai = [code_debug(end - eps_c + 1:end) code_debug(1:end - eps_c)];
    tz = [0:length(codeDelai) - 1]/Fs;
    
    % choose between fake signal and fake noise
    choice_fake = 'fake_noise'
    switch choice_fake
      case('fake_noise')
        'CHECK FAKE NOISE OPTION, NOISE GENER'
        voieI = 1.0*randn(1,length(tz));
        if (if_signal == true)
          voieQ = zeros(1,length(tz));
        else
          voieQ = 1.0*randn(1,length(tz));
        endif
      case('fake_signal')
        if (if_signal == true)
          % En l'absence de sp�cification de bande passante, on prend [0, Fs/2].
          % C = 1/2; P_bruit = 1/(C/N0)*C*Fs
          P_noise = 1/C_N0/2*Fs; sigma_noise = sqrt(P_noise);
          voieI = codeDelai.*cos(2*pi*(Fif + eps_f + LO_offset)*tz + pi/4) + ...
            + sigma_noise*randn(1,length(tz));
          voieQ = zeros(1,length(tz));
        else
          % C_I = C_Q = 1/2; P_bruit_I = P_bruit_Q = 1/(C/N0)*C*Fs
          P_noise = 1/C_N0/2*Fs; sigma_noise = sqrt(P_noise);
          voieI = codeDelai.*cos(2*pi*(Fif + eps_f + LO_offset)*tz + pi/4) + ...
            + sigma_noise*randn(1,length(tz));
          voieQ = codeDelai.*sin(2*pi*(Fif + eps_f + LO_offset)*tz + pi/4) + ...
            + sigma_noise*randn(1,length(tz));
        endif
      otherwise
    endswitch
    clear tz code_PRN1 code_debug codeDelai;
  case {'noise'}
    P_noise = 1;
    if (if_signal == true)
      sigma_noise = sqrt(P_noise);
      voieI = sigma_noise*randn(1,Ns);
      voieQ = sigma_noise*zeros(1,Ns);
    else
      sigma_noise = sqrt(P_noise/2);
      voieI = sigma_noise*randn(1,Ns);
      voieQ = sigma_noise*randn(1,Ns);
    endif
  otherwise

endswitch

voieQ = sign_Q*voieQ;
voieI = voieI - mean(voieI);
voieQ = voieQ - mean(voieQ);

% Le code PRN local
code_PRN1 = cacodeAB(satNum,Fc,Fs);
lC = length(code_PRN1);

% Code p�riodis� sur une p�riode d'int�gration coh�rente
code_PRN1_N = kron(ones(1,N),code_PRN1);
lC_N = lC*N;
clear code_PRN1;

lI = length(voieI);
% plot(voieI,'b'); pause;
% clf; plot(abs(fft(voieI)),'b'); pause;

%----L'ACQUISITION------------------------------------------------%

% Le temps qui correspond
t = [0:lC_N - 1]/Fs;

% Je veux Np �chantillons sur [0, tau_max], c'est � dire Np*(Nc*Tc)/tau_max sur
% une p�riode de code. Avec Fs j'en ai lC. Je dois donc interpoler d'un facteur
f_interp = Np*(Nc*Tc)/tau_max/lC; f_interp = 0;
if (f_interp <= 1)
  fprintf(1,"Interpolation factor <= 1. Set to 1.\n");
  Np = ceil(tau_max*lC/(Nc*Tc));
  f_interp = 1; N_interp = lC_N; Fs_interp = Fs;
  lC_i = lC;
else
  fprintf(1,'Interpolation factor = %f\n',f_interp);
  lC_i = round(Np*(Nc*Tc)/tau_max);
  N_interp = N*lC_i;
  Fs_interp = Fs/lC*lC_i;
endif
t_interp = [0:N_interp - 1]/Fs_interp;

lC_N_i = lC_i*N;
lC_2 = round(lC_N_i/2);

% Le nombre de z�ros pour padder le spectre, pour interpoler le signal temporel
N_pad = N_interp - lC_N;
pad = zeros(1,N_pad);

% Les fr�quences de la r�plique locale
fDop = [-dopp_max:dopp_step:dopp_max] - LO_offset + eps_f;
lDop = length(fDop);

co_I = zeros(1,lC_N);
si_I = zeros(1,lC_N);
fft_co_I = zeros(1,lC_N);
fft_si_I = zeros(1,lC_N);
fft_co_I_PRN = zeros(1,lC_N);
fft_si_I_PRN = zeros(1,lC_N);
fft_co_I_PRN_pad = zeros(1,N_interp);
fft_si_I_PRN_pad = zeros(1,N_interp);
corrIco = zeros(1,N_interp);
corrIsi = zeros(1,N_interp);
corrModCos = zeros(lDop,N_interp);
corrModSin = zeros(lDop,N_interp);

if (if_signal == false)
  co_Q = zeros(1,lC_N);
  si_Q = zeros(1,lC_N);
  fft_co_Q = zeros(1,lC_N);
  fft_si_Q = zeros(1,lC_N);
  fft_co_Q_PRN = zeros(1,lC_N);
  fft_si_Q_PRN = zeros(1,lC_N);
  fft_co_Q_PRN_pad = zeros(1,N_interp);
  fft_si_Q_PRN_pad = zeros(1,N_interp);
  corrQsi = zeros(1,N_interp);
  corrQco = zeros(1,N_interp);
endif
corrMod = zeros(lDop,N_interp);
corrModcompl = zeros(lDop,N_interp);

% La FFT du code PRN
fft_PRN = fft(code_PRN1_N);
clear code_PRN1_N;
% La conjugaison remplace le retournement temporel n�cessaire
% � l'impl�mentation de la corr�lation par la convolution.
fft_PRN = conj(fft_PRN);

h = waitbar(0,'Acquisition en cours');

% Frequency grid
for i = [1:lDop]

  waitbar(i/lDop,h);

  % Non-coherent summations
  for k = [1:M]
    
    co_I = voieI(1 + (k - 1)*lC_N:k*lC_N).*cos(2*pi*(Fif + fDop(i))*t);
    si_I = voieI(1 + (k - 1)*lC_N:k*lC_N).*sin(2*pi*(Fif + fDop(i))*t);
    fft_co_I = fft(co_I);
    fft_si_I = fft(si_I);
    if (if_signal == false)
      co_Q = voieQ(1 + (k - 1)*lC_N:k*lC_N).*cos(2*pi*(Fif + fDop(i))*t);
      si_Q = voieQ(1 + (k - 1)*lC_N:k*lC_N).*sin(2*pi*(Fif + fDop(i))*t);
      fft_co_Q = fft(co_Q);
      fft_si_Q = fft(si_Q);
    endif

    fft_co_I_PRN = fft_co_I.*fft_PRN;
    fft_si_I_PRN = fft_si_I.*fft_PRN;
    if (if_signal == false)
      fft_co_Q_PRN = fft_co_Q.*fft_PRN;
      fft_si_Q_PRN = fft_si_Q.*fft_PRN;
    endif

    % Il faut conserver la sym�trie du spectre, sinon le signal temporel 
    % n'est pas r�el apr�s l'ifft.
    if (f_interp > 1)
      fft_co_I_PRN_pad = [fft_co_I_PRN(1:lC_2) pad fft_co_I_PRN(lC_2 + 1:end)];
      fft_si_I_PRN_pad = [fft_si_I_PRN(1:lC_2) pad fft_si_I_PRN(lC_2 + 1:end)];
      if (if_signal == false)
        fft_co_Q_PRN_pad = [fft_co_Q_PRN(1:lC_2) pad fft_co_Q_PRN(lC_2 + 1:end)];
        fft_si_Q_PRN_pad = [fft_si_Q_PRN(1:lC_2) pad fft_si_Q_PRN(lC_2 + 1:end)];
      endif
    else
      fft_co_I_PRN_pad = fft_co_I_PRN;
      fft_si_I_PRN_pad = fft_si_I_PRN;
      if (if_signal == false)
        fft_co_Q_PRN_pad = fft_co_Q_PRN;
        fft_si_Q_PRN_pad = fft_si_Q_PRN;
      endif
    endif
    
    % ifft(X,N_interp) padde X avec des z�ros � la fin du spectre, ce qui rompt
    % la sym�trie et donne un signal complexe en temporel.
    corrIco = real(ifft(fft_co_I_PRN_pad));
    corrIsi = real(ifft(fft_si_I_PRN_pad));
    if (if_signal == false)
      corrQco = real(ifft(fft_co_Q_PRN_pad));
      corrQsi = real(ifft(fft_si_Q_PRN_pad));
    endif

    if (if_signal == false)
      corrModCos(i,:) = (corrIco + corrQsi);
      corrModSin(i,:) = (corrQco - corrIsi);
      % Pour une estimation correcte du rapport C/N0 :
      corrMod(i,:) += corrModCos(i,:).^2 + corrModSin(i,:).^2;
      % Pour mieux voir les lobes secondaires :
      % corrMod(i,:) += sqrt(corrModCos(i,:).^2 + corrModSin(i,:).^2);
      corrModcompl(i,:) += corrModCos(i,:) + j * corrModSin(i,:);
    else
      corrModCos(i,:) = corrIco;
      corrModSin(i,:) = -corrIsi;
      corrMod(i,:) += corrIco.^2 + corrIsi.^2;
      % corrMod(i,:) += sqrt(corrIco.^2 + corrIsi.^2);
      corrModcompl(i,:) += corrModCos(i,:) + j * corrModSin(i,:);
    endif
    
  end
end

close(h);

clear co_I si_I co_Q si_Q;
clear fft_co_I fft_si_I fft_co_Q fft_si_Q;
clear fft_co_I_PRN fft_si_I_PRN fft_co_Q_PRN fft_si_Q_PRN;
clear fft_co_I_PRN_pad fft_si_I_PRN_pad fft_co_Q_PRN_pad fft_si_Q_PRN_pad;
clear corrIco corrIsi corrQco corrQsi;
clear fft_PRN z_left z_right;

[corrMax, indDop] = max(abs(corrMod));
[corrMaxMax,indT] = max(corrMax);
fprintf(1,'Retard = %f ms\n',t_interp(indT)/1e-3-floor(t_interp(indT)/1e-3));
fprintf(1,'Doppler = %f Hz\n',fDop(indDop(indT)) + LO_offset);
fprintf(1,'Niveau Max de correlation = %f\n',corrMaxMax);

if (if_signal == false)
  % Power of the noise
  P_r = mean(mean(corrMod))/M;
  N0 = 8*P_r/Ti;
  % Power of the signal
  P_s = corrMaxMax/M;
  C = 8*P_s/Ti^2;
else
  % Power of the noise
  P_r = mean(mean(corrMod))/M;
  N0 = 8*P_r/Ti;
  % Power of the signal
  P_s = corrMaxMax/M;
  C = 8*P_s/Ti^2;
endif
CN0_dB = 10.0*log10(C/N0);

% Calculate detection threshold
%CN0_ref = 30; P_fa = 10^-3; P_d = 0.95;
%CN0_lin = 10^(CN0_ref/10);

%M_min = 1; M_max = 50;
%condition = true;

%while (condition && M_min < M_max)
%  % Threshold for Pfa
%  thresh = chi2inv(1 - P_fa,2*M_min);
%
%  % Non-centrality parameter
%  lambda = 2.0*CN0_lin*M_min*Ti;
%
%  if ncx2cdf(thresh,2*M_min,lambda) < (1 - P_d) 
%    condition = false;
%  else
%    M_min = M_min + 1;
%  end
%end

%if M_min == M_max
%  fprintf(1,'La boucle de calcul du seuil n a pas converg�\n');
%else
%  Pd_eff = 100.0 - ncx2cdf(thresh,2*M_min,lambda)*100.0;
%  fprintf('Pd = %.3f, threshold = %.3f, M = %d\n',Pd_eff,thresh,M_min);
%end

%thresh = thresh*P_r/M;

%if corrMaxMax > thresh
%  fprintf(1,'C/N0 = %.1f dB/Hz **\n',CN0_dB);
%else
%  fprintf(1,'C/N0 = %.1f dB/Hz\n',CN0_dB);
%end

clear corrMax corrMaxMax;

if interactive == true
  figure("name","Correlation au pic en fonction de l'�cart de fr�quence",...
  "numbertitle","off");
  subplot(1,3,1)
  plot(fDop,corrModCos(:,indT)); grid on;
  xlabel('Hz'); title('Correlation voie I');
  subplot(1,3,2)
  plot(fDop,corrModSin(:,indT)); grid on;
  xlabel('Hz'); title('Correlation voie Q');
  subplot(1,3,3)
  plot(fDop,corrMod(:,indT)); grid on;
  xlabel('Hz'); title('Somme des correlations I^2 + Q^2');
  
  figure("name","Correlation au pic en fonction du d�lai",...
  "numbertitle","off");
  t_plot = [0:lC_N_i - 1]/Fs_interp;
  subplot(1,3,1)
  plot(t_plot,corrModCos(indDop(indT),:)); grid on;
  xlabel('s'); title('Correlation voie I');
  subplot(1,3,2)
  plot(t_plot,corrModSin(indDop(indT),:)); grid on;
  xlabel('s'); title('Correlation voie Q');
  subplot(1,3,3)
  plot(t_plot,corrMod(indDop(indT),:)); grid on;
  xlabel('s'); title('Somme des correlations I^2 + Q^2');

  % figure();
  % [x,y] = meshgrid(t,fDop - LO_offset);
  % mesh(x,y,corrMod);
  % xlabel('s'); ylabel('Hz'); zlabel('Correlation');
  % title(sprintf('Satellite %i',satNum));
  % figure();
  % [x,y] = meshgrid(t_plot,fDop + LO_offset - eps_f);
  % mesh(x,y,corrMod);
  % xlabel('s'); ylabel('Hz'); zlabel('Correlation');
  % title(sprintf('Satellite %i',satNum));
  % Vue face au doppler
  % view(90,0);
endif

% On ne conserve que Np points autour du pic
corrMod_c = corrMod(:,indT - Np + 1:indT + Np);
corrModCos_c = corrModCos(:,indT - Np + 1:indT + Np);
corrModSin_c = corrModSin(:,indT - Np + 1:indT + Np);

if (strcmp(choice,"noise") == 1)
  save -text "correlator_output_noise.txt" corrModCos_c corrModSin_c;
  satNum = sauv_satNum;
endif

clear corrMod corrModCos corrModSin;

% Normalisation du pic � 1
[corrMax, indDop] = max(abs(corrMod_c));
[corrMaxMax,indT] = max(corrMax);
corrMod_c = corrMod_c/corrMaxMax;
clear corrMax corrMaxMax;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute maximum indT for corrModcompl
corrModcompl_t = corrModcompl(:,indT - Np + 1:indT + Np);
[corrMax_compl, indDop_compl] = max(abs(corrModcompl_t));
[corrMaxMax_compl,indT_compl] = max(corrMax_compl);

corr_noise = fftshift(ifft2(fft2(corrModcompl_crop) .* conj(fft2(corrModcompl_crop))));

switch choice_fake
  case('fake_noise')
    'CHECK CROP FOR FAKE NOISE'
    corrModcompl_crop = corrModcompl_t(:,size(corrModcompl_t)(2) / 2 - Np + 1:size(corrModcompl_t)(2) / 2 + Np);
  otherwise
    corrModcompl_crop = corrModcompl_t(:,indT_compl - Np + 1:indT_compl + Np);
endswitch


if interactive == true
  figure();
  t_mesh = [-Np + 1:Np]/Fs_interp;
  [x,y] = meshgrid(t_mesh,fDop + LO_offset - eps_f);
  mesh(x,y, real(corrModcompl_crop));
  xlabel('s'); ylabel('Hz'); zlabel('Correlation');
  title(sprintf('Satellite %i. I channel',satNum));
  
  figure();
  t_mesh = [-Np + 1:Np]/Fs_interp;
  [x,y] = meshgrid(t_mesh,fDop + LO_offset - eps_f);
  mesh(x,y, imag(corrModcompl_crop));
  xlabel('s'); ylabel('Hz'); zlabel('Correlation');
  title(sprintf('Satellite %i. Q channel',satNum));
  
  figure();
  t_mesh = [-Np + 1:Np]/Fs_interp;
  [x,y] = meshgrid(t_mesh,fDop + LO_offset - eps_f);
  mesh(x,y, abs(corrModcompl_crop));
  xlabel('s'); ylabel('Hz'); zlabel('Correlation');
  title(sprintf('Satellite %i. Module',satNum));
  
  figure();
  t_mesh = [-Np + 1:Np]/Fs_interp;
  [x,y] = meshgrid(t_mesh,fDop + LO_offset - eps_f);
  mesh(x,y, real(corr_noise));
  xlabel('s'); ylabel('Hz'); zlabel('Correlation');
  title(sprintf('Satellite %i. Module',satNum));
  % Vue face au doppler
  % view(90,0);
  
  figure("name","Correlation au pic en fonction du d�lai",...
  "numbertitle","off");
  subplot(1,3,1)
  plot(t_mesh,corrModCos_c(indDop(indT),:));  grid on;
  xlabel('s'); title('Correlation voie I');
  subplot(1,3,2)
  plot(t_mesh,corrModSin_c(indDop(indT),:));  grid on;
  xlabel('s'); title('Correlation voie Q');
  subplot(1,3,3)
  plot(t_mesh,corrMod_c(indDop(indT),:));  grid on;
  xlabel('s'); title('Somme des correlations I^2 + Q^2');
endif
clear corrMod_c corrModCos_c corrModSin_c;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WRITE DATA INTO .CSV FILE
% get real part of complex number for voieI
write_csv = 1

'CHECK I_CHANNEL OPTION'
corr_out = real(corrModcompl_crop);
size(corr_out)
if write_csv == 1
  file_path = strcat('outputs/i_channel/corrModcompl_crop_i_',datestr(now, 'yyyy_MM_dd_HH_mm_ss'),'.csv')
  csvwrite(file_path, corr_out);
endif

'CHECK Q_CHANNEL OPTION'
corr_out = imag(corrModcompl_crop);
'Check matrix size: ' 
size(corr_out)
if write_csv == 1
  file_path = strcat('outputs/q_channel/corrModcompl_crop_q_',datestr(now, 'yyyy_MM_dd_HH_mm_ss'),'.csv')
  csvwrite(file_path, corr_out);
endif

