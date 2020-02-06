% Zoom 2-D sur le pic de correlation.

% Attention, on peut toujours tomber sur une partie du signal où
% il y a un changement de bit d'information. Donc le niveau de 
% correlation ne sera pas optimal dans ce cas là si N > 1.

% Mode non-interactif, par exemple :
% interactive = false;
% path = './Data/';
% file = 'pluto_2084kS_2MHz_71_dB_GPS_04112019_1306.cfile'
% file_name = [path, file];
% Fs = 2.084e6; Fif = 0; LO_offset = -10e3; sign_Q = +1; file_format = 'cfile';
% if_signal = false;
% satNum = 6; eps_c = 0.130e-3; eps_f = -500;
% zoom_acquisition

% Mode interactif
% interactive = true, choice = 4, zoom_acquisition_no_interp_v2

% Note 1 : on prend N + 1 périodes d'intégration cohérente du signal, on calcule
% la correlation avec un signal local qui en fait seulement N, puis on 
% recommence avec les N + 1 périodes suivantes du signal. Soit un découpage du 
% signal en tranches de N + 1, alors qu'on corrèle seulement sur N. Il serait 
% plus intelligent de découper en tranches de N, avec recouvrement d'une 
% période.

% Note 2 : c'est seulement tout à la fin, en dehors de la boucle, qu'on ne 
% conserve la correlation que sur N périodes complètes. Il serait peut être 
% plus efficace de le faire au fur et à mesure, dans la boucle.

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
  choice = 0;
endif

addpath './Sub_Functions/';

%----LES CONSTANTES GPS--------------------------------------------%

FL1 = 1575.42e6;
Fc = 1.023e6; Tc = 1/Fc;
Nc = 1023;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AXIS ON DOPPLER/ CODE
% L'excursion Doppler maximale
dopp_max = 1000;
% Le pas en fréquence, en Hz
dopp_step = 10;

% L'excursion maximale en délai qui sera visualisée autour du pic :
tau_max = 2*Tc;    % Donc +- 2 chips autour du pic
% Le nombre de points de correlation sur [0, tau_max]
Np = 100

if (interactive == true)
  switch choice
    case {1}
      if_signal = true;
      path = 'C:\Users\blaisan\Downloads\'
      % path = 'C:\Users\blais\Downloads\';
      % path = '~/Téléchargements/';
      file = 'test_real_long.dat'
      file_name = [path,file];
      Fs = 23.104e6; Fif = 4.348e6; LO_offset = 0; sign_Q = 0;
      file_format = 'old';
      
      % 3, -4250 Hz, 0.165 ms, 27 dB/Hz : B_P = 100 Hz
      % 11, -250 Hz, 0.916 ms, 32 dB/Hz : B_P = 100 Hz
      % 14, -2250 Hz, 0.233 ms, 27 dB/Hz : B_P = 100 Hz
      % 17, 2750 Hz, 0.279 ms, 30 dB/Hz : B_P = 100 Hz
      % 19, -3500 Hz, 0.138 ms, 32 dB/Hz : B_P = 100 Hz
      % 20, 1750 Hz, 0.387 ms, 33 dB/Hz : B_P = 10 Hz

      % satNum = 3; B_P = 100; eps_c = 0.165e-3; eps_f = -4250;
      % satNum = 11; B_P = 100; eps_c = 0.916e-3; eps_f = -250;
      % satNum = 14; B_P = 100; eps_c = 0.233e-3; eps_f = -2250;
      % satNum = 17; B_P = 100; eps_c = 0.279e-3; eps_f = 2750
      satNum = 19; B_P = 100; eps_c = 0.138e-3; eps_f = -3500;
      % satNum = 20; B_P = 10; eps_c = 0.387e-3; eps_f = 1750;
    case {2}
      if_signal = true;
      path = './Data/';
      file = 'SwRxData_Band0_FE0_ANT0_f1575420000.stream' % test file
      file_name = [path,file];
      Fs = 20e6; Fif = 5000445.89; LO_offset = 0; sign_Q = +1;
      file_format = 'stream';
      
      satNum = 4; eps_c = 0.117e-3; eps_f = -1500;
      % satNum = 27; eps_c = 0.651e-3; eps_f = -250;
    case{3}
      if_signal = false;
      path = './Data/';
      file = 'pluto_2084kS_2MHz_71_dB_GPS_04112019_1306.cfile'
      file_name = [path, file];
      Fs = 2.084e6; Fif = 0; LO_offset = -10e3; sign_Q = +1;
      file_format = 'cfile';

      satNum =  6; eps_c = 0.129e-3; eps_f = -500;
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
      if_signal = false;
      if (if_signal == true)
        Fs = 23.104e6; Fif = 4.348e6; LO_offset = 0; sign_Q = 0;
        file_format = 'debug';
        
        satNum = 1; eps_c = 0.4e-3; eps_f = 1800;
      else
        Fs = 20e6; Fif = 0; LO_offset = 0; sign_Q = +1;
        file_format = 'debug';
        
        satNum = 1; eps_c = 0.4e-3; eps_f = 1800;
      endif
    case {4}
      if_signal = true;
      path = './Data/';
      file = 'SwRxData-191113145928_Band0_FE0_ANT0_f1575420000.stream' # no MP 
      file_name = [path,file];
      Fs = 20e6; Fif = 5000445.89; LO_offset = 0; sign_Q = +1;
      file_format = 'stream';
      
      satNum = 1; eps_c = 0.332e-3; eps_f = 1750;
      %satNum = 8; eps_c = 0.889e-3; eps_f = -2500;
      %satNum = 11; eps_c = 0.840e-3; eps_f = 250;
      %satNum = 17; eps_c = 0.369e-3; eps_f = 2500;
      %satNum = 18; eps_c = 0.004e-3; eps_f = -250;
      %satNum = 22; eps_c = 0.715e-3; eps_f = 2750;
      %satNum = 27; eps_c = 0.469e-3; eps_f = -3500;
      %satNum = 30; eps_c = 0.064e-3; eps_f = -750;
      
    case {5}
      if_signal = true;
      path = './Data/';
      file = 'SwRxData-191113160022_Band0_FE0_ANT0_f1575420000.stream' # no MP 
      file_name = [path,file];
      Fs = 20e6; Fif = 5000445.89; LO_offset = 0; sign_Q = +1;
      file_format = 'stream';
      
      satNum = 28; eps_c = 0.131e-3; eps_f = 3500;
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

% Nombre de périodes de code d'intégration cohérente
% La convolution se fera sur N + 1 périodes
N = 2

% Nombre de sommations non-cohérentes
M = 6

%----LES DONNEES---------------------------------------------------%

% Number of samples of the signal we need
% Pour une correlation avec N périodes de code, il faut prendre N + 1
% périodes de signal pour être sûr d'avoir une superposition de N
% périodes complètes de code.
Ns = (N + 1)*M*Nc/Fc*Fs;

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
    % Code périodisé (N + 1)*M fois
    code_debug = kron(ones(1,(N + 1)*M),code_PRN1);
    codeDelai = [code_debug(end - eps_c + 1:end) code_debug(1:end - eps_c)];
    tz = [0:length(codeDelai) - 1]/Fs;
    voieI = codeDelai.*cos(2*pi*(Fif + eps_f + LO_offset)*tz + pi/3) + ...
      + 0.0*randn(1,length(tz));
    if (if_signal == true)
      voieQ = zeros(1,length(tz));
    else
      voieQ = codeDelai.*sin(2*pi*(Fif + eps_f + LO_offset)*tz + pi/3) + ...
        + 0.0*randn(1,length(tz));
    endif
    clear tz code_PRN1 code_debug codeDelai;
  otherwise

endswitch

voieQ = sign_Q*voieQ;
voieI = voieI - mean(voieI);
voieQ = voieQ - mean(voieQ);

% Le code PRN local
code_PRN1 = cacodeAB(satNum,Fc,Fs);
lC = length(code_PRN1);

% We center the maximum in the final plot
if eps_c > Nc/2.0
  delai = round(eps_c - Nc/2.0);
else
  delai = round(eps_c + Nc/2.0);
endif

code_PRN1 = [code_PRN1(end - delai + 1:end) code_PRN1(1:end - delai)];

% Code périodisé sur une période d'intégration cohérente
code_PRN1_N = kron(ones(1,N),code_PRN1);
lC_N = lC*N;
clear code_PRN1;

lI = length(voieI);
% plot(voieI,'b'); pause;
% clf; plot(abs(fft(voieI)),'b'); pause;

%----L'ACQUISITION------------------------------------------------%

% On résoud le problème de la convolution circulaire :
% La longueur du signal sur une période d'intégration cohérente : N + 1.
% On va lui rajouter N + 1 périodes de zéros et réaliser la convolution
% circulaire sur 2*(N+1) points donc.
lC_N_p1 = lC*(N + 1);
lC_2N_p2 = 2*lC_N_p1;
lC_N_p2 = lC*(N + 2);
% On met le code local à la même longueur avec des zéros également.
code_PRN1_2N_p2 = [code_PRN1_N zeros(1,lC_N_p2)];
clear code_PRN1_N;

% Le temps qui correspond
t = [0:lC_2N_p2 - 1]/Fs;

% Je veux Np échantillons sur [0, tau_max], c'est à dire Np*(Nc*Tc)/tau_max sur
% une période de code. Avec Fs j'en ai lC. Je dois donc interpoler d'un facteur
f_interp = Np*(Nc*Tc)/tau_max/lC; f_interp = 0.9
if (f_interp <= 1)
  fprintf(1,"Interpolation factor <= 1. Set to 1.\n");
  Np = ceil(tau_max*lC/(Nc*Tc));
  f_interp = 1; N_interp = lC_2N_p2; Fs_interp = Fs;
  lC_i = lC;
else
  fprintf(1,'Interpolation factor = %f\n',f_interp);
  lC_i = round(Np*(Nc*Tc)/tau_max);
  N_interp = 2*(N + 1)*lC_i;
  Fs_interp = Fs/lC*lC_i;
endif
t_interp = [0:N_interp - 1]/Fs_interp;


lC_N_i = lC_i*N;
lC_N_p1_i = lC_i*(N + 1);
lC_2N_p2_i = 2*lC_N_p1_i;
lC_N_p2_i = lC_i*(N + 2);

% Le nombre de zéros pour padder le spectre, à gauche et à droite, 
% pour interpoler le signal temporel.
N_pad = N_interp - lC_2N_p2;
N_pad_left = round(N_pad/2); N_pad_right = N_pad - N_pad_left;

% Les fréquences de la réplique locale
fDop = [-dopp_max:dopp_step:dopp_max] - LO_offset + eps_f;
lDop = length(fDop);

co = zeros(1,lC_2N_p2);
si = zeros(1,lC_2N_p2);
fft_PRN1I = zeros(1,lC_2N_p2);
fft_PRN1Q = zeros(1,lC_2N_p2);
fftI = zeros(1,lC_2N_p2);
fftProdIco = zeros(1,lC_2N_p2);
fftProdIsi = zeros(1,lC_2N_p2);
corrIco = zeros(1,N_interp);
corrIsi = zeros(1,N_interp);

if (if_signal == false)
  fftQ = zeros(1,lC_2N_p2);
  fftProdQco = zeros(1,lC_2N_p2);
  fftProdQsi = zeros(1,lC_2N_p2);
  corrQsi = zeros(1,N_interp);
  corrQco = zeros(1,N_interp);
  corrModCos = zeros(lDop,N_interp);
  corrModSin = zeros(lDop,N_interp);
endif
corrMod = zeros(lDop,N_interp);

% Les zéros pour padder le spectre, à gauche et à droite.
z_left = zeros(1,N_pad_left);
z_right = zeros(1,N_pad_right);

h = waitbar(0,'Acquisition en cours');

% Frequency grid
for i = [1:lDop]

  waitbar(i/lDop,h);

  % Les répliques locales I (co) et Q (si) sont implicitement mises à zéros
  % sur les N + 2 dernières périodes par code_PRN1_2N_p2
  co = code_PRN1_2N_p2.*cos(2*pi*(Fif + fDop(i))*t);
  si = code_PRN1_2N_p2.*sin(2*pi*(Fif + fDop(i))*t);
  fft_PRN1I = fft(co);
  fft_PRN1Q = fft(si);

  % Non-coherent summations
  for k = [1:M]
    
    % Le signal est complèté avec N + 1 périodes de zéros
    fftI = fft([voieI(k*lC_N_p1:-1:1 + (k - 1)*lC_N_p1) zeros(1,lC_N_p1)]);
    if (if_signal == false)
      fftQ = fft([voieQ(k*lC_N_p1:-1:1 + (k - 1)*lC_N_p1) zeros(1,lC_N_p1)]);
    endif
    
    % Il faut remettre l'abscisse 0 de la correlation au centre du vecteur et
    % ensuite padder à droite et à gauche avec le même nombre de zéros pour
    % conserver la symétrie du spectre, sinon le signal temporel n'est pas 
    % réel après l'ifft.
    fftProdIco = [z_left fftshift(fft_PRN1I.*fftI) z_right];
    fftProdIsi = [z_left fftshift(fft_PRN1Q.*fftI) z_right];
    if (if_signal == false)
      fftProdQco = [z_left fftshift(fft_PRN1I.*fftQ) z_right];
      fftProdQsi = [z_left fftshift(fft_PRN1Q.*fftQ) z_right];
    else
      
    endif

    % ifft(fftProdIco,N_interp) padde fftProdIco avec des zéros à la fin du 
    % spectre, ce qui rompt la symétrie et donne un signal complexe en temporel.
    corrIco = real(ifft(fftProdIco));
    corrIsi = real(ifft(fftProdIsi));
    if (if_signal == false)
      corrQco = real(ifft(fftProdQco));
      corrQsi = real(ifft(fftProdQsi));
    endif

    if (if_signal == false)
      corrModCos(i,:) = (corrIco + corrQsi);
      corrModSin(i,:) = (corrQco - corrIsi);
      % corrMod(i,:) += corrModCos(i,:).^2 + corrModSin(i,:).^2;
      % Pour mieux voir les lobes secondaires :
      corrMod(i,:) += sqrt(corrModCos(i,:).^2 + corrModSin(i,:).^2);
    else
      % corrMod(i,:) += corrIco.^2 + corrIsi.^2;
      corrMod(i,:) += sqrt(corrIco.^2 + corrIsi.^2);
    endif
    
  end
end

close(h);

clear code_PRN1_2N_p2 voieI voieQ;
clear co si fft_PRN1I fft_PRN1Q fftI fftQ;
clear fftProdIco fftProdIsi fftProdQco fftProdQsi;
clear corrIco corrQsi corrIsi corrQco;

corrShift = corrMod(:,end:-1:1);
if (if_signal == false)
  corrModCos = corrModCos(:,end:-1:1);
  corrModSin = corrModSin(:,end:-1:1);
endif
clear corrMod;

% On ne conserve que la partie correspondant à la correlation sur N périodes
% complètes :
corrShift_t = corrShift(:,lC_N_p1_i:lC_N_p1_i + lC_N_i - 1);
if (if_signal == false)
  corrModCos_t = corrModCos(:,lC_N_p1_i:lC_N_p1_i + lC_N_i - 1);
  corrModSin_t = corrModSin(:,lC_N_p1_i:lC_N_p1_i + lC_N_i - 1);
endif
clear corrModCos corrModSin;

[corrMax, indDop] = max(abs(corrShift_t));
[corrMaxMax,indT] = max(corrMax);
fprintf(1,'Retard = %f ms\n',t_interp(indT)/1e-3-floor(t_interp(indT)/1e-3));
fprintf(1,'Doppler = %f Hz\n',fDop(indDop(indT)) + LO_offset);
fprintf(1,'Niveau Max de correlation = %f\n',corrMaxMax);
clear corrMax corrMaxMax;

if interactive == true
  figure("name","Correlation au pic en fonction de l'écart de fréquence",...
  "numbertitle","off");
  if (if_signal == false)
    subplot(1,4,1)
    plot(fDop,corrModCos_t(:,indT)); grid on;
    xlabel('Hz'); title('Correlation voie I');
    subplot(1,4,2)
    plot(fDop,corrModSin_t(:,indT)); grid on;
    xlabel('Hz'); title('Correlation voie Q');
    subplot(1,4,3)
    plot(fDop,corrShift_t(:,indT)); grid on;
    xlabel('Hz'); title('Somme des correlations I^2 + Q^2');
    subplot(1,4,4)
    t_plot = [0:lC_N_i - 1]/Fs_interp;
    plot(t_plot,corrShift_t(indDop(indT),:)); grid on;
    xlabel('s'); title('Somme des correlations I^2 + Q^2');
  else
    subplot(1,2,1)
    plot(fDop,corrShift_t(:,indT)); grid on;
    xlabel('Hz'); title('Somme des correlations I^2 + Q^2');
    subplot(1,2,2)
    t_plot = [0:lC_N_i - 1]/Fs_interp;
    plot(t_plot,corrShift_t(indDop(indT),:)); grid on;
    xlabel('s'); title('Somme des correlations I^2 + Q^2');
  endif
  
  % figure();
  % [x,y] = meshgrid(t,fDop - LO_offset);
  % mesh(x,y,corrShift);
  % xlabel('s'); ylabel('Hz'); zlabel('Correlation');
  % title(sprintf('Satellite %i',satNum));
  % figure();
  % [x,y] = meshgrid(t_plot,fDop + LO_offset - eps_f);
  % mesh(x,y,corrShift_t);
  % xlabel('s'); ylabel('Hz'); zlabel('Correlation');
  % title(sprintf('Satellite %i',satNum));
  % Vue face au doppler
  % view(90,0)
endif

clear t corrShift

% On ne conserve que Np points autour du pic
corrShift_c = corrShift_t(:,indT - Np + 1:indT + Np);
if (if_signal == false)
  corrModCos_c = corrModCos_t(:,indT - Np + 1:indT + Np);
  corrModSin_c = corrModSin_t(:,indT - Np + 1:indT + Np);
endif

clear corrShift_t corrModCos_t corrModSin_t;

% Normalisation du pic à 1
[corrMax, indDop] = max(abs(corrShift_c));
[corrMaxMax,indT] = max(corrMax);
corrShift_c = corrShift_c/corrMaxMax;
clear corrMax corrMaxMax;

if interactive == true
  figure();
  t_mesh = [-Np + 1:Np]/Fs_interp;
  [x,y] = meshgrid(t_mesh,fDop + LO_offset - eps_f);
  mesh(x,y,corrShift_c);
  xlabel('s'); ylabel('Hz'); zlabel('Correlation');
  title(sprintf('Satellite %i',satNum));
  % Vue face au doppler
  % view(90,0)
  figure();
  plot(t_mesh,corrShift_c(indDop(indT),:));  grid on;
  xlabel('s'); title('Somme des correlations I^2 + Q^2');
endif

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WRITE DATA INTO .CSV FILE
corr_out = corrShift_c;
write_csv = 0;
if write_csv
  csvwrite('corr_out_191113145928_prn4_no_interp_v2.csv', corr_out);
endif

%clear corrShift_c corrModCos_c corrModSin_c;
