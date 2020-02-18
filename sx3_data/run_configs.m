% Define intervals
clear all; %close all;

for satNum = [8, 11]
  
  %----PRN CONSTANTS ----------------------------------------------%
  if satNum == 1
    eps_c = 0.332e-3; eps_f = 1750;
  elseif satNum == 8 
    eps_c = 0.889e-3; eps_f = -2500;
  elseif satNum == 11
    eps_c = 0.840e-3; eps_f = 250;
  endif
  
  %----LES CONSTANTES GPS--------------------------------------------%
  FL1 = 1575.42e6;
  Fc = 1.023e6; Tc = 1/Fc;
  Nc = 1023;
  Fs = 20e6
  %----AXIS ON DOPPLER/ CODE--------------------------------------------%
  % Nombre de p�riodes de code d'int�gration coh�rente
  % La convolution se fera sur N + 1 p�riodes
  N = 1;
  Ti = N * 10^-3;
  % Nombre de sommations non-coh�rentes
  M = 1;
  % L'excursion Doppler maximale
  dopp_max = min(5.5/Ti, 800+2.5/Ti);
  % Le pas de recherche du Doppler
  deltaDop = 100;
  dopp_interv = [-dopp_max:deltaDop:dopp_max];

  % chip period
  % L'excursion maximale en d�lai qui sera visualis�e autour du pic :
  % Le nombre de points de correlation sur [0, tau_max]
  % Np = 100;
  % L'excursion maximale en d�lai qui sera visualis�e autour du pic :
  %tau_max = 2*Tc;    % Donc +- 2 chips autour du pic
  tau_max_left = Tc
  tau_max_right = 2.5*Tc
  % Le code PRN local
  code_PRN1 = cacodeAB(satNum,Fc,Fs);
  lC = length(code_PRN1);
  % Le nombre de points de correlation sur [0, tau_max]
  Np_left = ceil(tau_max_left*lC/(Nc*Tc));
  Np_right = ceil(tau_max_right*lC/(Nc*Tc));


  %----CONFIG EXECUTION MODE--------------------------------------------%
  interactive = true, choice = 4, write_csv = 1
  partition_step = 400000, partition_max = 1600000
  
  %----RUN--------------------------------------------%
  'CHECK satNum CONFIG'
  satNum
  test_acquisition_GPS_sx3_corr_modif
 endfor
