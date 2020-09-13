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
% for satNum = [1:32], satNum, test_acquisition_GPS, end

%close all;

% Mode interactif
%interactive = true, choice = 4
%test_acquisition_GPS_fake_noise_v3_corr_modif

% Note 1 : on prend N + 1 p�riodes d'int�gration coh�rente du signal, on calcule
% la correlation avec un signal local qui en fait seulement N, puis on 
% recommence avec les N + 1 p�riodes suivantes du signal. Soit un d�coupage du 
% signal en tranches de N + 1, alors qu'on corr�le seulement sur N. Il serait 
% plus intelligent de d�couper en tranches de N, avec recouvrement d'une 
% p�riode.

% Note 2 : c'est seulement tout � la fin, en dehors de la boucle, qu'on ne 
% conserve la correlation que sur N p�riodes compl�tes. Il serait peut �tre 
% plus efficace de le faire au fur et � mesure, dans la boucle.

% Mode interactif ou non ?
if exist("interactive","var") != 1 || (interactive == true)
  interactive
  choice
  'CHECK INTERACTIVE NOT EXIST'
  %clear -x choice;
  %clear -x choice_fake;
  interactive = true;
  if exist("choice","var") != 1
    choice = 1;
  endif
else
  clear -x file_name Fs Fif LO_offset sign_Q file_format if_signal satNum;
  interactive = false;
  choice = 0;
endif

%addpath './Sub_Functions/';

if (interactive == true)
  switch choice
    case {1}
      if_signal = true;
      path = 'C:\Users\blaisan\Downloads\'
      % path = 'C:\Users\blais\Downloads\';
      % path = '~/T�l�chargements/';
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
      file = 'SwRxData_Band0_FE0_ANT0_f1575420000.stream'
      file_name = [path,file];
      Fs = 20e6; Fif = 5000445.89; LO_offset = 0; sign_Q = +1;
      file_format = 'stream';
      
      % satNum = 4;
      satNum = 27;
      case {3}
      if_signal = false;
      path = './Data/';
      file = 'pluto_2084kS_2MHz_71_dB_GPS_04112019_1306.cfile'
      file_name = [path, file];
      Fs = 2.084e6; Fif = 0; LO_offset = -10e3; sign_Q = +1;
      file_format = 'cfile';
      
      satNum =  6;
    case {'test'}
      if_signal = false;
      path = './Data/';
      file_I = 'voieI_121208_16H35.dat';
      file_Q = 'voieQ_121208_16H35.dat';
      file_name_I = [path,file_I];
      file_name_Q = [path,file_Q];
      Fs = 10e6; Fif = 0; LO_offset = 0; sign_Q = +1; file_format = 'dat';
      
      % satNum = 3, 6, 11, ...
      satNum = 11;
    case {'debug'}
      'CHECK DEBUG CASE'
      if_signal = true;
      if (if_signal == true)
        'CHECK IF SIGNAL TRUE'
        Fs = 23.104e6; Fif = 4.348e6; LO_offset = 0; sign_Q = 0;
        file_format = 'debug';
        
        satNum = 1; eps_c = 0.4e-3; eps_f = 1800;
      else
        Fs = 20e6; Fif = 0; LO_offset = 0; sign_Q = +1;
        file_format = 'debug';
        
        satNum = 1; eps_c = 0.4e-3; eps_f = 1800;
      endif
      % Code delay, in samples
      eps_c = round(eps_c*Fs);
    case {4}
      'CHECK CASE 4 STREAM'
      'CHECK SAT PRN'
      satNum
      if_signal = true;
      path = './Data/';
      file = 'SwRxData-191113145928_Band0_FE0_ANT0_f1575420000.stream' # no MP 
      file_name = [path,file];
      Fs = 20e6; Fif = 5000445.89; LO_offset = 0; sign_Q = +1;
      file_format = 'stream';
      
    case {5}
      if_signal = true;
      path = './Data/';
      file = 'SwRxData-191113160022_Band0_FE0_ANT0_f1575420000.stream' # no MP 
      file_name = [path,file];
      Fs = 20e6; Fif = 5000445.89; LO_offset = 0; sign_Q = +1;
      file_format = 'stream';
      satNum =  28;
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
endif
    
% Sampling period
Ts = 1/Fs;

%----LES DONNEES---------------------------------------------------%

% Number of samples of the signal we need
% Pour une correlation avec N p�riodes de code, il faut prendre N + 1
% p�riodes de signal pour �tre s�r d'avoir une superposition de N
% p�riodes compl�tes de code.
Ns = (N + 1)*M*Nc/Fc*Fs;

%file_list = ['SwRxData-191113145928_Band0_FE0_ANT0_f1575420000.stream';
%             'SwRxData-191113151614_Band0_FE0_ANT0_f1575420000.stream';
%             'SwRxData-191113154206_Band0_FE0_ANT0_f1575420000.stream'];

file_list = ['SwRxData-191113145928_Band0_FE0_ANT0_f1575420000.stream'];
             
             
% parse through .stream files
for file_number = [1:size(file_list)(1)]
	file = file_list(file_number, :)
	
	% partition .stream file
	for start_time = [1: partition_step: partition_max]
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
		    %voieI = reader_IFEN_SX3_AB(file_name,Ns); voieI = voieI.';
        voieI = reader_IFEN_SX3_AB_partition(file_name,start_time,start_time + Ns); voieI = voieI.';
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
		    % Code p�riodis� (N + 1)*M fois
		    code_debug = kron(ones(1,(N + 1)*M),code_PRN1);
		    codeDelai = [code_debug(end - eps_c + 1:end) code_debug(1:end - eps_c)];
		    tz = [0:length(codeDelai) - 1]/Fs;
		    
		    % choose between fake signal and fake noise
		    choice_fake = 'fake_noise'
		    switch choice_fake
		      case('fake_signal')
			voieI = codeDelai.*cos(2*pi*(Fif + eps_f + LO_offset)*tz + pi/3) + 0.0*randn(1,length(tz));
			if (if_signal == true)
			  voieQ = zeros(1,length(tz));
			else
			  voieQ = codeDelai.*sin(2*pi*(Fif + eps_f + LO_offset)*tz + pi/3) + 0.0*randn(1,length(tz));
			endif
		      case('fake_noise')
			'CHECK FAKE NOISE OPTION, NOISE GENER'
			voieI = 1.0*randn(1,length(tz));
			if (if_signal == true)
			  voieQ = zeros(1,length(tz));
			else
			  voieQ = 1.0*randn(1,length(tz));
			endif
		      otherwise
		    endswitch
		  % clear tz code_debug codeDelai;
		  otherwise
		endswitch

		voieQ = sign_Q*voieQ;
		voieI = voieI - mean(voieI);
		voieQ = voieQ - mean(voieQ);

		% Code p�riodis� sur une p�riode d'int�gration coh�rente
		code_PRN1_N = kron(ones(1,N),code_PRN1);
		lC_N = lC*N;

		lI = length(voieI);
		% plot(voieI,'b'); pause;
		% clf; plot(abs(fft(voieI)),'b'); pause;

		%----L'ACQUISITION------------------------------------------------%

		% On r�soud le probl�me de la convolution circulaire :
		% La longueur du signal sur une p�riode d'int�gration coh�rente : N + 1.
		% On va lui rajouter N + 1 p�riodes de z�ros et r�aliser la convolution
		% circulaire sur 2*(N+1) points donc.
		lC_N_p1 = lC*(N + 1);
		lC_2N_p2 = 2*lC_N_p1;
		lC_N_p2 = lC*(N + 2);
		% On met le code local � la m�me longueur avec des z�ros �galement.
		code_PRN1_2N_p2 = [code_PRN1_N zeros(1,lC_N_p2)];
		clear code_PRN1_N;

		% Le temps qui correspond
		t = [0:lC_2N_p2 - 1]/Fs;

    % --------------------------------------------------------------------------
    % Define intervals
		% Les fr�quences de la r�plique locale
		fDop = dopp_interv - LO_offset + eps_f;
		lDop = length(fDop);

		co = zeros(1,lC_2N_p2);
		si = zeros(1,lC_2N_p2);
		fft_PRN1I = zeros(1,lC_2N_p2);
		fft_PRN1Q = zeros(1,lC_2N_p2);
		fftI = zeros(1,lC_2N_p2);
		fftProdIco = zeros(1,lC_2N_p2);
		fftProdIsi = zeros(1,lC_2N_p2);
		corrIco = zeros(1,lC_2N_p2);
		corrIsi = zeros(1,lC_2N_p2);

		if (if_signal == false)
		  fftQ = zeros(1,lC_2N_p2);
		  fftProdQco = zeros(1,lC_2N_p2);
		  fftProdQsi = zeros(1,lC_2N_p2);
		  corrQsi = zeros(1,lC_2N_p2);
		  corrQco = zeros(1,lC_2N_p2);
		  corrModCos = zeros(lDop,lC_2N_p2);
		  corrModSin = zeros(lDop,lC_2N_p2);
		endif
		corrMod = zeros(lDop,lC_2N_p2);
		corrModcompl = zeros(lDop, lC_2N_p2);

		h = waitbar(0,'Acquisition en cours');

		% Search for Doppler shift
		for i = [1:lDop]

		  waitbar(i/lDop,h);

		  % Les r�pliques locales I (co) et Q (si) sont implicitement mises � z�ros
		  % sur les N + 2 derni�res p�riodes par code_PRN1_2N_p2
		  co = code_PRN1_2N_p2.*cos(2*pi*(Fif + fDop(i))*t);
		  si = code_PRN1_2N_p2.*sin(2*pi*(Fif + fDop(i))*t);
		  fft_PRN1I = fft(co);
		  fft_PRN1Q = fft(si);

		  % Non-coherent summations
		  for k = [1:M]
		    
		    % Le signal est compl�t� avec N + 1 p�riodes de z�ros
		    fftI = fft([voieI(k*lC_N_p1:-1:1 + (k - 1)*lC_N_p1) zeros(1,lC_N_p1)]);
		    if (if_signal == false)
		      fftQ = fft([voieQ(k*lC_N_p1:-1:1 + (k - 1)*lC_N_p1) zeros(1,lC_N_p1)]);
		    endif
		    
		    fftProdIco = fft_PRN1I.*fftI;
		    fftProdIsi = fft_PRN1Q.*fftI;
		    if (if_signal == false)
		      fftProdQco = fft_PRN1I.*fftQ;
		      fftProdQsi = fft_PRN1Q.*fftQ;z
		    endif

		    corrIco = real(ifft(fftProdIco));
		    corrIsi = real(ifft(fftProdIsi));
		    if (if_signal == false)
		      corrQco = real(ifft(fftProdQco));
		      corrQsi = real(ifft(fftProdQsi));
		    endif

		    if (if_signal == false)
		      corrModCos(i,:) = (corrIco + corrQsi);
		      corrModSin(i,:) = (corrQco - corrIsi);
		      corrMod(i,:) += corrModCos(i,:).^2 + corrModSin(i,:).^2;
		      % Pour mieux voir les lobes secondaires :
		      % corrMod(i,:) += sqrt(corrModCos(i,:).^2 + corrModSin(i,:).^2);
		      corrModcompl(i,:) += corrModcos(i,:) + j * corrModsin(i,:);
		    else
		      %'CHECK LOOP IF SIGNAL'
		      corrModCos(i,:) = corrIco;
		      corrModSin(i,:) = -corrIsi;
		      corrMod(i,:) += corrIco.^2 + corrIsi.^2;
		      %corrMod(i,:) += sqrt(corrIco.^2 + corrIsi.^2);
		      corrModcompl(i,:) += corrModCos(i,:) + j * corrModSin(i,:);
		    endif
		  end
		end

		close(h);

		clear code_PRN1_2N_p2 voieI voieQ;
		clear co si fft_PRN1I fft_PRN1Q fftI fftQ;
		clear fftProdIco fftProdIsi fftProdQco fftProdQsi;
		clear corrIco corrQsi corrIsi corrQco;

		corrShift = corrMod(:,end:-1:1);
		corrModcompl = corrModcompl(:,end:-1:1);
		
		if (if_signal == false)
		  corrModCos = corrModCos(:,end:-1:1);
		  corrModSin = corrModSin(:,end:-1:1);
		endif
		clear corrMod;

		% On ne conserve que la partie correspondant � la correlation sur N p�riodes
		% compl�tes :
		corrShift_t = corrShift(:,lC_N_p1:lC_N_p1 + lC_N - 1);
		if (if_signal == false)
		  corrModCos_t = corrModCos(:,lC_N_p1:lC_N_p1 + lC_N - 1);
		  corrModSin_t = corrModSin(:,lC_N_p1:lC_N_p1 + lC_N - 1);
		endif
		clear corrModCos corrModSin;

		[corrMax, indDop] = max(abs(corrShift_t));
		[corrMaxMax,indT] = max(corrMax);
		fprintf(1,'Retard = %f ms\n',t(indT)/1e-3-floor(t(indT)/1e-3));
		fprintf(1,'Doppler = %f Hz\n',fDop(indDop(indT)) + LO_offset);
		fprintf(1,'Niveau Max de correlation = %f\n',corrMaxMax);
    
    % compute corrShift_c for output in .csv
		code_interv = indT-Np_left+1: indT+Np_right;
    corrShift_c = corrShift_t(:, code_interv);		
		corrShift_c = corrShift_c / corrMaxMax;		
    
		%z = xcorr(corrModcompl_crop);


		clear corrMax corrMaxMax;

		if interactive == true
		  %figure("name","Correlation au pic en fonction de l'�cart de fr�quence",...
		  %"numbertitle","off");
		  if (if_signal == false)
		    %subplot(1,4,1)
		    %plot(fDop,z(:,indT)); grid on;
		    %xlabel('Hz'); title('Correlation voie I');
		    %subplot(1,4,2)
		    %plot(fDop,corrModSin_t(:,indT)); grid on;
		    %xlabel('Hz'); title('Correlation voie Q');
		    %subplot(1,4,3)
		    %plot(fDop,corrShift_t(:,indT)); grid on;
		    %xlabel('Hz'); title('Somme des correlations I^2 + Q^2');
		    %subplot(1,4,4)
		    %t_plot = [0:lC_N - 1]/Fs;
		    %plot(t_plot,corrShift_t(indDop(indT),:)); grid on;
		    %xlabel('s'); title('Somme des correlations I^2 + Q^2');
		  else
		    %subplot(1,2,1)
		    %plot(fDop,corrShift_t(:,indT)); grid on;
		    %xlabel('Hz'); title('Somme des correlations I^2 + Q^2');
		    %subplot(1,2,2)
		    %t_plot = [0:lC_N - 1]/Fs;
		    %plot(t_plot,corrShift_t(indDop(indT),:)); grid on;
		    %xlabel('s'); title('Somme des correlations I^2 + Q^2');
		  endif

		  #figure();
		  #[x,y] = meshgrid(t,fDop - LO_offset);
		  #mesh(x,y,corrShift);
		  #xlabel('s'); ylabel('Hz'); zlabel('Correlation');
		  #title(sprintf('Satellite %i',satNum));
		  
		  %figure();
		  %[x,y] = meshgrid(t_plot,fDop + LO_offset);
		  %mesh(x,y,corrShift_t);
		  %xlabel('s'); ylabel('Hz'); zlabel('Correlation');
		  %title(sprintf('Satellite %i',satNum));
		  
		  % Plot correlation function of fake noise
		  plot_noise_corr = 0
		  if plot_noise_corr 
		    figure();
		    %[x,y] = meshgrid(t_plot,fDop + LO_offset);
		    [x,y] = meshgrid([1:size(corrShift)(2)], [1:size(corrShift)(1)]);
		    mesh(x,y,corrShift);
		    xlabel('s'); ylabel('Hz'); zlabel('Correlation');
		    title(sprintf('Satellite %i',satNum));
		    
		    figure();
		    %[x,y] = meshgrid(t_plot,fDop + LO_offset);
		    [x,y] = meshgrid([1:size(corrShift_t)(2)], [1:size(corrShift_t)(1)]);
		    mesh(x,y,corrShift_t);
		    xlabel('s'); ylabel('Hz'); zlabel('Correlation');
		    title(sprintf('Satellite %i',satNum));
		    
		    figure();
		    %[x,y] = meshgrid(t_plot,fDop + LO_offset);
		    [x,y] = meshgrid([1:size(corrModcompl_t)(2)], [1:size(corrModcompl_t)(1)]);
		    mesh(x,y,abs(corrModcompl_t));
		    xlabel('s'); ylabel('Hz'); zlabel('Correlation');
		    title(sprintf('Satellite %i',satNum));
		  endif
		  
		  plot_corrMod_corr = 1
		  if plot_corrMod_corr 
		    figure();
		    %[x,y] = meshgrid(t_plot,fDop + LO_offset);
		    [x,y] = meshgrid([1:size(corrShift_c)(2)], [1:size(corrShift_c)(1)]);
		    mesh(x,y,abs(corrShift_c));
		    xlabel('s'); ylabel('Hz'); zlabel('Correlation');
		    title(sprintf('Satellite %i',satNum));
		    
		    %figure();
		    %[x,y] = meshgrid(t_plot,fDop + LO_offset);
		    %[x,y] = meshgrid([1:size(z)(2)], [1:size(z)(1)]);
		    %mesh(x,y,abs(z));
		    %xlabel('s'); ylabel('Hz'); zlabel('Correlation');
		    %title(sprintf('Satellite %i',satNum));
		    
		  endif
		  
		  % Vue face au doppler
		  % view(90,0)
		endif

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% WRITE DATA INTO .CSV FILE
		% get real part of complex number for voieI
		if choice_fake = false
			'CHECK I_CHANNEL OPTION'
			corr_out = real(corrModcompl_crop);
			size(corr_out)
			if write_csv
			  file_path = strcat('outputs/i_channel/corrModcompl_crop_i_',datestr(now, 'yyyy_MM_dd_HH_mm_ss'),'.csv')
			  csvwrite(file_path, corr_out);
			endif

			'CHECK Q_CHANNEL OPTION'
			corr_out = imag(corrModcompl_crop);
			'Check matrix size: ' 
			size(corr_out)	
			if write_csv
			  file_path = strcat('outputs/q_channel/corrModcompl_crop_q_',datestr(now, 'yyyy_MM_dd_HH_mm_ss'),'.csv')
			  csvwrite(file_path, corr_out);
			endif
		else
			'CHECK WRITE SX3 CSV'
			corr_out = corrShift_c;
		      
			nom1 = file_list(file_number,[10:21])
			nom2 = '_slice_['
			nom3 = num2str(start_time);
			nom4 = ':';
			nom5 = num2str(start_time+Ns);
			nom6 = ']_sat_';
			fin_nom = num2str(satNum);
			nom_fichier = [nom1 nom2 nom3 nom4 nom5 nom6 fin_nom]
			file_path = strcat('outputs/', nom_fichier, '.csv')
			if write_csv
			   'CHECK FILE SAVED IN CSV'
			   file_path
			   %cd '/home/bellami/Documents/Export_GPS_Processing/Courbes_all'
			   csvwrite(file_path, corr_out);
			   %cd '/home/bellami/Documents/Export_GPS_Processing/'
			endif
		endif

		clear t corrShift;
		%clear corrShift_t corrModCos_t corrModSin_t corrShift_c;
	end
end
