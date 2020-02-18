function signal = reader_IFEN_SX3_AB_partition(fileIn, time_start, time_end)

%%% fileIN is the path to access the file that contains the data.
%%% Duration is the number of samples to read.

ver_software = ver();
software_name = ver_software.Name;

% Opening the file
fidIn = fopen(fileIn,'rb');

% Identifier file = -1 -> cant read the file
if (fidIn == -1) 
  error(sprintf('Erreur ouverture %s',fileIn));
  return;
end

if (software_name == 'MATLAB')
  [signal,nB] = fread(fidIn,duration,'bit2');

  if (nB ~= duration)
    error(sprintf('Erreur lecture %s',fileIn));
    return;
  end
  
  % Data conversion from 2nd's complement to amplitude format
  signal = signal.*2 + 1;
  
elseif (software_name == 'Octave')
  duration_4 = ceil((time_end - time_start)/4);
  
  fseek(fidIn, time_start, 'bof')
  [signal_4, nB_4] = fread(fidIn,duration_4,'int8');
  %[signal_4, nB_4] = fread(fidIn, time_start + time_end,'int8');
  
  fprintf(1,' time_start = %f \n', time_start);
  fprintf(1,' time_end = %f \n', time_end);
  fprintf(1,'duration_4 = %f \n', duration_4);
  fprintf(1,'singal_4 = %f \n', size(signal_4));
  'check'
  fprintf(1,'nb_4 = %f \n', size(nB_4));
  
  if (nB_4 ~= duration_4)
    error(sprintf('Erreur lecture %s',fileIn));
    return;
  end
  
  signal = zeros(4*nB_4,1);
  % Each byte of signal_4 contains 4 pairs of bits, each pair being converted 
  % into a byte, stored in turn into signal.
  signal(1:4:end) = bitand(signal_4,3);
  signal(2:4:end) = bitand(signal_4,12)/4;
  signal(3:4:end) = bitand(signal_4,48)/16;
  signal(4:4:end) = bitand(signal_4,192)/64;
  
  % Les 'bit2' ont été codés en complément à 2
  i_neg = find(signal > 1);
    signal(i_neg) = signal(i_neg) - 4;
    
  signal = signal.*2 + 1;
end

% Closing the file
fclose(fidIn);

% From reference code by IFEN
% data(1:16) = [-1 0 0 1 0 -1 -1 1 0 0 -2 0 -1 0 -1 0];
% dataAmplitude(1:16) = [-1 1 1 3 1 -1 -1 3 1 1 -3 1 -1 1 -1 1];
