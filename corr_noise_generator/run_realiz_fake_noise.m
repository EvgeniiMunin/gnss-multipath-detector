clear all; close all;

interactive = true;
choice = 'debug';
nb_realiz = argv(){1}

for realiz = [1: nb_realiz]
  %'CHECK argv(){1}', argv(){1}, realiz
  test_acquisition_GPS_fake_noise_v3_corr_modif
end