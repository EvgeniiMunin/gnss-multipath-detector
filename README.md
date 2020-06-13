# gnss_signal_generator

The model to predict the multipath anomaly in GPS L1 C/A signals



## Synthetic GNSS data generation and preparation
To generate the  you first need to give the execution permission to the ```run_acquisition_sampling.sh``` script
```
chmod +x ./run_acquisition_sampling.sh
```
Before generating the correlator output generate the noise using the following:
```
cd corr_noise_generator/
octave run_realiz_fake_noise.m <NB_SAMPLES> <MULTIPATH (0 or 1)>
```
Then execute the ```run_data_sampler.py``` script to generate the correlator outputs in the form of ```.csv``` files with the given carrier-to-noise ratio:
```
cd ..
python run_data_sampler.py --mp=<ULTIPATH (0 or 1)> --nb_samples=<NB_SAMPLES> --cn0=<CARRIER-TO-NOISE RATIO in dBHz> --discr=<DISCR LEVEL>
```



