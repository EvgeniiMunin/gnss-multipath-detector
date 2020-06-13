# gnss_signal_generator

The model generates the synthetic multipath anomaly in GPS L1 C/A signals and provides the CNN model to predict the presence of multipath in real and synthetic signal.

## Hardware/ Software
- GPU: 1xTesla K80
- Keras


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
python run_data_sampler.py --mp=<MULTIPATH (0 or 1)> --nb_samples=<NB_SAMPLES> --cn0=<CARRIER-TO-NOISE RATIO in dBHz> --discr=<DISCR LEVEL>
```

## DL Model
- MobileNet
- DenseNet121
The architectures were chosen as they are both suitable for mobile and embedded based CV applications where there are constraints of RAM and memory. The both architectures are produced by Google.

## Inference
To run the inference  run the script ```test_sx3_model.py``` giving the path for the input image converted into ```.csv``` format.

## Inference on Visual Processign Unit (VPU) Movidius Neural Compute Stick 2 (NCS2)
The script for running the inference on the NCS2 is located in the notebook ```convert_keras_ncs2_mpcnn.ipynb```. Here the trained Keras model is converted into Tensorflow computation graph and then into IR model compatible with NCS2.




