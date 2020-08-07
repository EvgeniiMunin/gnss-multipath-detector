# gnss-multipath-detector

The model generates the synthetic multipath anomaly in GPS L1 C/A signals and provides the CNN model to predict the presence of multipath in real and synthetic signal.

## Installation
To get started with pretrained multipath detection model based on MobileNet please pull the Docker image from DockerHub repository
```
docker pull evgeniimunin/mp-app-mod:latest
```
Then run the container as follows on the port 5000. This runs the server and makes it listen 
```
docker run -it -p 5000:5000 mp-app-mod
```
Download two .csv files for I,Q channels and move them to your current directory. To make the prediction of probability of multipath  open another terminal window and run the following command
```
curl -X POST -F matrixi=@imgi.csv -F matrixq=@imgq.csv http://localhost:5000/predict
```

## Deployment
The multipath detection app is also deployed on the GCP Kubernetes cluster and is available publicly by the API. To make the prediction run the following command
```
curl -X POST -F matrixi=@imgi.csv -F matrixq=@imgq.csv http://35.226.109.248:5000/predict
```

## Hardware/ Software
- GPU: 1xTesla K80
- Keras

## Synthetic GNSS data generation and preparation
To generate the GNSS correlator output data you first need to give the execution permission to the ```run_acquisition_sampling.sh``` script
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



