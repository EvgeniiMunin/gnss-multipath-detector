#!/bin/bash

# give permission for execution
chmod +x ./run_acquisition_sampling.sh

# execute run_realiz_fake_noise.m
cd corr_noise_generator/
octave run_realiz_fake_noise.m 2 1
echo "acquisition completed"

# execute run_data_sampler.py. generate mp/no_mp
cd ..
python datagen/run_data_sampler.py --mp=0 --nb_samples=1 --cn0=48
python datagen/run_data_sampler.py --mp=1 --nb_samples=1 --cn0=48
echo "sampling/ sum completed"

exit 0


