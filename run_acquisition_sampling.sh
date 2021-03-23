#!/bin/bash

# give permission for execution
#chmod +x ./run_acquisition_sampling.sh

# execute run_realiz_fake_noise.m
#cd datagen/corr_noise_gen/
#octave run_realiz_fake_noise.m 600 1
#echo "acquisition completed"

# execute run_data_sampler.py. generate mp/no_mp
#cd ..
python3 run_data_sampler.py --mp=0 --nb_samples=300 --dopp=125 --tau=1.5 --cn0=36 --discr=80
python3 run_data_sampler.py --mp=1 --nb_samples=300 --dopp=10 --tau=1.5 --cn0=36 --discr=80
echo "sampling/ sum completed"

exit 0


