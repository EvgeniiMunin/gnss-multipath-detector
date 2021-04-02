#!/bin/bash

# give permission for execution
#chmod +x ./run_acquisition_sampling.sh

# execute run_realiz_fake_noise.m
#<<<<<<< HEAD
#cd datagen/corr_noise_gen/
#octave run_realiz_fake_noise.m 600 1
#echo "acquisition completed"

# execute run_data_sampler.py. generate mp/no_mp
#cd ..
cd datagen/
for _ in `seq 1 4`;
do
	rand=$(echo "scale=10;($RANDOM/32768)*501 - 250"|bc -l)
	doppler=${rand%.*}
	#python3 run_data_sampler.py --mp=0 --nb_samples=1 --dopp=200 --tau=1.5 --cn0=43 --discr=80
	python3 run_data_sampler.py --mp=1 --nb_samples=1 --dopp=$doppler --tau=1.5 --cn0=43 --discr=80
done
#=======
#cd datagen/corr_noise_gen/
#octave run_realiz_fake_noise.m 20 1
echo "acquisition completed"

# execute run_data_sampler.py. generate mp/no_mp
cd ..
#for i in `seq 1 10`;
#do
#	python3 run_data_sampler.py --mp=0 --nb_samples=1 --cn0=48 --discr=80
#	python3 run_data_sampler.py --mp=1 --nb_samples=1 --cn0=48 --discr=80
#	echo $i
#done
#>>>>>>> modif
echo "sampling/ sum completed"

exit 0


