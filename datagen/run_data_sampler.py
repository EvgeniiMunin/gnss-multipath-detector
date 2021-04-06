# Autoreload
# import autoreload
#%load_ext autoreload
#%autoreload 2
import argparse
from data_sampler import DataSampler
import os
import numpy as np

TINT = 20e-3
delta_tau_min, delta_tau_max = 0, 3 / 2
delta_dopp_max = 125 # min(2.5 / TINT, 800)
delta_phase_min, delta_phase_max = 0, 2 * np.pi
alpha_att_min, alpha_att_max = 0.1, 0.9
dopp_max = min(5 / TINT, 800 + 2.5 / TINT)
tau_min, tau_max = -3 / 2, 5 / 2

def main():

    # define parser to pass options from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp", required=True, default=0)
    parser.add_argument("--nb_samples", required=True, default=1)
    parser.add_argument("--dopp", required=True, default=1)
    parser.add_argument("--tau", required=True, default=1)
    parser.add_argument("--cn0", required=True, default=1)
    parser.add_argument("--discr", required=True, default=1)
    args = parser.parse_args()

    multipath_option = False if args.mp == "0" else True
    print("check multipath option before sampling: ", args.mp, multipath_option)

    # Main for data generation
    discr_size_fd = int(args.discr)
    scale_code = int(args.discr)

    # multipath intervals
    delta_tau_interv = [delta_tau_min, float(args.tau)]
    delta_dopp_interv = [-int(args.dopp),int(args.dopp)]
    delta_phase_interv = [delta_phase_min, delta_phase_max]
    alpha_att_interv = [alpha_att_min, alpha_att_max]

    cn0_log = int(args.cn0)

    # code/ doppler interval
    dopp_interval = [-dopp_max, dopp_max]
    tau_interval = [tau_min, tau_max]

    # noise_i_path = r"corr_noise_gen/outputs/i_channel/*.csv"
    # noise_q_path = r"corr_noise_gen/outputs/q_channel/*.csv"
    noise_i_path = r'corr_noise_gen/outputs/snap_debug_noise_sat_01_89x81/*_I.csv'
    noise_q_path = r'corr_noise_gen/outputs/snap_debug_noise_sat_01_89x81/*_Q.csv'

    #save_path = r"synth_data/discr_{}_dopp-{}_delay-{}_cn-{}/".format(discr_size_fd, args.dopp, args.tau, args.cn0)
    # Modification to put all the signals with same cn0 and delay but with different doppler in a same and unique folder
    save_path = r"synth_data/experiences/"
    os.makedirs(os.path.dirname(save_path + "/no_mp/"), exist_ok=True)
    os.makedirs(os.path.dirname(save_path + "/mp/"), exist_ok=True)
    
    data_sampler = DataSampler(
        discr_size_fd=discr_size_fd,
        scale_code=scale_code,
        Tint=TINT,
        multipath_option=multipath_option,
        delta_tau_interv=delta_tau_interv,
        delta_dopp_interv=delta_dopp_interv,
        delta_phase_interv=delta_phase_interv,
        alpha_att_interv=alpha_att_interv,
        tau=tau_interval,
        dopp=dopp_interval,
        cn0_log=cn0_log,
    )

    data_sampler.read_noise(
        noise_i_path,
        noise_q_path,
        matrix_shape=(discr_size_fd, scale_code),
        nb_samples=int(args.nb_samples),
    )
    data_sampler.generate_corr(nb_samples=int(args.nb_samples))
    data_sampler.sum_matr(save_csv=True, save_path=save_path)


if __name__ == "__main__":
    main()
    print("sampling completed")
