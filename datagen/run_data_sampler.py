# Autoreload 
#import autoreload
#%load_ext autoreload
#%autoreload 2
import argparse
from data_sampler import DataSampler
import os

def main():
    
    # define parser to pass options from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', required=True, default=0)
    parser.add_argument('--nb_samples', required=True, default=1)
    parser.add_argument('--cn0', required=True, default=1)
    parser.add_argument('--discr', required=True, default=1)
    args = parser.parse_args()
    
    multipath_option = False if args.mp == '0' else True
    print('check multipath option before sampling: ', args.mp, multipath_option)
    
    # coherent integration period
    Tint = 20e-3
    
    # Main for data generation 
    discr_size_fd = int(args.discr)
    scale_code = int(args.discr)
    
    # multipath intervals
    delta_tau_interv = [0, 3/2]
    delta_dopp_max = min(3/Tint, 800)
    delta_dopp_interv = [-delta_dopp_max, delta_dopp_max]
    delta_phase = 0
    alpha_att_interv = [0.5, 0.9]

    cn0_log = int(args.cn0)
    print('CHECK CNO ratio: ', cn0_log)
    
    # doppler interval
    dopp_max = min(5.5/Tint, 800+2.5/Tint)
    dopp_interval = [-dopp_max, dopp_max]
    # code interval
    tau_interval = [-3/2, 5/2]
    
    print('MODS IN INTERVALS CHECKED')  
    
    noise_i_path = r'corr_noise_gen/outputs/i_channel/*.csv'
    noise_q_path = r'corr_noise_gen/outputs/q_channel/*.csv'
    
    save_path = r'synth_data/discr_{}/'.format(discr_size_fd)
    os.makedirs(os.path.dirname(save_path + '/no_mp/'), exist_ok=True)
    os.makedirs(os.path.dirname(save_path + '/mp/'), exist_ok=True)
    
    data_sampler = DataSampler(
                discr_size_fd=discr_size_fd,
                scale_code=scale_code,
                Tint=Tint,
                multipath_option=multipath_option,
                delta_tau_interv=delta_tau_interv, 
                delta_dopp_interv=delta_dopp_interv,
                delta_phase=delta_phase,
                alpha_att_interv=alpha_att_interv,
                tau=tau_interval, 
                dopp=dopp_interval,
                cn0_log=cn0_log
            )
    
    data_sampler.read_noise(noise_i_path, noise_q_path, matrix_shape=(discr_size_fd, scale_code), nb_samples=int(args.nb_samples))
    data_sampler.generate_corr(nb_samples=int(args.nb_samples))
    data_sampler.sum_matr(save_csv=True, save_path=save_path)

if __name__ == '__main__':
    main()
    print('sampling completed')
