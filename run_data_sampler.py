# Autoreload 
#import autoreload
#%load_ext autoreload
#%autoreload 2

from data_sampler import DataSampler

def main():
    discr_size_fd = 40
    scale_code = 40
    delta_tau_interv = [0.1, 0.8]
    delta_dopp_interv = [-1000, 1000]
    delta_phase = 0
    alpha_att_interv = [0.5, 0.9]
    cn0_log=50
    
    # define intervals
    # chip rate/ period of PRN code
    Fc = 1.023e6
    Tc = 1/Fc
    Nc = 1023
    Fs = 20e6
    # coherent integration period
    Tint = 1e-3
    # doppler interval
    dopp_max = min(5.5/Tint, 800+2.5/Tint)
    dopp_interval = [-dopp_max, dopp_max]
    # length of local PRN code
    lC = 20000
    # code intervals
    tau_interval = [-3/2, 5/2]
    tau_prime_interval = [0, 4]
    
    print('MODS IN INTERVALS CHECKED')    
    
    nb_samples = 5
    
    noise_i_path = r'corr_noise_generator/outputs/i_channel/*.csv'
    noise_q_path = r'corr_noise_generator/outputs/q_channel/*.csv'
    
    data_sampler = DataSampler(
                discr_size_fd=discr_size_fd,
                scale_code=scale_code,
                Tint=Tint,
                multipath_option=False,
                delta_tau_interv=delta_tau_interv, 
                delta_dopp_interv=delta_dopp_interv,
                delta_phase=delta_phase,
                alpha_att_interv=alpha_att_interv,
                tau=tau_interval, 
                dopp=dopp_interval,
                cn0_log=cn0_log
            )
    data_sampler.read_noise(noise_i_path, noise_q_path, matrix_shape=(discr_size_fd, scale_code), nb_samples=nb_samples, noise_factor=0.5)
    data_sampler.generate_corr(nb_samples=nb_samples, multipath_option=False)
    data_sampler.sum_matr()

if __name__ == '__main__':
    main()
    print('sampling completed')