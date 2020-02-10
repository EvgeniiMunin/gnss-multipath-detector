# Autoreload 
#import autoreload
#%load_ext autoreload
#%autoreload 2

from data_sampler import DataSampler

def main():
    discr_size_fd = 20
    scale_code = 20
    tau = [0, 2]
    dopp = [-2000, 2000]
    delta_tau_interv = [0.1, 0.8]
    delta_dopp_interv = [-1000, 1000]
    delta_phase = 0
    alpha_att_interv = [0.5, 0.9]
    Tint = 1e-3
    cn0_log=30
    nb_samples = 3
    
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
                tau=tau, dopp=dopp,
                cn0_log=cn0_log
            )
    data_sampler.read_noise(noise_i_path, noise_q_path, nb_samples=nb_samples, noise_factor=0.5)
    data_sampler.generate_corr(nb_samples=nb_samples, multipath_option=False)
    data_sampler.sum_matr()

if __name__ == '__main__':
    main()
    print('sampling completed')
    