import numpy as np
import cv2
import glob

# Import modules
import datetime
from data_generator import CorrDatasetV2, FakeNoiseDataset

import matplotlib.pyplot as plt

NOISE_COEF = 1


class DataSampler:
    def __init__(
        self,
        discr_size_fd,
        scale_code,
        Tint=10 ** -3,
        multipath_option=False,
        delta_tau_interv=None,
        delta_dopp_interv=None,
        delta_phase_interv=None,
        alpha_att_interv=None,
        tau=None,
        dopp=None,
        cn0_log=50,
    ):

        if dopp is None:
            dopp = [-2000, 2000]
        if tau is None:
            tau = [0, 2]
        self.discr_size_fd = discr_size_fd
        self.scale_code = scale_code

        self.tau = tau
        self.dopp = dopp
        self.delta_tau_interv = delta_tau_interv
        # Doppler could be negative and we need it as a label
        if delta_dopp_interv[0] > 0:
            self.delta_dopp_interv = [delta_dopp_interv[1],delta_dopp_interv[0]]
        else:
            self.delta_dopp_interv = delta_dopp_interv
        self.alpha_att_interv = alpha_att_interv
        self.delta_phase_interv = delta_phase_interv

        self.Tint = Tint
        self.cn0_log = cn0_log
        self.multipath_option = multipath_option

        # compute cn0 form cn0_log
        self.cn0 = 10 ** (0.1 * self.cn0_log)

    def read_noise(self, i_path, q_path, matrix_shape, nb_samples: int):
        fake_noise_generator = FakeNoiseDataset()

        # read i channel
        paths = np.array(glob.glob(i_path))
        ids = np.random.randint(0, len(paths) - 1, nb_samples).astype(int)
        self.noise_i_samples = fake_noise_generator.build(
            paths[ids], discr_shape=matrix_shape
        )
        # read q channel
        paths = np.array(glob.glob(q_path))
        ids = np.random.randint(0, len(paths) - 1, nb_samples).astype(int)
        self.noise_q_samples = fake_noise_generator.build(
            paths[ids], discr_shape=matrix_shape
        )

        # compute noise factor
        p = (self.noise_i_samples[0] ** 2 + self.noise_q_samples[0] ** 2).max()
        var_i = np.var(self.noise_i_samples[0])
        var_q = np.var(self.noise_q_samples[0])
        noise_factor_i = np.sqrt(p / (2 * var_i * self.Tint * self.cn0)) * NOISE_COEF
        noise_factor_q = np.sqrt(p / (2 * var_q * self.Tint * self.cn0)) * NOISE_COEF

        # rotate noise vector by 90 degs
        self.noise_i_samples = np.rot90(self.noise_i_samples, axes=(1, 2))
        self.noise_q_samples = np.rot90(self.noise_i_samples, axes=(1, 2))

        # apply noise factor
        self.noise_i_samples *= noise_factor_i
        self.noise_q_samples *= noise_factor_q

        self.noise_i_samples = np.transpose(self.noise_i_samples, [0, 2, 1])
        self.noise_q_samples = np.transpose(self.noise_q_samples, [0, 2, 1])

    def generate_corr(self, nb_samples=13):
        # no_mp/ mp option
        Dataset = CorrDatasetV2(
            discr_size_fd=self.discr_size_fd,
            scale_code=self.scale_code,
            Tint=self.Tint,
            multipath_option=self.multipath_option,
            delta_tau_interv=self.delta_tau_interv,
            delta_dopp_interv=self.delta_dopp_interv,
            delta_phase_interv=self.delta_phase_interv,
            alpha_att_interv=self.alpha_att_interv,
            tau=self.tau,
            dopp=self.dopp,
            cn0_log=self.cn0_log,
        )

        samples = Dataset.build(nb_samples=nb_samples)
        # extract separately I,Q channels
        sample_i_func = lambda x: x["table"][..., 0]
        sample_q_func = lambda x: x["table"][..., 1]
        self.i_samples = np.array(list(map(sample_i_func, samples)))
        self.q_samples = np.array(list(map(sample_q_func, samples)))


    def sum_matr(self, save_csv=True, save_path=None):

        # check matrices shapes
        try:
            # check correspondance among arrays shapes
            if (self.i_samples.shape[1] != self.noise_i_samples.shape[1]) or (
                self.i_samples.shape[2] != self.noise_i_samples.shape[2]
            ):
                print("Wrong arrays shapes. ValueError exception")
                raise ValueError

            if self.i_samples.shape[0] != self.noise_i_samples.shape[0]:
                # print('Nb samples correction: ', self.i_samples.shape, self.noise_i_samples.shape)
                min_nb_samples = min(
                    self.i_samples.shape[0], self.noise_i_samples.shape[0]
                )
                self.i_samples = self.i_samples[:min_nb_samples, ...]
                self.q_samples = self.q_samples[:min_nb_samples, ...]
                self.noise_i_samples = self.noise_i_samples[:min_nb_samples, ...]
                self.noise_q_samples = self.noise_q_samples[:min_nb_samples, ...]

            matr_i = np.sum([self.i_samples, self.noise_i_samples], axis=0)
            matr_q = np.sum([self.q_samples, self.noise_q_samples], axis=0)

            # save matrix into csv
            if save_csv:
                for i in range(matr_i.shape[0]):
                    print("---------- EXAMPLE {} --------".format(i))
                    datetime_now = datetime.datetime.now()
                    # save i/q_channel
                    if self.multipath_option:
                        pathi = save_path + "mp/Voie_I/channel_i_{}_{}_{}_{}_{}.csv".format(
                            str(datetime_now), ("%.9f" % self.delta_tau_interv[1]),("%.2f" % self.cn0_log), 
                            str(self.delta_dopp_interv[1]).zfill(4), ("%.6f" % self.delta_phase_interv[1])
                        )
                        pathq = save_path + "mp/Voie_Q/channel_q_{}_{}_{}_{}_{}.csv".format(
                            str(datetime_now), ("%.9f" % self.delta_tau_interv[1]),("%.2f" % self.cn0_log),
                            str(self.delta_dopp_interv[1]).zfill(4), ("%.6f" % self.delta_phase_interv[1])
                        )
                    else:
                        pathi = save_path + "no_mp/Voie_I/channel_i_{}_{}.csv".format(
                            str(datetime_now), self.delta_dopp_interv[1]
                        )
                        pathq = save_path + "no_mp/Voie_Q/channel_q_{}_{}.csv".format(
                            str(datetime_now), self.delta_dopp_interv[1]
                        )
                    print(pathi)
                    np.savetxt(pathi, matr_i[i, ...], delimiter=",")
                    np.savetxt(pathq, matr_q[i, ...], delimiter=",")
            else:
                return matr_i, matr_q

        except ValueError:
            print("TEST yes ValueError",ValueError)
            print(
                "Wrong arrays shapes: sampels: {}, {}; noise samples {}, {}".format(
                    self.i_samples.shape,
                    self.q_samples.shape,
                    self.noise_i_samples.shape,
                    self.noise_q_samples.shape,
                )
            )
