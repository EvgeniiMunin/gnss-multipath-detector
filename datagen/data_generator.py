# Autoreload
# import autoreload
#%load_ext autoreload
#%autoreload 2

import numpy as np
import math
from scipy import signal
import cv2

# adjustment coeffs
SINC_WIDTH_COEF = 1


class CorrDatasetV2:
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
            dopp = [-1000, 1000]
        if tau is None:
            tau = [0, 2]
        self.discr_size_fd = discr_size_fd
        self.scale_code = scale_code

        self.cn0_log = cn0_log  # C/N0
        self.Tint = Tint

        self.multipath_option = multipath_option
        self.delta_tau_interv = delta_tau_interv
        self.delta_dopp_interv = delta_dopp_interv
        self.delta_phase_interv = delta_phase_interv
        self.alpha_att_interv = alpha_att_interv
        self.tau = tau
        self.dopp = dopp

        # calculate SNR
        self.sign_amp = 1
        self.sign_power = 8 * self.sign_amp / self.Tint ** 2
        self.noise_psd = self.sign_power / 10 ** (0.1 * self.cn0_log)

    def __sin_cos_matrix__(
        self, multipath=False, delta_dopp=0, delta_phase=0, xk=0, yk=0
    ):
        dopp_axis = np.linspace(
            start=self.dopp[0], stop=self.dopp[1], num=self.discr_size_fd
        )
        cos_array = np.array(
            [math.cos(math.pi * x * self.Tint + delta_phase) for x in dopp_axis]
        )
        sin_array = np.array(
            [math.sin(math.pi * x * self.Tint + delta_phase) for x in dopp_axis]
        )
        return cos_array, sin_array

    def __noise_model__(self):
        noise_corr_mean = 0
        noise_corr_std = math.sqrt(self.noise_psd * self.Tint / 16)
        return noise_corr_mean, noise_corr_std

    def __generate_peak__(
        self,
        multipath=False,
        delta_dopp=0,
        delta_tau=0,
        delta_phase=0,
        alpha_att=1,
        ref_features=False,
    ):
        x = np.linspace(self.dopp[0], self.dopp[1], self.discr_size_fd)
        y = np.linspace(self.tau[0], self.tau[1], self.scale_code)

        # Create empty matrix for peaks
        matrix_i = np.zeros((self.discr_size_fd, self.scale_code))
        matrix_q = np.zeros((self.discr_size_fd, self.scale_code))
        matrix_tr_i = np.zeros((self.discr_size_fd, self.scale_code // 2))
        matrix_tr_q = np.zeros((self.discr_size_fd, self.scale_code // 2))

        # Convert tau/ doppler deviation into pixel scale
        xk = int(x.mean() + delta_dopp / (x.max() - x.min()) * self.discr_size_fd)
        yk = int(y.mean() + delta_tau / (y.max() - y.min()) * self.scale_code)

        # adjust xk
        if xk <= -matrix_i.shape[0] // 2:
            xk = matrix_i.shape[0] - abs(xk)
        if xk >= matrix_i.shape[0] // 2:
            xk = -(matrix_i.shape[0] - abs(xk))

        # navigation bit
        bit_nav = np.random.choice([-1, 1])
        mp_bit_nav = np.random.choice([-1, 1])
        #bit_nav = -1
        #mp_bit_nav = 1

        # Generate triangle function
        func1 = bit_nav * self.sign_amp * signal.triang(self.scale_code // 2)

        # Generate sinc*sin / sinc*cos functions for I, Q channels
        sin_cos_array = self.__sin_cos_matrix__(
            multipath=multipath,
            delta_dopp=delta_dopp,
            delta_phase=delta_phase,
            xk=xk,
            yk=yk,
        )
        func2i = (
            self.sign_amp * np.sinc(x * self.Tint * SINC_WIDTH_COEF) * sin_cos_array[0]
        )
        func2q = (
            self.sign_amp * np.sinc(x * self.Tint * SINC_WIDTH_COEF) * sin_cos_array[1]
        )

        # Only 1 principal peak
        for i, (pointi, pointq) in enumerate(zip(func2i, func2q)):
            matrix_tr_i[i] = alpha_att * func1 * pointi
            matrix_tr_q[i] = alpha_att * func1 * pointq

        # sum matrix_tr and matrix of background according interval offset
        matrix_i[:, 5 : (self.scale_code // 2 + 5)] = matrix_tr_i
        matrix_q[:, 5 : (self.scale_code // 2 + 5)] = matrix_tr_q

        # Apply offset on dopp (xk) and code to mutlipath peak
        if multipath:
            if xk >= 0:
                matrix_i = mp_bit_nav * matrix_i[: matrix_i.shape[0] - xk, : matrix_i.shape[1] - yk]
                matrix_q = mp_bit_nav * matrix_q[: matrix_q.shape[0] - xk, : matrix_q.shape[1] - yk]
            else:
                matrix_i = mp_bit_nav * matrix_i[abs(xk):, : matrix_i.shape[1] - yk]
                matrix_q = mp_bit_nav * matrix_q[abs(xk):, : matrix_q.shape[1] - yk]

        matrix = np.concatenate((matrix_i[..., None], matrix_q[..., None]), axis=2)
        return matrix, xk, yk

    def build(self, nb_samples=10, ref_features=False):
        data_samples = []
        for i in range(nb_samples):
            data = {}

            # Set random delta_tau / delta_dopp
            delta_taui = np.random.uniform(
                low=self.delta_tau_interv[0], high=self.delta_tau_interv[1]
            )
            # delta_doppi = np.random.normal(0, self.delta_dopp_interv[1])
            # modification for doppler labelling
            delta_doppi = self.delta_dopp_interv[1]
            delta_phasei = np.random.uniform(
                low=self.delta_phase_interv[0], high=self.delta_phase_interv[1]
            )
            alpha_atti = np.random.uniform(
                low=self.alpha_att_interv[0], high=self.alpha_att_interv[1]
            )

            # Generate matrices: main, multipath
            if self.multipath_option:
                matrix, x, y = self.__generate_peak__()
                matrix_mp, x, y = self.__generate_peak__(
                    multipath=self.multipath_option,
                    delta_dopp=delta_doppi,
                    delta_tau=delta_taui,
                    delta_phase=delta_phasei,
                    alpha_att=alpha_atti,
                    ref_features=ref_features,
                )

                if x >= 0:
                    matrix[x:, y:] = matrix_mp + matrix[x:, y:]
                else:
                    matrix[: matrix.shape[0] - abs(x), y:] = (
                        matrix_mp + matrix[: matrix.shape[0] - abs(x), y:]
                    )
            else:
                matrix, x, y = self.__generate_peak__(
                    delta_phase=0, ref_features=ref_features
                )

            module = np.sqrt(matrix[..., 0] ** 2 + matrix[..., 1] ** 2)
            # matrix[..., 0] = (matrix[..., 0] - module.min()) / (
                # module.max() - module.min()
            # )
            max_amp_I = max(abs(matrix[..., 0].min()),matrix[..., 0].max())
            matrix[..., 0] = matrix[..., 0]/max_amp_I
            # matrix[..., 1] = (matrix[..., 1] - module.min()) / (
                # module.max() - module.min()
            # )
            # Est-ce-qu'il ne vaudrait pas mieux faire la normalisation sur
            # l'image avant ajout du multitrajet ?
            matrix[..., 1] = matrix[..., 1]/max_amp_I
            # module = (module - module.min()) / (module.max() - module.min())
            module = np.sqrt(matrix[..., 0] ** 2 + matrix[..., 1] ** 2)

            data["table"] = matrix
            data["module"] = module[..., None]

            data_samples.append(data)

        return np.array(data_samples)


class FakeNoiseDataset:
    def __init__(self):
        self.noise_data = []

    def __preprocess__(self, path):
        a = np.genfromtxt(path, delimiter=",")
        a = cv2.resize(a, self.discr_shape)
        return a

    def build(self, paths, discr_shape=(40, 40)):
        self.discr_shape = discr_shape
        noise_data = []
        for path in paths:
            noise_matr = self.__preprocess__(path)

            # scale noise matrix
            # noise_matr = (noise_matr - noise_matr.min()) / (
                # noise_matr.max() - noise_matr.min()
            # )
            # noise_matr = noise_matr/max(abs(noise_matr.min()),noise_matr.max())
            noise_data.append(noise_matr)
        return np.array(noise_data)


def filter_2der(img, kernel_size):
    filt = np.array([1, -2, 1])[:, None]
    img = cv2.medianBlur(np.float32(img), kernel_size)
    img = cv2.filter2D(img, cv2.CV_32F, filt)
    return img
