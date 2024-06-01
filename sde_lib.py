"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
from ast import Not
import torch
import numpy as np
from utils.utils import *
# from skimage.transform import radon, iradon
from absl import flags


FLAGS = flags.FLAGS


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, config):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.config = config
        self.N = config.model.num_scales

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, mri_score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        config = self.config
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.config = config
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, input):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                grad = score_fn(x, t)
                # meas_grad = Emat_xyt(x, False, csm, atb_mask) - c2r(atb)
                # meas_grad = Emat_xyt(meas_grad, True, csm, atb_mask)
                # meas_grad /= torch.norm(meas_grad)
                # meas_grad *= torch.norm(grad)
                # meas_grad *= self.config.sampling.mse
                drift = drift - diffusion[:, None, None, None] ** 2 * \
                    grad * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t, input):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                grad = score_fn(x, t)
                # meas_grad = Emat_xyt(x, False, csm, atb_mask) - c2r(atb)
                # meas_grad = Emat_xyt(meas_grad, True, csm, atb_mask)
                # meas_grad /= torch.norm(meas_grad)
                # meas_grad *= torch.norm(grad)
                # meas_grad *= self.config.sampling.mse
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * \
                    grad * (0.5 if self.probability_flow else 1.)
                
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VESDE(SDE):
    def __init__(self, config):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(config)
        self.sigma_min = config.model.sigma_min
        self.sigma_max = config.model.sigma_max
        self.N = config.model.num_scales
        self.discrete_sigmas = torch.exp(torch.linspace(
            np.log(self.sigma_min), np.log(self.sigma_max), self.N))
        self.config = config

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        # Eq.30
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std
        


    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G


class VPSDE(SDE):
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(config)
        self.beta_0 = config.model.beta_min
        self.beta_1 = config.model.beta_max
        self.N = config.model.num_scales
        self.config = config
        self.discrete_betas = torch.linspace(self.beta_0 / self.N, self.beta_1 / self.N, self.N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        # Eq.32
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - \
            torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        # 等价无穷小：(1 + x)^alpha - 1 ~ alpha * x ===> -1/2 * beta * x ~ [(1 - beta) ^ 1/2 - 1]x
        # TODO: 为啥这里不直接算，反而要等价无穷小?
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
  def __init__(self, config):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(config)
    self.beta_0 = config.model.beta_min
    self.beta_1 = config.model.beta_max
    self.N = config.model.num_scales

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class PETSDE(SDE):
    def __init__(self, config):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(config)
        self.sigma_min = config.model.sigma_min # 0.01
        self.sigma_max = config.model.sigma_max # 348
        self.mri_sigma_max = config.model.mri_sigma_max # 1
        self.N = config.model.num_scales # 1000
        # corrector_mse = self.corrector_mse
        self.discrete_sigmas = torch.exp(torch.linspace(
            np.log(self.sigma_min), np.log(self.sigma_max), self.N))
        self.config = config

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        raise NotImplementedError

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def mri_marginal_prob(self, x, t):
        std = self.sigma_min * (self.mri_sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def pet_prior_sampling(self):
        return torch.randn((120,800))

    def prior_logp(self, z):
        raise NotImplementedError

    def discretize(self, x, t, z):
        raise NotImplementedError
    

    def reverse(self, pet_score_fn, mri_score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        discrete_sigmas = self.discrete_sigmas
        config = self.config
        # sde_fn = self.sde
        # discretize_fn = self.discretize
        
        marginal_prob_fn = self.marginal_prob

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, mri):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                sigma = config.model.sigma_min* (config.model.sigma_max / config.model.sigma_min) ** t
                drift = torch.zeros_like(x)
                diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(config.model.sigma_max) \
                                 - np.log(config.model.sigma_min)), device=t.device))

                
                grad = pet_score_fn(x, t)

                res = mri_score_fn(x, t)
                save_mat('.', res, 'res', 0, False)
                with torch.enable_grad():
                    x = x.clone().detach().requires_grad_(True)
                    std = marginal_prob_fn(x, t)[1]
                    mri_grad = (mri_score_fn(x, t) - mri) ** 2 / (2 * std ** 2)
                    mri_grad.backward(mri_grad.data)
                
                    drift = drift - diffusion[:, None, None, None] ** 2 * \
                        (grad) * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion

                dt = -1. / self.N
                z = torch.randn_like(x)

                x_mean = x + drift * dt
                x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
                return x, x_mean

            def discretize(self, x, t, un_pet, un_mri, atb_mask):
                scaler = get_data_scaler(FLAGS.config)
                timestep = (t * (self.N - 1) / T).long()
                sigma = discrete_sigmas.to(t.device)[timestep]
                adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                            discrete_sigmas[timestep - 1].to(t.device))
                f_x = torch.zeros_like(x[:,0,:,:])
                G_z = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
                if not config.training.joint:
                    grad = pet_score_fn(x, t)

                    res = mri_score_fn(x, t)
                    save_mat('.', res, 'res', 0, False)
                    with torch.enable_grad():
                        x = x.clone().detach().requires_grad_(True)
                        std = marginal_prob_fn(x, t)[1]
                        mri_grad = (mri_score_fn(x, t) - mri) ** 2 / (2 * std ** 2)
                        mri_grad = torch.mean(mri_grad)
                        mri_grad.backward(retain_graph=True)

                        rev_f = f_x - G_z[:, None, None, None] ** 2 * \
                            (grad - x.grad) * (0.5 if self.probability_flow else 1.)
                        # rev_f = f_x - G_z[:, None, None, None] ** 2 * \
                        #         grad * (0.5 if self.probability_flow else 1.)

                    rev_G = G_z

                    # x_mean = x - rev_f
                    # x = x_mean + rev_G

                    z = torch.randn_like(x)
                    x_mean = x - rev_f
                    x = x_mean + rev_G * z

                    return x, x_mean
                else:
                    grad = pet_score_fn(x, t)
                    
                    # if config.model.channel_merge:
                    #     grad = grad[:, 0, :, :]
                    #     grad = torch.unsqueeze(grad, 1)

                    # TODO MRI reconstruction
                    cx_mri = torch.zeros_like(x[:, 1, :, :])
                    x_mri = torch.complex(x[:, 1, :, :], cx_mri)
                    x_mri = torch.unsqueeze(x_mri, 1)
                    # TODO 是否需要乘以atb_mask
                    mri_recon = ifft2c_2d((fft2c_2d(x_mri) * atb_mask) - un_mri) # 复数
                    # print('mri_recon.shape=',mri_recon.shape)
  
                    mri_recon = torch.squeeze(mri_recon, 0)
                    mri_recon = mri_recon.cpu().numpy()
                    mri_recon = mri_recon.real.astype(np.float32)
                    mri_recon = torch.from_numpy(mri_recon).to(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
                    
                    x_mri = x_mri.cpu().numpy()
                    x_mri = x_mri.real.astype(np.float32)
                    x_mri = torch.from_numpy(x_mri).to(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))

                    # pet reconstruction
                    x_pet = x[:, 0, :, :]
                    x_pet = torch.unsqueeze(x_pet, 1)
                    x_pet = torch.abs(x_pet)
                 
                    radon1 = radon(x_pet) - un_pet
                    pet_recon = fbp(radon1)            
                    pet_recon = scaler(pet_recon).to(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
                    # save_mat('.', pet_recon, 'pet_recon', 0, normalize=False)
                    
                    grad_pet = grad[:,0,:,:]
                    # print('grad_pet_min_pre=',grad_pet.min())
                    # print('grad_pet_max_pre=',grad_pet.max())
                    grad_pet = torch.unsqueeze(grad_pet, 1)
                    pet_recon /= torch.norm(pet_recon)
                    pet_recon *= torch.norm(grad_pet)
                    pet_recon *= 1.5
                    rev_f_pet = f_x - G_z[:, None, None, None] ** 2 * \
                                (grad_pet - pet_recon)  * (0.8 if self.probability_flow else 1.)
                    # rev_f_pet = f_x - G_z[:, None, None, None] ** 2 * \
                    #             (grad_pet)  * (0.5 if self.probability_flow else 1.)  
                                

                    grad_mri = grad[:,1,:,:]
                    # print('grad_mri_min_pre=',grad_mri.min())
                    # print('grad_mri_max_pre=',grad_mri.max())
                    grad_mri = torch.unsqueeze(grad_mri, 1)
                    mri_recon /= torch.norm(mri_recon)
                    mri_recon *= torch.norm(grad_mri)
                    mri_recon *= 1
                    rev_f_mri = f_x - G_z[:, None, None, None] ** 2 * \
                                (grad_mri - mri_recon) * (0.8 if self.probability_flow else 1.)
                    # rev_f_mri = f_x - G_z[:, None, None, None] ** 2 * \
                    #             (grad_mri) * (0.5 if self.probability_flow else 1.)
                    rev_G = G_z
                    # print('rev_G=',rev_G)
                    x = x[:, 0, :, :]
                    x = torch.unsqueeze(x, 1)
                    z = torch.randn_like(x)
                    x_mean_pet = x_pet - rev_f_pet
                    x_pet = x_mean_pet + rev_G * z
                    x_mean_mri = x_mri - rev_f_mri
                    x_mri = x_mean_mri + rev_G * z
                    x = torch.cat((x_pet,x_mri), 1)
                    x_mean = torch.cat((x_mean_pet, x_mean_mri), 1)
                    # print('x_predictor=',x)

                    return x, x_mean

        return RSDE()


