import torch
import torch.nn as nn
import warnings
from .ddpm.unet import create_model
from .precond import VPPrecond


__MODEL__ = {}


def register_model(name: str):
    def wrapper(cls):
        if name in __MODEL__ and __MODEL__[name] != cls:
            warnings.warn(f"Model '{name}' is already registered.", UserWarning)
        __MODEL__[name] = cls
        cls.name = name
        return cls
    return wrapper


def get_model(name: str, **kwargs):
    if name not in __MODEL__:
        raise NameError(f"Model '{name}' is not registered.")
    return __MODEL__[name](**kwargs)


class DiffusionModel(nn.Module):
    """
    Base Diffusion Model class.
    Requires overriding either 'score' or 'tweedie' method.
    """

    def __init__(self):
        super(DiffusionModel, self).__init__()
        if (self.score.__func__ is DiffusionModel.score and
            self.tweedie.__func__ is DiffusionModel.tweedie):
            raise NotImplementedError("Either 'score' or 'tweedie' method must be overridden.")

    def score(self, x, sigma):
        """
        Compute the score function \nabla_{x_t} log p(x_t; sigma_t).

        Args:
            x (Tensor): Noisy input tensor at time t, shape [B, *data_shape].
            sigma (float): Noise level at time t.
        """
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def tweedie(self, x, sigma):
        """
        Compute the expected clean data given noisy data.

        Args:
            x (Tensor): Noisy input tensor at time t, shape [B, *data_shape].
            sigma (float): Noise level at time t.
        """
        return x + self.score(x, sigma) * sigma ** 2

    def get_in_shape(self):
        """Return the shape of the model's input data."""
        pass


@register_model(name='ddpm')
class DDPM(DiffusionModel):
    """
    DDPM (Diffusion Denoising Probabilistic Model).
    Attributes:
        model (VPPrecond): The neural network used for denoising.

    Methods:
        __init__(self, model_config, device='cuda'): Initializes the DDPM object.
        tweedie(self, x, sigma): Applies the DDPM model to denoise the input, using VP preconditioning from EDM.
    """

    def __init__(self, model_config, device='cuda', requires_grad=False):
        super().__init__()
        self.model = VPPrecond(model=create_model(**model_config), learn_sigma=model_config['learn_sigma'],
                               conditional=model_config['class_cond']).to(device)
        self.model.eval()
        self.model.requires_grad_(requires_grad)
        self.image_size = model_config['image_size']

    def tweedie(self, x, sigma):
        return self.model(x, torch.as_tensor(sigma).to(x.device))

    def get_in_shape(self):
        return (3, self.image_size, self.image_size)
        
