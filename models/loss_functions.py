import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import torch
import torch.nn.functional as F


def _nan2zero(x):
    """
    Replaces NaN values in the tensor `x` with zeros.

    Args:
        x: Input tensor.

    Returns:
        Tensor with NaNs replaced by zeros.
    """
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    """
    Replaces NaN values in the tensor `x` with infinity.

    Args:
        x: Input tensor.

    Returns:
        Tensor with NaNs replaced by infinity.
    """
    return torch.where(torch.isnan(x), torch.zeros_like(x) + float('inf'), x)


def _nelem(x):
    """
    Counts the number of non-NaN elements in the tensor `x`.
    If there are no non-NaN elements, returns 1 to avoid division by zero.

    Args:
        x: Input tensor.

    Returns:
        Number of non-NaN elements.
    """
    nelem = torch.sum(~torch.isnan(x).float())
    return torch.where(nelem == 0., torch.tensor(1., dtype=x.dtype), nelem)

# TODO: May change this to a function in order to simplify.
# theta is the dispersion parameter and is a scalar
# scale_factor, scales the negative binomial mean before the calculation of the loss,
# to balance the learning rates of theta and the network weights.
class nb_loss(nn.Module):
    def __init__(self, theta=None, masking=False, scale_factor=1.0, debug=False):
        super(nb_loss, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.masking = masking
        self.theta = theta

    def forward(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        y_true = y_true.float()
        y_pred = y_pred.float() * scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

        # Clip theta
        theta = min(self.theta, 1e6)

        t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + \
             (y_true * (torch.log(theta + eps) - torch.log(y_pred + eps)))

        if self.debug:
            assert torch.isfinite(y_pred).all(), 'y_pred has inf/nans'
            assert torch.isfinite(t1).all(), 't1 has inf/nans'
            assert torch.isfinite(t2).all(), 't2 has inf/nans'

            print(f"Histogram t1: {t1.histc()}")
            print(f"Histogram t2: {t2.histc()}")

        final = t1 + t2

        final = torch.nan_to_num(final, nan=float('inf'))

        if mean:
            if self.masking:
                final = final.sum() / nelem
            else:
                final = final.mean()

        return final