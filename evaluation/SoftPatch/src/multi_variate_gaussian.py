"""Multi Variate Gaussian Distribution."""

from typing import Any, List, Optional

import torch
from torch import Tensor, nn


class MultiVariateGaussian(nn.Module):
    """Multi Variate Gaussian Distribution."""

    def __init__(self, n_features, n_patches):
        super().__init__()

        self.register_buffer("mean", torch.zeros(n_features, n_patches))
        self.register_buffer("inv_covariance", torch.eye(n_features).unsqueeze(0).repeat(n_patches, 1, 1))

        self.mean: Tensor
        self.inv_covariance: Tensor

    @staticmethod
    def _cov(
        observations: Tensor,
        rowvar: bool = False,
        bias: bool = False,
        ddof: Optional[int] = None,
        aweights: Tensor = None,
    ) -> Tensor:

        # ensure at least 2D
        if observations.dim() == 1:
            observations = observations.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and observations.shape[0] != 1:
            observations = observations.t()

        if ddof is None:
            if bias == 0:
                ddof = 1
            else:
                ddof = 0

        weights = aweights
        weights_sum: Any

        if weights is not None:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float)  # pylint: disable=not-callable
            weights_sum = torch.sum(weights)
            avg = torch.sum(observations * (weights / weights_sum)[:, None], 0)
        else:
            avg = torch.mean(observations, 0)

        # Determine the normalization
        if weights is None:
            fact = observations.shape[0] - ddof
        elif ddof == 0:
            fact = weights_sum
        elif aweights is None:
            fact = weights_sum - ddof
        else:
            fact = weights_sum - ddof * torch.sum(weights * weights) / weights_sum

        observations_m = observations.sub(avg.expand_as(observations))

        if weights is None:
            x_transposed = observations_m.t()
        else:
            x_transposed = torch.mm(torch.diag(weights), observations_m).t()

        covariance = torch.mm(x_transposed, observations_m)
        covariance = covariance / fact

        return covariance.squeeze()

    def forward(self, embedding: Tensor) -> List[Tensor]:
        """Calculate multivariate Gaussian distribution.

        Args:
          embedding (Tensor): CNN features whose dimensionality is reduced via either random sampling or PCA.

        Returns:
          mean and inverse covariance of the multi-variate gaussian distribution that fits the features.
        """
        device = embedding.device
        patch, _, channel = embedding.shape
        embedding_vectors = embedding.permute(1, 2, 0)

        # batch, channel, height, width = embedding.size()
        # embedding_vectors = embedding.view(batch, channel, height * width)
        self.mean = torch.mean(embedding_vectors, dim=0)
        covariance = torch.zeros(size=(channel, channel, patch), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(patch):
            covariance[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity
            # (evals, evecs) = torch.eig(covariance[:, :, i])  #
            # compaction[i] = evals[:, 0].max() #torch.max(evals[:, 0])
        # calculate inverse covariance as we need only the inverse
        self.inv_covariance = torch.linalg.inv(covariance.permute(2, 0, 1))
        # compaction = covariance.norm(p=2, dim=(0, 1))

        return [self.mean, self.inv_covariance]  #
        # return [self.mean, self.inv_covariance, compaction]  #

    def fit(self, embedding: Tensor) -> List[Tensor]:
        """Fit multi-variate gaussian distribution to the input embedding.

        Args:
            embedding (Tensor): Embedding vector extracted from CNN.

        Returns:
            Mean and the covariance of the embedding.
        """
        return self.forward(embedding)
