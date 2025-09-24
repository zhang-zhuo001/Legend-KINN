import torch
import torch.nn as nn


class LegendreKANLayer(nn.Module):
    """
    Represents a single layer of the Chebyshev-based KAN network.
    """

    def __init__(self, input_dim, output_dim, degree):
        """
        Initialize the layer with input/output dimensions and Chebyshev polynomial degree.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            degree (int): Degree of the Chebyshev polynomials used.
        """
        super(LegendreKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        # Initialize trainable coefficients for Chebyshev polynomials
        self.cheby2_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )

        nn.init.normal_(
            self.cheby2_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1))
        )

    def forward(self, x):
        """
        Forward pass through the layer using Chebyshev polynomials.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Ensure the input has the correct shape
        x = torch.reshape(x, (-1, self.inputdim))  # Reshape to (batch_size, inputdim)

        # Normalize input to [-1, 1] using Tanh (assuming x is not already normalized)
        x = torch.tanh(x)

        # Initialize tensor to store Chebyshev polynomials of the second kind
        cheby2 = torch.ones(
            x.shape[0], self.inputdim, self.degree + 1, device=x.device
        )  # Shape: (batch_size, inputdim, degree+1)

        # Compute Chebyshev polynomials using the recurrence relation
        if self.degree >= 1:
            cheby2[:, :, 1] = x
        if self.degree >= 2:
            cheby2[:, :, 2] = 0.5 * (3 * x ** 2 - 1)
        if self.degree >= 3:
            cheby2[:, :, 3] = 0.5 * (5 * x ** 3 - 3 * x)
        if self.degree >= 4:
            cheby2[:, :, 4] = 0.125 * (35 * x ** 4 - 30 * x ** 2 + 3)

            # Perform Chebyshev interpolation using the coefficients
        # einsum "bid,iod->bo" performs weighted summation over polynomial terms
        y = torch.einsum(
            "bid,iod->bo", cheby2, self.cheby2_coeffs
        )  # Output shape: (batch_size, output_dim)

        # Ensure the output has the correct shape
        y = y.view(-1, self.outdim)
        return y


class LegendreKAN(nn.Module):
    """
    Represents the Chebyshev-based Kolmogorov–Arnold Network (KAN).
    """

    def __init__(self, network, degree):
        """
        Initialize the ChebyKAN network.

        Args:
            network (list[int]): List defining the number of nodes in each layer.
            degree (int): Degree of Chebyshev polynomials for each layer.
        """
        super(LegendreKAN, self).__init__()
        self.network = network  #
        self.layers = nn.ModuleList()

        # Define the layers based on the specified network architecture
        for i in range(len(network) - 1):
            self.layers.append(LegendreKANLayer(network[i], network[i + 1], degree))

    def forward(self, x):
        """
        Forward pass through the entire network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = x.view(-1, self.network[0])  # Reshape input to match the first layer

        # Pass the input through all layers sequentially
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x

class LegendreKAN_norm(nn.Module):
    """
    Represents the Chebyshev-based Kolmogorov–Arnold Network (KAN).
    """

    def __init__(self, network,X, degree,device):
        """
        Initialize the ChebyKAN network.

        Args:
            network (list[int]): List defining the number of nodes in each layer.
            degree (int): Degree of Chebyshev polynomials for each layer.
        """

        super(LegendreKAN_norm, self).__init__()
        super().__init__()

        self.X_mean = torch.from_numpy(X.mean(0, keepdims=True)).float()
        self.X_std = torch.from_numpy(X.std(0, keepdims=True)).float()
        self.X_mean = self.X_mean.to(device)
        self.X_std = self.X_std.to(device)

        self.network = network  #
        self.layers = nn.ModuleList()

        # Define the layers based on the specified network architecture
        for i in range(len(network) - 1):
            self.layers.append(LegendreKANLayer(network[i], network[i + 1], degree))

    def forward(self, x):
        """
        Forward pass through the entire network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = ((x - self.X_mean) / self.X_std)  # z-score norm
        x = x.view(-1, self.network[0])  # Reshape input to match the first layer

        # Pass the input through all layers sequentially
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x