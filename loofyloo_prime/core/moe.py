
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    A simple expert network.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initializes the Expert.

        Args:
            input_dim (int): The dimension of the input.
            output_dim (int): The dimension of the output.
        """
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the Expert.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, output_dim).
        """
        return self.fc(x)

class GatingNetwork(nn.Module):
    """
    A gating network that determines which experts to use.
    """
    def __init__(self, input_dim, num_experts):
        """
        Initializes the GatingNetwork.

        Args:
            input_dim (int): The dimension of the input.
            num_experts (int): The number of experts.
        """
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        Forward pass of the GatingNetwork.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, num_experts).
        """
        return F.softmax(self.fc(x), dim=-1)

class MoELayer(nn.Module):
    """
    A Mixture of Experts (MoE) layer.
    """
    def __init__(self, input_dim, output_dim, num_experts):
        """
        Initializes the MoELayer.

        Args:
            input_dim (int): The dimension of the input.
            output_dim (int): The dimension of the output.
            num_experts (int): The number of experts.
        """
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        """
        Forward pass of the MoELayer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, output_dim).
        """
        gating_weights = self.gating(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=2)
        return output
