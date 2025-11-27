import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (
    degree,
    remove_self_loops,
    add_self_loops,
    add_remaining_self_loops,
)
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import ChebConv
import config

class Velocity(nn.Module):
    def __init__(self, init_v=7):
        super(Velocity, self).__init__()
        self.ve = nn.Parameter(torch.FloatTensor([init_v]))

    def forward(self, LA):
        return torch.pow(LA, self.ve)



class TEDGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(TEDGCN, self).__init__()
        self.ve = Velocity(7)
        self.W = nn.Linear(in_channels, hidden_dim)
        self.softmax = softmax
        self.bn = nn.BatchNorm1d(hidden_dim)
        if config.dataset in ["squirrel"]:
            self.p = 0.1
        else:
            self.p = 0.2
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(self.p)
        self.MLP = nn.Linear(hidden_dim, out_channels)

    def forward(self, X, La, U):
        V_La = self.ve(La)
        out_A = torch.mm(torch.mm(U, torch.diag(V_La)), torch.transpose(U, 0, 1))
        out = torch.mm(out_A, X)
        out = self.W(out)
        hidden_emd = out
        if not config.small_graph and config.dataset not in []:
            out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.MLP(out)
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.nn.Sigmoid()(out), hidden_emd


import torch
import torch.nn as nn
import torch.nn.functional as F

class EigenExpert(nn.Module):
    def __init__(self, node_dim, hidden_dim=64):
        super().__init__()
        # MLP to process each eigenvector
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, U_i):
        """Process eigenvectors to produce importance weights.
        
        Args:
            U_i: [n_nodes, n_eigenvecs] tensor of eigenvectors
            
        Returns:
            V_i: [n_eigenvecs] tensor of importance weights
        """
        # Process each eigenvector independently
        weights = []
        for i in range(U_i.size(1)):
            # Extract single eigenvector [n_nodes]
            vec = U_i[:, i]
            # Add channel dimension [n_nodes, 1]
            vec = vec.unsqueeze(-1)
            # Process through MLP [n_nodes, 1]
            transformed = self.mlp(vec)
            # Average across nodes to get importance [1]
            weight = transformed.mean(dim=0)
            weights.append(weight)
        # Combine into single tensor [n_eigenvecs]
        return torch.cat(weights, dim=0)



class Morgan(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(Morgan, self).__init__()
        self.softmax = softmax
        num_experts = 5
        self.num_experts = num_experts
        
        # Create module list of experts
        self.experts = nn.ModuleList([
            EigenExpert(node_dim=1) for _ in range(num_experts)
        ])
        
        # Dynamic gating network
        # self.gate = nn.Sequential(
        #     nn.Linear(num_experts, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, num_experts),
        #     nn.Softmax(dim=-1)
        # )

        # for homepholy
        self.gate = nn.Sequential(
            nn.Linear(num_experts, num_experts),
            nn.ReLU(),
            nn.Linear(num_experts, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Feature transformation layers
        self.W = nn.Linear(in_channels, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MLP = nn.Linear(hidden_dim, out_channels)

    def forward(self, X, La, U):
        n = La.size(0)
        # Calculate split points for num_experts groups
        split_points = torch.linspace(0, n, self.num_experts + 1, dtype=torch.long)
        
        expert_outputs = []
        stats = []
        
        for i in range(self.num_experts):
            start_idx, end_idx = split_points[i], split_points[i+1]
            La_group = La[start_idx:end_idx]
            U_group = U[:, start_idx:end_idx]
            
            # Apply expert to get importance weights
            V = self.experts[i](U_group)
            # Create eigen-graph for this group
            A_exp = torch.mm(torch.mm(U_group, torch.diag(V)), U_group.transpose(0, 1))
            expert_outputs.append(A_exp)
            # Collect summary statistic (mean eigenvalue)
            stats.append(La_group.mean())
        
        # Prepare gating input and compute weights
        stats_tensor = torch.stack(stats).unsqueeze(0)  # [1, num_experts]
        gate_weights = self.gate(stats_tensor)  # [1, num_experts]
        #print(X)
        print(gate_weights)
        
        # Combine eigen-graphs using gating weights
        out_A2 = torch.zeros_like(expert_outputs[0])
        for i, weight in enumerate(gate_weights[0]):
            out_A2 += weight * expert_outputs[i]
        
        # Apply combined eigen-graph to input features
        out = torch.mm(out_A2, X)
        out = self.W(out)
        hidden_emd = out
        
        # Conditional batch norm
        if not config.small_graph and config.dataset not in []:
            out = self.bn(out)
            
        out = self.act(out)
        out = self.dropout(out)
        out = self.MLP(out)
        
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.sigmoid(out), hidden_emd
    
class SLOG_B__(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(Morgan, self).__init__()
        self.ve = Velocity(7)
        self.ve2 = Velocity(0)
        self.W = nn.Linear(in_channels, hidden_dim)
        self.softmax = softmax
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MLP = nn.Linear(hidden_dim, out_channels)

    def forward(self, X, La, U):
        V_La = self.ve(La)
        out_A = torch.mm(torch.mm(U, torch.diag(V_La)), torch.transpose(U, 0, 1))
        V_La2 = self.ve2((2 * (La - 0.00000001) - 1) ** 2 + 1)
        out_A2 = torch.mm(torch.mm(U, torch.diag(V_La2)), torch.transpose(U, 0, 1))
        out = torch.mm(out_A, X)
        out = torch.mm(out_A2, out)
        out = self.W(out)
        hidden_emd = out
        if not config.small_graph and config.dataset not in []:
            out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.MLP(out)
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.nn.Sigmoid()(out), hidden_emd


class SLOG_wo_S1(nn.Module): # for ablation study
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(SLOG_wo_S1, self).__init__()
        self.ve = Velocity(7)
        self.ve2 = Velocity(0)
        self.W = nn.Linear(in_channels, hidden_dim)
        self.softmax = softmax
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MLP = nn.Linear(hidden_dim, out_channels)

    def forward(self, X, La, U):
        V_La2 = self.ve2((2 * (La - 0.00000001) - 1) ** 2 + 1)
        out_A2 = torch.mm(torch.mm(U, torch.diag(V_La2)), torch.transpose(U, 0, 1))
        out = torch.mm(out_A2, X)
        out = self.W(out)
        hidden_emd = out
        if not config.small_graph and config.dataset not in []:
            out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.MLP(out)
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.nn.Sigmoid()(out), hidden_emd

class SLOG_wo_S2(nn.Module): # for ablation study
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(SLOG_wo_S2, self).__init__()
        self.ve = Velocity(7)
        self.ve2 = Velocity(0)
        self.W = nn.Linear(in_channels, hidden_dim)
        self.softmax = softmax
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MLP = nn.Linear(hidden_dim, out_channels)

    def forward(self, X, La, U):
        V_La = self.ve(La)
        out_A = torch.mm(torch.mm(U, torch.diag(V_La)), torch.transpose(U, 0, 1))
        out = torch.mm(out_A, X)
        out = self.W(out)
        hidden_emd = out
        if not config.small_graph and config.dataset not in []:
            out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.MLP(out)
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.nn.Sigmoid()(out), hidden_emd

class SLOG_N(nn.Module):
    def __init__(
        self, in_channels, hidden_dim, out_channels, layer_num=3, softmax=True, res=True
    ):
        super(SLOG_N, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.bns1 = nn.ModuleList()
        self.bns2 = nn.ModuleList()
        self.tedgcns = nn.ModuleList()
        self.lines = nn.ModuleList()
        self.tedgcns.append(Morgan(in_channels, hidden_dim, hidden_dim, False))
        if layer_num == 1:
            self.lines.append(nn.Linear(in_channels, out_channels))
        else:
            self.lines.append(nn.Linear(in_channels, hidden_dim))
        for i in range(layer_num - 2):
            self.lines.append(nn.Linear(hidden_dim, hidden_dim))
            self.tedgcns.append(Morgan(hidden_dim, hidden_dim, hidden_dim, False))
        for i in range(layer_num - 1):
                self.bns1.append(nn.BatchNorm1d(hidden_dim))
                self.bns2.append(nn.BatchNorm1d(hidden_dim))
        self.bns1.append(nn.BatchNorm1d(hidden_dim))
        self.tedgcns.append(Morgan(hidden_dim, hidden_dim, out_channels, False))
        self.lines.append(nn.Linear(hidden_dim, out_channels))
        self.act = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.MLP = nn.Linear(hidden_dim, out_channels)
        self.softmax = softmax
        self.layer_num = layer_num
        self.res = res

    def forward(self, X, La, U, edge_weight=None):
        for i in range(self.layer_num - 1):
            out, hidden_emd = self.tedgcns[i](X, La, U)
            hidden_emd = self.bns1[i](hidden_emd)
            hidden_emd = self.act(hidden_emd)
            hidden_emd = self.dropout(hidden_emd)
            if self.res:
                X = self.act(hidden_emd + self.lines[i](X))
                X = self.bns2[i](self.dropout(X))
            else:
                X = hidden_emd
            
        out, hidden_emd = self.tedgcns[self.layer_num - 1](X, La, U)
        hidden_emd = self.bns1[self.layer_num - 1](hidden_emd)
        hidden_emd = self.act(hidden_emd)
        hidden_emd = self.dropout(hidden_emd)
        if self.res:
            X = self.act2(self.MLP(hidden_emd)) + self.lines[self.layer_num - 1](X)
        else:
            X = self.act2(self.MLP(hidden_emd))

        if self.softmax:
            return F.log_softmax(X, dim=1), None
        return X, None

class SLOG_B_gp(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, softmax=True):
        super(SLOG_B_gp, self).__init__()
        self.W = nn.Linear(in_channels, hidden_dim)
        self.MLP = nn.Linear(hidden_dim, out_channels)
        self.softmax = softmax
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.p = 0.1

    def forward(self, X, La, U, d1, d2):
        # V_La = self.ve(La)
        V_La = torch.pow(La, d1)
        out_A = torch.mm(torch.mm(U, torch.diag(V_La)), torch.transpose(U, 0, 1))
        
        V_La2 = torch.pow((2 * (La - 0.00000001) - 1) ** 2 + 1, d2)


        out_A2 = torch.mm(torch.mm(U, torch.diag(V_La2)), torch.transpose(U, 0, 1))
        out = torch.mm(out_A, X)
        out = torch.mm(out_A2, out)
        out = self.W(out)
        out = torch.relu(out)
        out = F.dropout(out, training=self.training, p=self.p)
        hidden_emd = out
        out = self.MLP(out)
        if self.softmax:
            return F.log_softmax(out, dim=1), hidden_emd
        return torch.nn.Sigmoid()(out), hidden_emd