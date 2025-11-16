import torch
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.Utils import NeuralNet


class GraphAttentionLayer(NeuralNet):
    """
    Single Graph Attention Layer for message passing between agents.
    Implements attention mechanism: α_ij = softmax(LeakyReLU(a^T [W z_i || W z_j]))
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Weight matrix W for linear transformation
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters: a^T [W z_i || W z_j]
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.init()
    
    def init(self):
        # Optimizer will be set by parent class
        pass
    
    def forward(self, z, adj_mask=None):
        """
        Forward pass of GAT layer.
        
        Args:
            z: Tensor of shape (n_agents, batch_size, in_features) - agent embeddings
            adj_mask: Optional adjacency mask of shape (n_agents, n_agents) for k-NN graph
                     If None, uses fully connected graph
        
        Returns:
            Updated embeddings: (n_agents, batch_size, out_features)
            Attention weights: (n_agents, n_agents, batch_size) for visualization
        """
        n_agents = z.shape[0]
        batch_size = z.shape[1]
        
        # Linear transformation: W z
        h = torch.matmul(z, self.W)  # (n_agents, batch_size, out_features)
        
        # Compute attention coefficients for all pairs
        # h_i: (batch_size, out_features) for each agent
        attention_scores = []
        for i in range(n_agents):
            h_i = h[i]  # (batch_size, out_features)
            scores_i = []
            for j in range(n_agents):
                h_j = h[j]  # (batch_size, out_features)
                # Concatenate: [W z_i || W z_j]
                h_concat = torch.cat([h_i, h_j], dim=-1)  # (batch_size, 2*out_features)
                # Compute attention: a^T [W z_i || W z_j]
                e_ij = self.leaky_relu(torch.matmul(h_concat, self.a)).squeeze(-1)  # (batch_size,)
                scores_i.append(e_ij)
            attention_scores.append(torch.stack(scores_i, dim=0))  # (n_agents, batch_size)
        
        attention_scores = torch.stack(attention_scores, dim=0)  # (n_agents, n_agents, batch_size)
        
        # Apply adjacency mask if provided (for k-NN graph)
        if adj_mask is not None:
            # adj_mask: (n_agents, n_agents), broadcast to (n_agents, n_agents, batch_size)
            adj_mask_expanded = adj_mask.unsqueeze(-1).expand(-1, -1, batch_size)
            attention_scores = attention_scores.masked_fill(adj_mask == 0, float('-inf'))
        
        # Softmax over neighbors
        attention_weights = F.softmax(attention_scores, dim=1)  # (n_agents, n_agents, batch_size)
        # Apply dropout before aggregation (but keep structure for testing)
        if self.training:
            attention_weights = F.dropout(attention_weights, self.dropout, training=True)
        
        # Aggregate messages: z_i^(t+1) = σ(Σ_j α_ij W z_j)
        # h: (n_agents, batch_size, out_features)
        # attention_weights: (n_agents, n_agents, batch_size)
        # For each agent i, we want: sum_j (alpha_ij * h_j) over all neighbors j
        h_updated = []
        for i in range(n_agents):
            # Get attention weights for agent i: (n_agents, batch_size)
            alpha_i = attention_weights[i]  # (n_agents, batch_size)
            # Expand for broadcasting: (n_agents, batch_size, 1)
            alpha_i_expanded = alpha_i.unsqueeze(-1)  # (n_agents, batch_size, 1)
            # Multiply and sum over neighbors (dim=0): (batch_size, out_features)
            h_weighted = torch.sum(alpha_i_expanded * h, dim=0)  # (batch_size, out_features)
            h_updated.append(h_weighted)
        
        # Stack: (n_agents, batch_size, out_features)
        h_updated = torch.stack(h_updated, dim=0)
        
        # Apply non-linearity (LeakyReLU as in paper)
        h_updated = F.leaky_relu(h_updated, negative_slope=self.alpha)
        
        return h_updated, attention_weights


class GraphAttentionNetwork(NeuralNet):
    """
    Multi-layer Graph Attention Network for iterative message passing.
    Supports T steps of message passing as specified in the formulation.
    """
    def __init__(self, embedding_dim, hidden_dim=None, num_layers=1, dropout=0.1, alpha=0.2):
        super(GraphAttentionNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.alpha = alpha
        
        # Build GAT layers
        layers = []
        in_dim = embedding_dim
        for i in range(num_layers):
            out_dim = self.hidden_dim if i < num_layers - 1 else embedding_dim
            layers.append(GraphAttentionLayer(in_dim, out_dim, dropout, alpha))
            in_dim = out_dim
        
        self.layers = nn.ModuleList(layers)
        self.init()
    
    def init(self):
        # Optimizer will be set by parent class
        pass
    
    def forward(self, z, adj_mask=None, num_steps=1):
        """
        Forward pass with T steps of message passing.
        
        Args:
            z: Initial embeddings (n_agents, batch_size, embedding_dim)
            adj_mask: Optional adjacency mask (n_agents, n_agents) for k-NN graph
            num_steps: Number of message passing iterations T
        
        Returns:
            Updated embeddings after T steps: (n_agents, batch_size, embedding_dim)
            All attention weights: List of (n_agents, n_agents, batch_size) tensors
        """
        z_current = z
        all_attention_weights = []
        
        for step in range(num_steps):
            # Pass through all GAT layers
            for layer in self.layers:
                z_current, attn_weights = layer(z_current, adj_mask)
                all_attention_weights.append(attn_weights)
            # After each step, z_current should maintain shape (n_agents, batch_size, embedding_dim)
        
        return z_current, all_attention_weights


def build_knn_adjacency_mask(n_agents, k=2, similarity_matrix=None):
    """
    Build k-NN adjacency mask for communication graph.
    
    Args:
        n_agents: Number of agents
        k: Number of nearest neighbors (including self)
        similarity_matrix: Optional (n_agents, n_agents) matrix for similarity-based k-NN
                          If None, uses distance-based k-NN (e.g., based on agent indices)
    
    Returns:
        adj_mask: (n_agents, n_agents) binary mask where 1 indicates edge
    """
    adj_mask = torch.zeros(n_agents, n_agents, dtype=torch.float32)
    
    if similarity_matrix is not None:
        # Use similarity matrix to find k-NN
        for i in range(n_agents):
            # Get k nearest neighbors (including self)
            _, topk_indices = torch.topk(similarity_matrix[i], k, dim=0)
            adj_mask[i, topk_indices] = 1.0
    else:
        # Default: k-NN based on circular distance (ring-like but with k neighbors)
        # This is a simple default; can be replaced with actual similarity metrics
        for i in range(n_agents):
            # Include self
            adj_mask[i, i] = 1.0
            # Include k-1 additional nearest neighbors (circular, one direction)
            # For k=2: self + 1 neighbor = 2 total
            # For k=3: self + 2 neighbors = 3 total
            neighbors_to_add = k - 1
            for offset in range(1, neighbors_to_add + 1):
                adj_mask[i, (i + offset) % n_agents] = 1.0
    
    return adj_mask

