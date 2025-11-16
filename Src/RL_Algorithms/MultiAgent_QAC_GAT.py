import numpy as np
import torch
from torch import tensor, float32
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.Utils import NeuralNet
from Src.RL_Algorithms.QAC_C2DMapping import QAC_C2DMapping
from Src.Utils import Basis, Actor, Critic
from Src.Utils.GAT import GraphAttentionNetwork, build_knn_adjacency_mask


class IntentEmbeddingNetwork(NeuralNet):
    """
    Separate embedding network that takes state and outputs intent embedding z_i.
    This network learns to encode agent's intent/belief over optimal future actions.
    """
    def __init__(self, state_dim, embedding_dim, hidden_dim=64, config=None):
        super(IntentEmbeddingNetwork, self).__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Network architecture: state -> hidden -> embedding
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        
        self.init()
    
    def init(self):
        if self.config is not None:
            # Use embedding learning rate if available, otherwise use state_lr
            lr = getattr(self.config, 'embedding_lr', getattr(self.config, 'state_lr', 1e-3))
            self.optim = self.config.optim(self.parameters(), lr=lr)
        else:
            # Default optimizer (will be set later)
            self.optim = None
    
    def forward(self, state):
        """
        Forward pass: state -> intent embedding z_i
        
        Args:
            state: (batch_size, state_dim) tensor
        
        Returns:
            embedding: (batch_size, embedding_dim) tensor
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        z = self.fc3(x)  # No activation on final layer to allow full range
        return z


class MultiAgent_QAC_GAT(QAC_C2DMapping):
    """
    Multi-Agent QAC with Graph Attention Network for communication.
    Extends QAC_C2DMapping to support multiple agents with intent embedding exchange.
    """
    def __init__(self, config):
        # Store config first
        self.config = config
        
        # Multi-agent specific parameters
        self.n_agents = getattr(config, 'n_agents', config.env.num_machines)  # Number of agents (machines)
        self.embedding_dim = getattr(config, 'embedding_dim', 16)  # Dimension of intent embeddings
        self.gat_k = getattr(config, 'gat_k', 2)  # k for k-NN graph
        self.gat_train_steps = getattr(config, 'gat_train_steps', 2)  # T=2 during training
        self.gat_eval_steps = getattr(config, 'gat_eval_steps', 1)  # T=1 during evaluation
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.1)  # Weight for contrastive loss
        
        # Initialize base Agent class (skip QAC_C2DMapping initialization to avoid conflicts)
        from Src.RL_Algorithms.Agent import Agent
        Agent.__init__(self, config)
        
        # Initialize multi-agent components
        self._init_multi_agent_components()
    
    def _init_multi_agent_components(self):
        """Initialize multi-agent specific components"""
        # Get state feature dimension (per-agent observation space)
        # In multi-agent mode, each agent has its own observation
        per_agent_state_dim = self.config.env.observation_space.shape[0]  # Should be 4: [e_i, w_i, queue_i, global_avg]
        
        # Create a custom config for state features (per-agent observation space)
        # We need to temporarily modify the env's observation_space for Basis initialization
        original_obs_space = self.config.env.observation_space
        # Create a temporary observation space with per-agent dimensions
        from Src.Utils.Utils import Space
        temp_obs_space = Space(
            low=self.config.env.observation_space.low,
            high=self.config.env.observation_space.high,
            dtype=self.config.env.observation_space.dtype
        )
        self.config.env.observation_space = temp_obs_space
        
        # State feature extractor
        self.state_features = Basis.get_Basis(config=self.config)
        # Ensure correct dimensions
        self.state_features.state_dim = per_agent_state_dim
        self.state_features.feature_dim = per_agent_state_dim  # For now, no feature transformation
        
        # Restore original observation space
        self.config.env.observation_space = original_obs_space
        
        # Per-agent intent embedding networks
        self.embedding_networks = nn.ModuleList([
            IntentEmbeddingNetwork(
                state_dim=self.state_features.feature_dim,
                embedding_dim=self.embedding_dim,
                hidden_dim=getattr(self.config, 'embedding_hidden_dim', 64),
                config=self.config
            ) for _ in range(self.n_agents)
        ])
        
        # Graph Attention Network for communication
        self.gat = GraphAttentionNetwork(
            embedding_dim=self.embedding_dim,
            hidden_dim=getattr(self.config, 'gat_hidden_dim', self.embedding_dim),
            num_layers=1,
            dropout=getattr(self.config, 'gat_dropout', 0.1),
            alpha=getattr(self.config, 'gat_alpha', 0.2)
        )
        self.gat.init()
        self.gat.optim = self.config.optim(self.gat.parameters(), lr=getattr(self.config, 'gat_lr', 1e-3))
        
        # Build k-NN adjacency mask for communication graph
        self.adj_mask = build_knn_adjacency_mask(self.n_agents, k=self.gat_k)
        
        # Per-agent actors (Gaussian policy for continuous actions)
        self.actors = nn.ModuleList([
            Actor.Gaussian(
                action_dim=1,  # Scalar action per agent
                state_dim=self.state_features.feature_dim + self.embedding_dim,  # State + communicated embedding
                config=self.config
            ) for _ in range(self.n_agents)
        ])
        
        # Per-agent critics (Q-value networks)
        self.critics = nn.ModuleList([
            Critic.Qval(
                state_dim=self.state_features.feature_dim + self.embedding_dim,
                action_dim=1,  # Scalar action
                config=self.config
            ) for _ in range(self.n_agents)
        ])
        
        # Update modules list for saving/loading
        self.modules = [('gat', self.gat)]
        for i in range(self.n_agents):
            self.modules.append((f'embedding_net_{i}', self.embedding_networks[i]))
            self.modules.append((f'actor_{i}', self.actors[i]))
            self.modules.append((f'critic_{i}', self.critics[i]))
        
        # Storage for embeddings (for contrastive loss)
        self.prev_embeddings = None
        
        self.weights_changed = True
    
    def _extract_intent_embeddings(self, observations):
        """
        Extract intent embeddings from observations for all agents.
        
        Args:
            observations: List of n_agents observations, each is (state_dim,) array
        
        Returns:
            embeddings: (n_agents, 1, embedding_dim) tensor
        """
        embeddings = []
        for i, obs in enumerate(observations):
            obs_tensor = tensor(obs, dtype=float32).unsqueeze(0)  # (1, state_dim)
            # Get state features
            state_features = self.state_features.forward(obs_tensor)  # (1, feature_dim)
            # Get intent embedding
            z_i = self.embedding_networks[i](state_features)  # (1, embedding_dim)
            embeddings.append(z_i)
        
        # Stack: (n_agents, 1, embedding_dim)
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings
    
    def _communicate_embeddings(self, embeddings, training=True):
        """
        Perform message passing through GAT.
        
        Args:
            embeddings: (n_agents, batch_size, embedding_dim) tensor
            training: Whether in training mode (affects T steps)
        
        Returns:
            communicated_embeddings: (n_agents, batch_size, embedding_dim) tensor
        """
        num_steps = self.gat_train_steps if training else self.gat_eval_steps
        communicated_embeddings, _ = self.gat(embeddings, adj_mask=self.adj_mask, num_steps=num_steps)
        return communicated_embeddings
    
    def get_action(self, observations, training=True):
        """
        Get actions for all agents.
        
        Args:
            observations: List of n_agents observations
            training: Whether in training mode
        
        Returns:
            actions: List of n_agents scalar actions
            embeddings: (n_agents, embedding_dim) numpy array (for logging)
        """
        # Extract intent embeddings
        embeddings = self._extract_intent_embeddings(observations)  # (n_agents, 1, embedding_dim)
        
        # Communicate through GAT
        communicated_embeddings = self._communicate_embeddings(embeddings, training=training)  # (n_agents, 1, embedding_dim)
        
        # Store embeddings for contrastive loss
        if training:
            self.prev_embeddings = embeddings.detach()
        
        # Get actions for each agent
        actions = []
        for i, obs in enumerate(observations):
            obs_tensor = tensor(obs, dtype=float32).unsqueeze(0)  # (1, state_dim)
            state_features = self.state_features.forward(obs_tensor)  # (1, feature_dim)
            # Concatenate state features with communicated embedding
            comm_emb_i = communicated_embeddings[i].squeeze(0)  # (1, embedding_dim)
            actor_input = torch.cat([state_features, comm_emb_i], dim=-1)  # (1, feature_dim + embedding_dim)
            
            # Get action from actor
            a_hat, _ = self.actors[i].get_action(actor_input, training=training)
            # Action is already scalar, just extract value
            action = a_hat.cpu().item()
            actions.append(action)
        
        # Convert embeddings to numpy for return
        embeddings_np = communicated_embeddings.squeeze(1).cpu().data.numpy()  # (n_agents, embedding_dim)
        
        return actions, embeddings_np
    
    def _compute_contrastive_loss(self, embeddings_t, embeddings_t1):
        """
        Compute InfoNCE contrastive loss.
        
        Positive pairs: (z_i(t), z_i(t+1)) - same agent, consecutive timesteps
        Negative pairs: (z_i(t), z_j(t)) - different agents, same timestep
        
        Args:
            embeddings_t: (n_agents, embedding_dim) - embeddings at time t
            embeddings_t1: (n_agents, embedding_dim) - embeddings at time t+1
        
        Returns:
            contrastive_loss: scalar tensor
        """
        n_agents = embeddings_t.shape[0]
        
        # Normalize embeddings
        embeddings_t = F.normalize(embeddings_t, p=2, dim=1)
        embeddings_t1 = F.normalize(embeddings_t1, p=2, dim=1)
        
        # Compute similarity matrix: (n_agents, n_agents)
        # similarity[i, j] = dot(z_i(t), z_j(t+1))
        similarity_matrix = torch.matmul(embeddings_t, embeddings_t1.t())  # (n_agents, n_agents)
        
        # Positive pairs: diagonal elements (z_i(t), z_i(t+1))
        positive_similarities = torch.diag(similarity_matrix)  # (n_agents,)
        
        # Negative pairs: off-diagonal elements (z_i(t), z_j(t+1)) for j != i
        # For each agent i, negatives are all j != i
        losses = []
        temperature = 0.1  # Temperature parameter for InfoNCE
        
        for i in range(n_agents):
            # Positive: z_i(t) with z_i(t+1)
            pos = positive_similarities[i] / temperature
            
            # Negatives: z_i(t) with z_j(t+1) for all j
            # Use all similarities in row i (including positive, but we'll subtract it)
            logits = similarity_matrix[i] / temperature  # (n_agents,)
            
            # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum_j exp(neg_j)))
            # = -pos + log(exp(pos) + sum_j exp(neg_j))
            loss_i = -pos + torch.logsumexp(logits, dim=0)
            losses.append(loss_i)
        
        contrastive_loss = torch.mean(torch.stack(losses))
        return contrastive_loss
    
    def update(self, observations, actions, embeddings, rewards, next_observations, done):
        """
        Update all agents' policies and critics.
        
        Args:
            observations: List of n_agents observations at time t
            actions: List of n_agents actions (scalars)
            embeddings: (n_agents, embedding_dim) - embeddings used for actions
            rewards: List of n_agents rewards
            next_observations: List of n_agents observations at time t+1
            done: Boolean indicating episode end
        
        Returns:
            loss_actor: Average actor loss across agents
            loss_critic: Average critic loss across agents
            loss_contrastive: Contrastive loss
        """
        # Convert to tensors
        obs_tensors = [tensor(obs, dtype=float32).unsqueeze(0) for obs in observations]
        next_obs_tensors = [tensor(obs, dtype=float32).unsqueeze(0) for obs in next_observations]
        action_tensors = [tensor([a], dtype=float32) for a in actions]
        reward_tensors = [tensor([r], dtype=float32) for r in rewards]
        
        # Extract embeddings for current and next states
        embeddings_t = self._extract_intent_embeddings(observations).squeeze(1)  # (n_agents, embedding_dim)
        embeddings_t1 = self._extract_intent_embeddings(next_observations).squeeze(1)  # (n_agents, embedding_dim)
        
        # Compute contrastive loss
        loss_contrastive = self._compute_contrastive_loss(embeddings_t, embeddings_t1)
        
        # Compute policy and critic losses for each agent
        actor_losses = []
        critic_losses = []
        
        for i in range(self.n_agents):
            # Get state features
            state_features = self.state_features.forward(obs_tensors[i])  # (1, feature_dim)
            next_state_features = self.state_features.forward(next_obs_tensors[i])  # (1, feature_dim)
            
            # Get communicated embeddings (for actor/critic input)
            embeddings_batch = embeddings_t.unsqueeze(1)  # (n_agents, 1, embedding_dim)
            comm_embeddings = self._communicate_embeddings(embeddings_batch, training=True)
            comm_emb_i = comm_embeddings[i].squeeze(0)  # (1, embedding_dim)
            next_comm_emb_i = comm_embeddings[i].squeeze(0)  # Same for simplicity (could recompute)
            
            # Actor input: state + communicated embedding
            actor_input = torch.cat([state_features, comm_emb_i], dim=-1)  # (1, feature_dim + embedding_dim)
            next_actor_input = torch.cat([next_state_features, next_comm_emb_i], dim=-1)
            
            # Critic: Q(s, a)
            critic_input = torch.cat([state_features, comm_emb_i], dim=-1)
            action_normalized = action_tensors[i] / float(self.config.env.max_jobs_per_machine)  # Normalize action
            q_value = self.critics[i].forward(critic_input, action_normalized.unsqueeze(0))
            
            # Next Q-value
            next_action, _ = self.actors[i].get_action(next_actor_input, training=True)
            next_action_normalized = next_action / float(self.config.env.max_jobs_per_machine)
            next_q_value = self.critics[i].forward(next_actor_input, next_action_normalized).detach()
            
            # TD target
            target_q = reward_tensors[i] + self.config.gamma * next_q_value * (1 - int(done))
            
            # Critic loss
            loss_critic_i = F.huber_loss(q_value, target_q)
            critic_losses.append(loss_critic_i)
            
            # Actor loss (policy gradient)
            log_prob, _ = self.actors[i].get_log_prob(actor_input, action_tensors[i].unsqueeze(0))
            td_error = (target_q - q_value).detach()
            loss_actor_i = -torch.sum(td_error * log_prob)
            actor_losses.append(loss_actor_i)
        
        # Average losses
        loss_actor = torch.mean(torch.stack(actor_losses))
        loss_critic = torch.mean(torch.stack(critic_losses))
        
        # Total loss
        total_loss = loss_actor + loss_critic + self.contrastive_weight * loss_contrastive
        
        # Backward pass
        self.clear_gradients()
        total_loss.backward()
        
        # Update all modules
        for _, module in self.modules:
            if hasattr(module, 'step'):
                module.step(clip_norm=1)
        
        return loss_actor.cpu().data.numpy(), loss_critic.cpu().data.numpy(), loss_contrastive.cpu().data.numpy()
    
    def reset(self):
        """Reset agent state (e.g., clear previous embeddings)"""
        self.prev_embeddings = None
        for _, module in self.modules:
            if hasattr(module, 'reset'):
                module.reset()
    
    def save(self):
        """Save all agent models"""
        if self.config.save_model:
            for name, module in self.modules:
                module.save(self.config.paths['checkpoint'] + name + '.pt')

