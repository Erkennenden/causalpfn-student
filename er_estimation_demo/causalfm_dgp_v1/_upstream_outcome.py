"""OutcomeGenerator extracted unchanged from CausalFM-toolkit standard.py.

Apache-2.0; see LICENSE and SOURCE_MANIFEST.json. Only the standalone imports
and this header are added. The network wrapper uses its graph/parameter
sampling; generate_outcomes() is intentionally not called.
"""
from typing import Callable, Dict, Optional, Tuple
import networkx as nx
import numpy as np


class OutcomeGenerator:
    """
    Generates potential outcomes Y(0) and Y(1) based on covariates and treatment.
    
    Uses an MLP-like network to model the outcome function.
    """
    
    def __init__(
        self,
        prior_layers: Callable[[], int] = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable[[], int] = lambda: np.random.randint(8, 30),
        prior_weight: Callable[[], float] = lambda: np.random.normal(0, 1),
        edge_drop_prob: float = 0.3,
        activation: Callable = lambda x: np.tanh(x)
    ):
        """
        Initialize the outcome generator.
        
        Args:
            prior_layers: Callable returning the number of layers
            prior_hidden_size: Callable returning the hidden size
            prior_weight: Callable returning edge weights
            edge_drop_prob: Probability of dropping edges
            activation: Activation function (default: tanh)
        """
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.edge_drop_prob = edge_drop_prob
        self.activation = activation
        self.outcome_network: Optional[nx.DiGraph] = None
        self.weights: Dict = {}
        self.biases: Dict = {}
        self.noise_distributions: Dict = {}
        self.output_node: Optional[int] = None
        
    def sample_noise_distribution(self) -> Callable:
        """Sample a noise distribution from a meta-distribution."""
        dist_type = np.random.choice(["normal", "uniform", "laplace", "logistic"])
        
        if dist_type == "normal":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.normal(0, scale, size)
        elif dist_type == "uniform":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.uniform(-scale, scale, size)
        elif dist_type == "laplace":
            scale = np.random.uniform(0.1, 0.5)
            return lambda size: np.random.laplace(0, scale, size)
        else:  # logistic
            scale = np.random.uniform(0.1, 0.5)
            return lambda size: np.random.logistic(0, scale, size)
    
    def construct_outcome_network(
        self, num_layers: int, hidden_size: int, input_size: int
    ) -> nx.DiGraph:
        """Construct an MLP-like network for outcome generation."""
        G = nx.DiGraph()
        node_id = 0
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [1]
        nodes_by_layer = []
        
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
                G.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        
        self.output_node = nodes_by_layer[-1][0]
        
        for layer_idx in range(num_layers - 1):
            for src in nodes_by_layer[layer_idx]:
                for dst in nodes_by_layer[layer_idx + 1]:
                    G.add_edge(src, dst)
        
        # Drop edges (except to output)
        edges = list(G.edges())
        num_edges_to_drop = int(len(edges) * self.edge_drop_prob)
        
        if num_edges_to_drop > 0:
            edges_to_drop = np.random.choice(
                len(edges), size=num_edges_to_drop, replace=False
            )
            for idx in edges_to_drop:
                u, v = edges[idx]
                if v != self.output_node:
                    G.remove_edge(u, v)
        
        return G
    
    def sample_outcome_network_parameters(self) -> None:
        """Sample weights, biases, and noise distributions."""
        for node in self.outcome_network.nodes():
            parents = list(self.outcome_network.predecessors(node))
            if parents:
                self.weights[node] = {
                    parent: self.prior_weight() for parent in parents
                }
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()
    
    def forward_propagate(
        self, z: np.ndarray, noise_samples: Dict[int, float]
    ) -> Dict[int, float]:
        """Forward propagate through the network."""
        node_values = {}
        node_layers = nx.get_node_attributes(self.outcome_network, 'layer')
        max_layer = max(node_layers.values())
        
        for layer in range(max_layer + 1):
            layer_nodes = [n for n, l in node_layers.items() if l == layer]
            for node in layer_nodes:
                parents = list(self.outcome_network.predecessors(node))
                
                if layer == 0:
                    node_values[node] = z[node]
                else:
                    if parents:
                        weighted_sum = sum(
                            self.weights[node][parent] * node_values[parent] 
                            for parent in parents
                        )
                    else:
                        weighted_sum = 0
                    
                    noise = noise_samples[node]
                    value = weighted_sum + self.biases[node] + noise
                    if node != self.output_node:
                        value = self.activation(value)
                    node_values[node] = value
        
        return node_values
    
    def generate_outcomes(
        self, X: np.ndarray, A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate potential outcomes.
        
        Args:
            X: Covariate matrix of shape (n_samples, n_features)
            A: Treatment array
            
        Returns:
            Tuple of (Y, Y0, Y1) - observed and potential outcomes
        """
        n_samples, n_features = X.shape
        
        num_layers = max(3, self.prior_layers())
        hidden_size = self.prior_hidden_size()
        
        self.outcome_network = self.construct_outcome_network(
            num_layers, hidden_size, n_features + 1
        )
        self.sample_outcome_network_parameters()
        
        Y = np.zeros(n_samples, dtype=float)
        Y0 = np.zeros(n_samples, dtype=float)
        Y1 = np.zeros(n_samples, dtype=float)
        
        for i in range(n_samples):
            noise_samples = {
                node: self.noise_distributions[node](1)[0]
                for node in self.outcome_network.nodes()
            }
            
            # Outcome under A=0
            z0 = np.concatenate([X[i], [0]])
            node_values0 = self.forward_propagate(z0, noise_samples)
            Y0[i] = node_values0[self.output_node]
            
            # Outcome under A=1
            z1 = np.concatenate([X[i], [1]])
            node_values1 = self.forward_propagate(z1, noise_samples)
            Y1[i] = node_values1[self.output_node]
            
            # Factual outcome
            Y[i] = Y0[i] if A[i] == 0 else Y1[i]
        
        return Y, Y0, Y1

