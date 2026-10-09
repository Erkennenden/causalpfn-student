"""Original CausalFM class bodies; see SOURCE_MANIFEST.json and LICENSE.causalfm.
Only network construction/parameter sampling are called by the no-U adapter.
"""
from typing import Callable, Dict, Optional, Tuple
import networkx as nx
import numpy as np


class BaseMLPGenerator:
    """
    Base class for MLP-like network generators.
    
    Provides common functionality for generating treatments, mediators,
    and outcomes using MLP-based structural equations.
    """
    
    def __init__(
        self,
        prior_layers: Callable[[], int] = lambda: np.random.randint(2, 5),
        prior_hidden_size: Callable[[], int] = lambda: np.random.randint(8, 30),
        prior_weight: Callable[[], float] = lambda: np.random.normal(0, 1),
        activation: Callable[[np.ndarray], np.ndarray] = lambda x: np.tanh(x),
        use_layer_noise: bool = True,
        layer_noise_scale: float = 0.1,
    ):
        """
        Initialize the MLP generator.
        
        Args:
            prior_layers: Callable returning the number of layers
            prior_hidden_size: Callable returning the hidden size
            prior_weight: Callable returning edge weights
            activation: Activation function (default: tanh)
            use_layer_noise: Whether to add noise at each layer
            layer_noise_scale: Scale of the layer noise
        """
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.activation = activation
        self.use_layer_noise = use_layer_noise
        self.layer_noise_scale = layer_noise_scale
        
        self.network: Optional[nx.DiGraph] = None
        self.weights: Dict = {}
        self.biases: Dict = {}
        self.noise_distributions: Dict = {}
        self.output_node: Optional[int] = None

    def sample_noise_distribution(self) -> Callable:
        """Sample a noise distribution from a meta-distribution."""
        dist_type = np.random.choice(["normal", "uniform", "laplace", "logistic"])
        scale_map = {
            "normal": (0.1, 1.0),
            "uniform": (0.1, 1.0),
            "laplace": (0.1, 0.5),
            "logistic": (0.1, 0.5)
        }
        scale = np.random.uniform(*scale_map[dist_type])
        
        dist_map = {
            "normal": lambda s: np.random.normal(0, scale, s),
            "uniform": lambda s: np.random.uniform(-scale, scale, s),
            "laplace": lambda s: np.random.laplace(0, scale, s),
            "logistic": lambda s: np.random.logistic(0, scale, s)
        }
        return dist_map[dist_type]

    def _construct_network(
        self, 
        num_layers: int, 
        hidden_size: int, 
        input_size: int, 
        output_size: int
    ) -> nx.DiGraph:
        """
        Construct an MLP-like network.
        
        Args:
            num_layers: Number of layers
            hidden_size: Size of hidden layers
            input_size: Number of input features
            output_size: Number of outputs
            
        Returns:
            Directed graph representing the network
        """
        G = nx.DiGraph()
        node_id = 0
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 2) + [output_size]
        nodes_by_layer = []
        
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = [node_id + i for i in range(size)]
            nodes_by_layer.append(layer_nodes)
            for node in layer_nodes:
                G.add_node(node, layer=layer_idx)
            node_id += size

        if output_size == 1:
            self.output_node = nodes_by_layer[-1][0]
        
        for i in range(num_layers - 1):
            for src in nodes_by_layer[i]:
                for dst in nodes_by_layer[i + 1]:
                    G.add_edge(src, dst)
                    
        return G

    def _sample_network_parameters(self) -> None:
        """Sample weights, biases, and noise distributions for the network."""
        for node in self.network.nodes():
            parents = list(self.network.predecessors(node))
            if parents:
                self.weights[node] = {parent: self.prior_weight() for parent in parents}
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()

    def _forward_propagate(self, z: np.ndarray) -> float:
        """
        Forward propagate input through the network.
        
        Args:
            z: Input array
            
        Returns:
            Output value
        """
        node_values = {}
        node_layers = nx.get_node_attributes(self.network, 'layer')
        
        for i in range(len(z)):
            node_values[i] = z[i]
            
        for layer in sorted(list(set(node_layers.values())))[1:]:
            for node in [n for n, l in node_layers.items() if l == layer]:
                parents = list(self.network.predecessors(node))
                weighted_sum = sum(
                    self.weights[node][p] * node_values[p] for p in parents
                )

                if self.use_layer_noise:
                    noise = np.random.normal(0, self.layer_noise_scale)
                else:
                    noise = 0.0
                
                value = weighted_sum + self.biases[node] + noise
                if node != self.output_node:
                    value = self.activation(value)
                node_values[node] = value
                
        return node_values[self.output_node]
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
