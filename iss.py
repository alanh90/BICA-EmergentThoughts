"""
Intrinsic Scenario Synthesis (ISS) Research Framework

This implementation provides a flexible architecture for exploring subconscious-like
processing in AI systems through multiple processing layers. The system is designed
for research experimentation while maintaining computational efficiency.

Key Features:
- Dynamic 3D importance space for noise and memory integration
- Flexible latent representation learning
- Continuous scenario generation and evaluation
- Adaptive feedback integration
"""
import time
from dataclasses import dataclass
from collections import deque
from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import logging

# For tokenizer and model
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ISSConfig:
    """Configuration class for the ISS (Intrinsic Scenario Synthesis) system"""
    # Architecture dimensions
    hidden_size: int = 768
    latent_dim: int = 256
    importance_dim: int = 32
    batch_size: int = 1  # Adjusted for single text input

    # Memory parameters
    max_memories: int = 1000
    temporal_decay: float = 0.1
    importance_threshold: float = 0.3

    # Processing parameters
    scan_size: int = 64
    checkpoint_size: int = 8
    device: str = 'cuda'

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.importance_dim > self.latent_dim:
            raise ValueError("importance_dim cannot be larger than latent_dim")
        if self.scan_size < 1:
            raise ValueError("scan_size must be positive")
        if self.checkpoint_size < 0:
            raise ValueError("checkpoint_size cannot be negative")
        if self.temporal_decay < 0:
            raise ValueError("temporal_decay must be non-negative")
        if not 0 <= self.importance_threshold <= 1:
            raise ValueError("importance_threshold must be between 0 and 1")


@dataclass
class ImportanceSpace:
    """
    Represents the 3D importance space where latent representations evolve over time.
    The third dimension tracks importance values that shift dynamically based on context.
    """
    values: torch.Tensor  # [batch, width, depth, importance]
    temporal_scale: float = 0.1
    importance_threshold: float = 0.3

    def update(self, positions: torch.Tensor, importances: torch.Tensor):
        """Update importance values while maintaining temporal consistency"""
        # Decay existing importance values
        self.values = self.values * (1 - self.temporal_scale)

        # Update with new importance values
        mask = importances > self.importance_threshold
        self.values[positions[mask]] = importances[mask]


class NoiseLayer(nn.Module):
    """
    Layer 1: Noise and Memory Integration

    This layer creates meaningful noise patterns by combining random variations with
    learned importance patterns. It maintains temporal consistency through a memory
    mechanism and processes input states through a two-stage projection pipeline.
    """

    def __init__(self, config: ISSConfig):
        super().__init__()
        self.config = config
        self.importance_dim = config.importance_dim
        self.latent_dim = config.latent_dim

        # Two-stage projection pipeline
        self.hidden_proj = nn.Linear(config.hidden_size, self.latent_dim)
        self.importance_proj = nn.Linear(self.latent_dim, self.importance_dim)

        # Initialize importance space
        self.register_buffer(
            'importance_space',
            torch.zeros(config.batch_size, self.latent_dim,
                        self.latent_dim, self.importance_dim)
        )

        self.alpha = nn.Parameter(torch.tensor(0.7))
        self.prev_noise = None

    def forward(self, input_states: torch.Tensor, memory_states: Optional[torch.Tensor] = None):
        """
        Generate noise with meaningful patterns based on input and memory states.

        Args:
            input_states: Current input tensor [batch_size, seq_len, hidden_size]
            memory_states: Optional tensor of relevant memories

        Returns:
            Noise tensor with semantic meaning [batch_size, seq_len, latent_dim, importance_dim]
        """
        batch_size, seq_len = input_states.shape[:2]

        # Generate base noise matching sequence length
        base_noise = self._generate_base_noise(batch_size, seq_len)

        # Extract importance patterns
        importance_patterns = self._extract_importance(input_states)

        # Integrate memory if available
        if memory_states is not None:
            memory_importance = self._extract_importance(memory_states)
            importance_patterns = self._combine_importance(
                importance_patterns, memory_importance)
        else:
            memory_importance = None

        # Apply importance patterns to noise
        noise = self._apply_importance(base_noise, importance_patterns)

        return noise

    def _generate_base_noise(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Generate base noise matching input sequence dimensions"""
        current_noise = torch.randn(
            batch_size, seq_len, self.latent_dim,
            device=self.importance_space.device
        )

        if self.prev_noise is not None:
            # Maintain partial continuity with previous noise
            current_noise = (
                    self.alpha * current_noise +
                    (1 - self.alpha) * self.prev_noise[:, -seq_len:]
            )

        self.prev_noise = current_noise.detach()
        return current_noise

    def _extract_importance(self, states: torch.Tensor) -> torch.Tensor:
        """Extract importance patterns from input states"""
        # Project through both layers
        latent = self.hidden_proj(states)
        importance = self.importance_proj(latent)
        return torch.sigmoid(importance)

    def _combine_importance(self, current: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Combine current and memory importance patterns"""
        gate = torch.sigmoid(self.importance_proj(self.hidden_proj(current)))
        combined = gate * current + (1 - gate) * memory
        return combined

    def _apply_importance(self, noise: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        """
        Apply importance patterns to noise tensor.

        Args:
            noise: [batch_size, seq_len, latent_dim]
            importance: [batch_size, seq_len, importance_dim]

        Returns:
            [batch_size, seq_len, latent_dim, importance_dim]
        """
        # Reshape noise and importance for broadcasting
        noise = noise.unsqueeze(-1)  # [batch_size, seq_len, latent_dim, 1]
        importance = importance.unsqueeze(2)  # [batch_size, seq_len, 1, importance_dim]

        # Multiply with broadcasting
        shaped_noise = noise * importance
        return shaped_noise


class ScenarioGenerator(nn.Module):
    """
    Layer 3: Hypothetical Scenario Generation

    This layer transforms noise patterns into concrete scenarios while maintaining
    contextual relationships. It uses an efficient scanning mechanism to process
    long sequences in manageable chunks.
    """

    def __init__(self, config: ISSConfig):
        super().__init__()
        self.config = config

        # Project context to latent space for processing
        self.context_encoder = nn.Linear(config.hidden_size, config.latent_dim)

        # Project combined features back to scenario space
        # Note: Input size accounts for noise_patterns, context, and carry state
        combined_dim = config.latent_dim * config.importance_dim + config.latent_dim + config.hidden_size
        self.scenario_decoder = nn.Linear(combined_dim, config.hidden_size)

        # Processing parameters
        self.scan_size = config.scan_size
        self.checkpoint_size = config.checkpoint_size

    def forward(
            self,
            noise_patterns: torch.Tensor,  # [batch, seq_len, latent_dim, importance_dim]
            context_states: torch.Tensor,  # [batch, seq_len, hidden_size]
            importance_values: torch.Tensor  # [batch, seq_len, importance_dim]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate scenarios from noise patterns while maintaining context relationships.
        """
        batch_size, seq_len = context_states.shape[:2]

        # Encode context into latent space
        context_encoded = self.context_encoder(context_states)  # [batch, seq_len, latent_dim]

        # Prepare for chunk processing
        num_chunks = (seq_len + self.scan_size - 1) // self.scan_size
        scenarios_list = []
        metadata_list = []

        # Initialize carry state
        carry_state = self._init_carry_state(batch_size)

        # Process chunks
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.scan_size
            end_idx = min(start_idx + self.scan_size, seq_len)

            # Get chunk of data
            noise_chunk = noise_patterns[:, start_idx:end_idx]  # [batch, chunk_size, latent_dim, importance_dim]
            context_chunk = context_encoded[:, start_idx:end_idx]  # [batch, chunk_size, latent_dim]
            importance_chunk = importance_values[:, start_idx:end_idx]  # [batch, chunk_size, importance_dim]

            # Generate scenarios for chunk
            scenarios, new_carry_state = self._generate_chunk(
                noise_chunk, context_chunk, importance_chunk, carry_state
            )

            carry_state = new_carry_state

            scenarios_list.append(scenarios)
            metadata_list.append(self._compute_metadata(scenarios))

        # Combine results
        scenarios = torch.cat(scenarios_list, dim=1)
        return scenarios, metadata_list

    def _generate_chunk(
            self,
            noise_chunk: torch.Tensor,
            context_chunk: torch.Tensor,
            importance_chunk: torch.Tensor,
            carry_state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate scenarios for a chunk of data by combining noise, context, and state.
        """
        batch_size, chunk_size = noise_chunk.shape[:2]

        # Reshape noise patterns to combine latent and importance dimensions
        noise_reshaped = noise_chunk.reshape(batch_size, chunk_size, -1)

        # Prepare carry state expansion
        carry_hidden = carry_state['hidden'].unsqueeze(1).expand(-1, chunk_size, -1)

        # Combine all features
        combined = torch.cat([
            noise_reshaped,  # [batch, chunk_size, latent_dim * importance_dim]
            context_chunk,  # [batch, chunk_size, latent_dim]
            carry_hidden  # [batch, chunk_size, hidden_size]
        ], dim=-1)

        # Generate scenarios
        scenarios = self.scenario_decoder(combined)

        # Update carry state
        new_carry = {
            'hidden': scenarios[:, -1],  # Last timestep's output
            'importance': importance_chunk[:, -1]  # Last timestep's importance
        }

        return scenarios, new_carry

    def _init_carry_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Initialize carry state for scanning"""
        return {
            'hidden': torch.zeros(
                batch_size, self.config.hidden_size,
                device=self.config.device
            ),
            'importance': torch.zeros(
                batch_size, self.config.importance_dim,
                device=self.config.device
            )
        }

    def _compute_metadata(self, scenarios: torch.Tensor) -> Dict[str, Any]:
        """Compute metadata for generated scenarios"""
        return {
            'complexity': self._estimate_complexity(scenarios),
            'coherence': self._estimate_coherence(scenarios),
            'novelty': self._estimate_novelty(scenarios)
        }

    def _estimate_complexity(self, scenarios: torch.Tensor) -> torch.Tensor:
        """Estimate scenario complexity"""
        return torch.norm(scenarios, dim=-1).mean()

    def _estimate_coherence(self, scenarios: torch.Tensor) -> torch.Tensor:
        """Estimate scenario coherence"""
        diffs = scenarios[:, 1:] - scenarios[:, :-1]
        return -torch.norm(diffs, dim=-1).mean()

    def _estimate_novelty(self, scenarios: torch.Tensor) -> torch.Tensor:
        """Estimate scenario novelty"""
        return torch.std(scenarios, dim=1).mean()


class ScenarioEvaluator(nn.Module):
    """
    Layer 4: Scenario Evaluation and Ranking

    Evaluates generated scenarios based on multiple criteria including
    risk assessment, benefit analysis, and coherence checking.
    """

    def __init__(self, config: ISSConfig):
        super().__init__()
        self.config = config

        # Evaluation networks for different criteria
        self.risk_evaluator = self._build_evaluator()
        self.benefit_evaluator = self._build_evaluator()
        self.coherence_evaluator = self._build_evaluator()

        # Importance weighting for different evaluation criteria
        self.importance_weights = nn.Parameter(torch.ones(3))

    def _build_evaluator(self) -> nn.Module:
        """Build evaluation network with consistent dimensions"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.latent_dim),
            nn.ReLU(),
            nn.Linear(self.config.latent_dim, 1)
        )

    def forward(
            self,
            scenarios: torch.Tensor,  # [batch_size, seq_len, hidden_size]
            context_states: torch.Tensor,
            metadata_list: List[Dict]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate scenarios across multiple criteria.

        Args:
            scenarios: Generated scenarios to evaluate
            context_states: Original context for reference
            metadata_list: List of metadata from scenario generation

        Returns:
            Tuple containing:
            - Selected scenarios tensor [batch_size, num_selected, hidden_size]
            - Dictionary of evaluation metrics
        """
        batch_size, seq_len, hidden_size = scenarios.shape

        # Compute evaluation scores
        risk_scores = self.risk_evaluator(scenarios).squeeze(-1)  # [batch_size, seq_len]
        benefit_scores = self.benefit_evaluator(scenarios).squeeze(-1)  # [batch_size, seq_len]
        coherence_scores = self.coherence_evaluator(scenarios).squeeze(-1)  # [batch_size, seq_len]

        # Normalize weights for combining scores
        weights = F.softmax(self.importance_weights, dim=0)

        # Combine scores
        combined_scores = (
                weights[0] * risk_scores +
                weights[1] * benefit_scores +
                weights[2] * coherence_scores
        )  # [batch_size, seq_len]

        # Select top scenarios
        k = min(3, seq_len)
        scores, indices = torch.topk(combined_scores, k=k, dim=1)  # [batch_size, k]

        # Gather selected scenarios
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        selected_scenarios = torch.gather(scenarios, 1, indices_expanded)

        # Compile metrics
        metrics = {
            'risk_scores': risk_scores,
            'benefit_scores': benefit_scores,
            'coherence_scores': coherence_scores,
            'combined_scores': combined_scores,
            'selected_scores': scores
        }

        return selected_scenarios, metrics


class ISS(nn.Module):
    """
    Complete Intrinsic Scenario Synthesis system implementing all layers.
    """

    def __init__(self, config: ISSConfig):
        super().__init__()
        self.config = config

        # Initialize components
        self.noise_layer = NoiseLayer(config)
        self.scenario_generator = ScenarioGenerator(config)
        self.scenario_evaluator = ScenarioEvaluator(config)

        # Memory management
        self.memory_bank = deque(maxlen=config.max_memories)
        self.importance_threshold = config.importance_threshold

        # Memory features
        self.memory_embedding = nn.Linear(config.hidden_size, config.latent_dim)
        self.memory_importance = nn.Linear(config.latent_dim, 1)

        # Temporal features
        self.temporal_decay = config.temporal_decay
        self.memory_timestamps = deque(maxlen=config.max_memories)

        logger.info(f"Initialized ISS with config: {config}")

    def forward(
            self,
            input_states: torch.Tensor,
            memory_states: Optional[torch.Tensor] = None,
            return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Process input through all ISS layers.

        Args:
            input_states: Input representation tensor
            memory_states: Optional memory states to consider
            return_intermediates:

        Returns:
            Dictionary containing generated scenarios and processing metrics
        """
        # Layer 1: Generate noise patterns with memory integration
        noise_patterns = self.noise_layer(input_states, memory_states)
        importance_values = noise_patterns[..., -1]

        # Layer 2 is integrated within Layer 1 and ScenarioGenerator

        # Layer 3: Generate scenarios using efficient scanning
        scenarios, metadata = self.scenario_generator(
            noise_patterns, input_states, importance_values
        )

        # Layer 4: Evaluate scenarios across multiple criteria
        selected_scenarios, metrics = self.scenario_evaluator(
            scenarios, input_states, metadata
        )

        # Update memory bank with new experiences
        self._update_memory(selected_scenarios, metrics)

        # Decay old memories based on temporal factors
        self._apply_memory_decay()

        # Prepare return values
        results = {
            'scenarios': selected_scenarios,
            'metrics': metrics
        }

        if return_intermediates:
            results.update({
                'noise_patterns': noise_patterns,
                'importance_values': importance_values,
                'generation_metadata': metadata,
                'evaluation_metrics': metrics
            })

        return results

    def _update_memory(
            self,
            scenarios: torch.Tensor,  # [batch_size, num_selected, hidden_size]
            metrics: Dict[str, torch.Tensor]
    ) -> None:
        """
        Update memory bank with new scenarios based on importance values.

        Args:
            scenarios: Selected scenarios tensor [batch_size, num_selected, hidden_size]
            metrics: Dictionary containing evaluation metrics including 'selected_scores'
                    for the chosen scenarios
        """
        # Calculate importance scores for selected scenarios only
        # scenarios shape: [batch_size, num_selected, hidden_size]
        scenario_embeddings = self.memory_embedding(scenarios)  # [batch_size, num_selected, latent_dim]
        importance_scores = self.memory_importance(scenario_embeddings).squeeze(-1)  # [batch_size, num_selected]

        # Use the selected_scores from metrics instead of full combined_scores
        # selected_scores shape: [batch_size, num_selected]
        combined_importance = importance_scores * metrics['selected_scores']

        # Iterate through batch and scenarios
        batch_size, num_selected = combined_importance.shape
        for batch_idx in range(batch_size):
            for scenario_idx in range(num_selected):
                importance_value = combined_importance[batch_idx, scenario_idx].item()

                # Only store scenarios above importance threshold
                if importance_value > self.importance_threshold:
                    memory_entry = {
                        'scenario': scenarios[batch_idx, scenario_idx].detach(),
                        'importance': importance_value,
                        'timestamp': time.time()
                    }

                    self.memory_bank.append(memory_entry)
                    self.memory_timestamps.append(time.time())

        # Log memory update for debugging
        logger.debug(f"Updated memory bank, current size: {len(self.memory_bank)}")

        # Optional: Enforce memory limit if needed
        while len(self.memory_bank) > self.config.max_memories:
            self.memory_bank.popleft()
            self.memory_timestamps.popleft()

    def _apply_memory_decay(self) -> None:
        """
        Apply temporal decay to stored memories and remove those below threshold.

        This implements a natural forgetting mechanism where older memories
        gradually fade unless they are reinforced through repeated activation.
        """
        if not self.memory_bank:  # Handle empty memory bank case
            return

        current_time = time.time()

        # Create new deques to store retained memories
        retained_memories = deque(maxlen=self.config.max_memories)
        retained_timestamps = deque(maxlen=self.config.max_memories)

        # Process each memory
        for memory, timestamp in zip(self.memory_bank, self.memory_timestamps):
            # Calculate time-based decay
            time_diff = current_time - timestamp
            decay_factor = np.exp(-self.temporal_decay * time_diff)

            # Update importance with decay
            new_importance = memory['importance'] * decay_factor

            # Keep memory if still important enough
            if new_importance > self.importance_threshold:
                memory['importance'] = new_importance
                retained_memories.append(memory)
                retained_timestamps.append(timestamp)

        # Update memory bank with retained memories
        self.memory_bank = retained_memories
        self.memory_timestamps = retained_timestamps

    def get_memory_states(self) -> Optional[torch.Tensor]:
        """
        Retrieve current memory states for processing.

        Returns:
            Tensor of memory states if memories exist, None otherwise
        """
        if not self.memory_bank:
            return None

        memory_tensors = []
        for memory in self.memory_bank:
            memory_tensors.append(memory['scenario'])

        return torch.stack(memory_tensors, dim=0)

    def reset_memories(self) -> None:
        """Clear all stored memories, resetting the system state."""
        self.memory_bank.clear()
        self.memory_timestamps.clear()
        logger.info("Reset all memories")


def main():
    """Example usage of the ISS system."""
    # Create configuration
    config = ISSConfig(
        hidden_size=768,
        latent_dim=256,
        importance_dim=32,
        batch_size=1,  # Adjusted for single text input
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Initialize system
    iss = ISS(config).to(config.device)

    # Example text input
    text_input = "If a person walks to a horse farm"

    # Use a tokenizer and model to get embeddings
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').to(config.device)

    inputs = tokenizer(text_input, return_tensors='pt').to(config.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings
    input_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

    # Process input
    results = iss(input_states, return_intermediates=True)

    # Print results
    print("\nGenerated Scenarios:")
    print("-" * 40)
    print(f"Scenario shape: {results['scenarios'].shape}")

    # If desired, decode the selected scenarios back to text
    # For simplicity, let's just print the top scenario embeddings
    top_scenario = results['scenarios'][0, 0]  # First batch, top scenario

    # Note: Since the scenario is in embedding space, to interpret it we need to map it back to tokens
    # This requires additional steps and is beyond the scope of this code
    # For now, we'll just print the embedding

    print(f"Top scenario embedding (first 10 elements): {top_scenario[:10]}")

    print("\nEvaluation Metrics:")
    print("-" * 40)
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value.mean().item():.3f}")


if __name__ == "__main__":
    main()
