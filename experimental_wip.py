import time
import logging
from dataclasses import dataclass
from collections import deque
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

@dataclass
class ISSConfig:
    """Configuration class for the ISS system"""
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
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            device=self.config.device
        )

        if self.prev_noise is not None:
            prev_noise_seq_len = self.prev_noise.shape[1]
            if prev_noise_seq_len >= seq_len:
                prev_noise_slice = self.prev_noise[:, -seq_len:]
            else:
                # Pad prev_noise to match current seq_len
                pad_size = seq_len - prev_noise_seq_len
                prev_noise_slice = F.pad(self.prev_noise, (0, 0, 0, pad_size))
                prev_noise_slice = prev_noise_slice[:, -seq_len:]
            current_noise = (
                self.alpha * current_noise +
                (1 - self.alpha) * prev_noise_slice
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
        # For simplicity, average the importance patterns
        combined = (current + memory) / 2
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
    contextual relationships.
    """

    def __init__(self, config: ISSConfig):
        super().__init__()
        self.config = config

        # Project context to latent space for processing
        self.context_encoder = nn.Linear(config.hidden_size, config.latent_dim)

        # Project combined features back to scenario space
        # Input size accounts for noise_patterns and context
        combined_dim = config.latent_dim * config.importance_dim + config.latent_dim
        self.scenario_decoder = nn.Linear(combined_dim, config.hidden_size)

    def forward(
        self,
        noise_patterns: torch.Tensor,  # [batch, seq_len, latent_dim, importance_dim]
        context_states: torch.Tensor,  # [batch, seq_len, hidden_size]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate scenarios from noise patterns while maintaining context relationships.
        """
        batch_size, seq_len = context_states.shape[:2]

        # Encode context into latent space
        context_encoded = self.context_encoder(context_states)  # [batch, seq_len, latent_dim]

        # Flatten noise patterns
        noise_flat = noise_patterns.view(batch_size, seq_len, -1)  # [batch, seq_len, latent_dim * importance_dim]

        # Combine noise and context
        combined = torch.cat([noise_flat, context_encoded], dim=-1)  # [batch, seq_len, combined_dim]

        # Generate scenarios
        scenarios = self.scenario_decoder(combined)  # [batch, seq_len, hidden_size]

        # Metadata placeholder (could include complexity, coherence, etc.)
        metadata_list = [{} for _ in range(batch_size)]

        return scenarios, metadata_list

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

        # Select top scenarios based on combined scores
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

class ScenarioDecoder(nn.Module):
    """
    Layer 5: Surface-Level Scenario Selection

    Decoder Network to Generate Scenario Texts from ISS Embeddings
    """

    def __init__(self, config: ISSConfig, device):
        super().__init__()
        self.config = config
        self.device = device

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
        self.decoder_model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B').to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure padding token is set

        # Add a projection layer to match GPT-Neo's hidden size
        self.projection = nn.Linear(config.hidden_size, self.decoder_model.config.hidden_size)

    def forward(self, scenario_embeddings: torch.Tensor) -> List[str]:
        batch_size, num_selected, hidden_size = scenario_embeddings.shape
        scenario_texts = []

        for i in range(batch_size):
            for j in range(num_selected):
                embedding = scenario_embeddings[i, j].unsqueeze(0)  # Shape: [1, hidden_size]

                # Project the embedding to match GPT-Neo's hidden size
                projected_embedding = self.projection(embedding)  # Shape: [1, hidden_size]

                # Prepare inputs_embeds
                inputs_embeds = projected_embedding.unsqueeze(1)  # Shape: [1, seq_len=1, hidden_size]

                # Create attention mask of shape [batch_size, seq_len]
                attention_mask = torch.ones((inputs_embeds.size(0), inputs_embeds.size(1)), dtype=torch.long).to(self.device)

                # Generate text using the decoder model
                outputs = self.decoder_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_length=50,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )
                # Decode the generated tokens
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                scenario_texts.append(text.strip())

        return scenario_texts

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

    def forward(self, input_states: torch.Tensor, memory_states: Optional[torch.Tensor] = None, return_intermediates: bool = False):
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

        # Layer 3: Generate scenarios
        scenarios, metadata = self.scenario_generator(
            noise_patterns, input_states
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

def generate_text_with_influence(input_text: str, scenario_texts: List[str], tokenizer, model, config) -> str:
    """
    Generate text using the LLM influenced by ISS-generated scenario texts.
    """
    # Combine the input text with the scenario texts
    prompt = f"{input_text}\n\nPossible scenarios:\n"
    for idx, scenario in enumerate(scenario_texts, 1):
        prompt += f"{idx}. {scenario}\n"

    prompt += "\nContinuation:\n"

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(config.device)

    # Prepare attention mask
    attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(config.device)

    # Generate text with adjusted parameters
    max_length = input_ids.size(1) + 50  # Adjust as needed
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        num_beams=5,
        early_stopping=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract the continuation
    output_text = generated_text[len(prompt):].strip()

    return output_text

def test_scenario_1(iss, scenario_decoder, bert_tokenizer, bert_model, gpt_tokenizer, gpt_model, config):
    """
    Test case 1: A man runs towards the edge of a building
    Expected output should relate to him falling or preventing the fall.
    """
    text_input = "A man runs towards the edge of a building."
    bert_inputs = bert_tokenizer(text_input, return_tensors='pt').to(config.device)

    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)

    input_states = bert_outputs.last_hidden_state

    # Process input through ISS
    results = iss(input_states, return_intermediates=True)

    # Decode scenarios
    scenario_embeddings = results['scenarios']
    scenario_texts = scenario_decoder(scenario_embeddings)

    # Generate influenced text
    generated_text = generate_text_with_influence(text_input, scenario_texts, gpt_tokenizer, gpt_model, config)

    print("\nTest Scenario 1 Output:")
    print("-" * 40)
    print(f"Input: {text_input}")
    print("\nGenerated Scenarios:")
    for idx, scenario in enumerate(scenario_texts, 1):
        print(f"{idx}. {scenario}")

    print("\nFinal Output:")
    print(generated_text)

def test_scenario_2(iss, scenario_decoder, bert_tokenizer, bert_model, gpt_tokenizer, gpt_model, config):
    """
    Test case 2: A man sees $5000 on the ground that he can take for free with no consequences
    Expected output should reflect the benefit of picking up the money.
    """
    text_input = "A man sees $5000 on the ground that he can take for free with no consequences."
    bert_inputs = bert_tokenizer(text_input, return_tensors='pt').to(config.device)

    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)

    input_states = bert_outputs.last_hidden_state

    # Process input through ISS
    results = iss(input_states, return_intermediates=True)

    # Decode scenarios
    scenario_embeddings = results['scenarios']
    scenario_texts = scenario_decoder(scenario_embeddings)

    # Generate influenced text
    generated_text = generate_text_with_influence(text_input, scenario_texts, gpt_tokenizer, gpt_model, config)

    print("\nTest Scenario 2 Output:")
    print("-" * 40)
    print(f"Input: {text_input}")
    print("\nGenerated Scenarios:")
    for idx, scenario in enumerate(scenario_texts, 1):
        print(f"{idx}. {scenario}")

    print("\nFinal Output:")
    print(generated_text)

def main():
    """Main function to initialize components and run tests."""
    # Create configuration
    config = ISSConfig()

    # Initialize ISS system
    print("Initializing ISS system...")
    iss = ISS(config).to(config.device)
    print("ISS system initialized.")

    # Initialize Tokenizers and Models
    print("Loading BERT tokenizer and model...")
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased').to(config.device)
    print("BERT tokenizer and model loaded.")

    # Initialize GPT-Neo for scenario decoding and final text generation
    print("Loading GPT-Neo tokenizer and model...")
    gpt_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    gpt_model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B').to(config.device)
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # Ensure padding token is set
    print("GPT-Neo tokenizer and model loaded.")

    # Initialize Scenario Decoder
    print("Initializing Scenario Decoder...")
    scenario_decoder = ScenarioDecoder(config, config.device)
    print("Scenario Decoder initialized.")

    # Run Test Scenario 1
    test_scenario_1(iss, scenario_decoder, bert_tokenizer, bert_model, gpt_tokenizer, gpt_model, config)

    # Run Test Scenario 2
    test_scenario_2(iss, scenario_decoder, bert_tokenizer, bert_model, gpt_tokenizer, gpt_model, config)

if __name__ == "__main__":
    main()
