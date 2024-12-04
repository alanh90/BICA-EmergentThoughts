import time
import logging
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional

import torch
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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

class NoiseLayer:
    """
    Layer 1: Noise and Memory Integration

    This layer introduces controlled noise into the input text to generate variations.
    """

    def __init__(self, config: ISSConfig, tokenizer, model):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

    def forward(self, input_text: str, memory_texts: Optional[List[str]] = None) -> List[str]:
        """
        Generate variations of the input text by introducing noise.

        Args:
            input_text: Original input text.
            memory_texts: Optional list of memory texts.

        Returns:
            List of noisy input texts.
        """
        variations = [input_text]
        # Introduce noise by paraphrasing or slight modifications
        num_variations = 2  # Number of variations to generate

        for _ in range(num_variations):
            # Create a prompt for paraphrasing
            prompt = f"Paraphrase the following sentence:\n\"{input_text}\"\nParaphrase:"
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.config.device)
            attention_mask = torch.ones_like(input_ids)

            # Generate paraphrase
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + 50,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
            )
            paraphrase = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the paraphrased sentence
            paraphrased_sentence = paraphrase[len(prompt):].split('\n')[0].strip().strip('"')
            variations.append(paraphrased_sentence)

        # Integrate memory texts if available
        if memory_texts:
            variations.extend(memory_texts)

        return variations

class ScenarioGenerator:
    """
    Layer 3: Hypothetical Scenario Generation

    Generates hypothetical scenarios based on noisy input variations.
    """

    def __init__(self, config: ISSConfig, tokenizer, model):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

    def forward(self, variations: List[str], num_scenarios=3) -> List[str]:
        """
        Generate scenarios based on input variations.

        Args:
            variations: List of input text variations.

        Returns:
            List of generated scenario texts.
        """
        scenarios = []
        for variation in variations:
            # Generate scenarios using the language model
            prompt = (
                f"Based on the following situation:\n\"{variation}\"\n\n"
                "List possible scenarios that could happen next:\n"
                "1."
            )
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.config.device)
            attention_mask = torch.ones_like(input_ids)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + 150,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract scenarios from the generated text
            lines = generated_text[len(prompt):].split('\n')
            count = 1
            current_scenario = ""
            for line in lines:
                line = line.strip()
                if line.startswith(f"{count}."):
                    # Save the previous scenario if it exists
                    if current_scenario:
                        scenarios.append(current_scenario.strip())
                        if len(scenarios) >= num_scenarios:
                            break
                    # Start a new scenario
                    current_scenario = line[line.find('.') + 1:].strip()
                    count += 1
                elif line.startswith(tuple(f"{i}." for i in range(1, num_scenarios + 1))):
                    # Handles cases where numbering skips or repeats
                    current_scenario = line[line.find('.') + 1:].strip()
                    count = int(line[0]) + 1
                elif line:
                    # Continue building the current scenario
                    current_scenario += ' ' + line
            # Add the last scenario if it exists
            if current_scenario and len(scenarios) < num_scenarios:
                scenarios.append(current_scenario.strip())
        return scenarios[:num_scenarios]


class ScenarioEvaluator:
    """
    Layer 4: Scenario Evaluation and Ranking

    Evaluates generated scenarios based on multiple criteria.
    """

    def __init__(self, config: ISSConfig):
        self.config = config
        from transformers import pipeline
        # Specify the zero-shot classification model
        self.classifier = pipeline(
            'zero-shot-classification',
            model='facebook/bart-large-mnli',
            device=0 if config.device == 'cuda' else -1
        )

    def forward(self, scenarios: List[str], input_text: str) -> List[Tuple[str, float]]:
        candidate_labels = ["high risk", "high benefit", "neutral"]
        evaluated_scenarios = []
        for scenario in scenarios:
            result = self.classifier(scenario, candidate_labels)
            label_scores = dict(zip(result['labels'], result['scores']))
            combined_score = (
                label_scores.get("high benefit", 0) * 1.0 +
                label_scores.get("high risk", 0) * 0.5 +
                label_scores.get("neutral", 0) * 0.1
            )
            evaluated_scenarios.append((scenario, combined_score))
        evaluated_scenarios.sort(key=lambda x: x[1], reverse=True)
        return evaluated_scenarios

class ISS:
    """
    Complete Intrinsic Scenario Synthesis system integrating all layers.
    """

    def __init__(self, config: ISSConfig, tokenizer, model):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.noise_layer = NoiseLayer(config, tokenizer, model)
        self.scenario_generator = ScenarioGenerator(config, tokenizer, model)
        self.scenario_evaluator = ScenarioEvaluator(config)

        # Memory management
        self.memory_bank = deque(maxlen=config.max_memories)
        self.importance_threshold = config.importance_threshold
        self.temporal_decay = config.temporal_decay
        self.memory_timestamps = deque(maxlen=config.max_memories)

        logger.info(f"Initialized ISS with config: {config}")

    def generate(self, input_text: str) -> List[Tuple[str, float]]:
        """
        Generate and evaluate scenarios based on the input text.

        Args:
            input_text: Original input text.

        Returns:
            List of evaluated scenarios.
        """
        # Retrieve memory texts
        memory_texts = self.get_memory_texts()

        # Layer 1: Generate variations with noise and memory integration
        variations = self.noise_layer.forward(input_text, memory_texts)

        # Layer 3: Generate scenarios based on variations
        scenarios = self.scenario_generator.forward(variations)

        # Layer 4: Evaluate scenarios
        evaluated_scenarios = self.scenario_evaluator.forward(scenarios, input_text)

        # Update memory with top scenarios
        self.update_memory(evaluated_scenarios)

        # Apply memory decay
        self.apply_memory_decay()

        return evaluated_scenarios

    def get_memory_texts(self) -> List[str]:
        """Retrieve texts from memory bank."""
        if not self.memory_bank:
            return []
        return [memory['text'] for memory in self.memory_bank]

    def update_memory(self, evaluated_scenarios: List[Tuple[str, float]]) -> None:
        """Update memory bank with new scenarios based on importance."""
        current_time = time.time()
        for scenario, score in evaluated_scenarios:
            if score > self.importance_threshold:
                memory_entry = {
                    'text': scenario,
                    'importance': score,
                    'timestamp': current_time
                }
                self.memory_bank.append(memory_entry)
                self.memory_timestamps.append(current_time)

    def apply_memory_decay(self) -> None:
        """Decay memory importance over time."""
        if not self.memory_bank:
            return
        current_time = time.time()
        retained_memories = deque()
        retained_timestamps = deque()
        for memory, timestamp in zip(self.memory_bank, self.memory_timestamps):
            time_diff = current_time - timestamp
            decay_factor = np.exp(-self.temporal_decay * time_diff)
            new_importance = memory['importance'] * decay_factor
            if new_importance > self.importance_threshold:
                memory['importance'] = new_importance
                retained_memories.append(memory)
                retained_timestamps.append(timestamp)
        self.memory_bank = retained_memories
        self.memory_timestamps = retained_timestamps

    def reset_memories(self) -> None:
        """Reset the memory bank."""
        self.memory_bank.clear()
        self.memory_timestamps.clear()
        logger.info("Reset all memories")

def generate_final_output(input_text: str, selected_scenarios: List[str], tokenizer, model, config) -> str:
    prompt = f"{input_text}\n\nConsidering these possible scenarios:\n"
    for idx, scenario in enumerate(selected_scenarios, 1):
        prompt += f"{idx}. {scenario}\n"
    prompt += "\nWrite a coherent continuation of the story that incorporates one or more of these scenarios."

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(config.device)
    attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.size(1) + 200,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = generated_text[len(prompt):].strip()
    return output_text

def test_scenario_1(iss, tokenizer, model, config):
    """
    Test case 1: A man runs towards the edge of a building
    Expected output should relate to him falling or preventing the fall.
    """
    text_input = "A man runs towards the edge of a building."
    # Generate and evaluate scenarios using ISS
    evaluated_scenarios = iss.generate(text_input)
    # Select top scenarios
    selected_scenarios = [scenario for scenario, score in evaluated_scenarios[:3]]
    # Generate final output
    generated_text = generate_final_output(text_input, selected_scenarios, tokenizer, model, config)
    # Print outputs
    print("\nTest Scenario 1 Output:")
    print("-" * 40)
    print(f"Input: {text_input}")
    print("\nGenerated Scenarios:")
    for idx, scenario in enumerate(selected_scenarios, 1):
        print(f"{idx}. {scenario}")
    print("\nFinal Output:")
    print(generated_text)

def test_scenario_2(iss, tokenizer, model, config):
    """
    Test case 2: A man sees $5000 on the ground that he can take for free with no consequences
    Expected output should reflect the benefit of picking up the money.
    """
    text_input = "A man sees $5000 on the ground that he can take for free with no consequences."
    # Generate and evaluate scenarios using ISS
    evaluated_scenarios = iss.generate(text_input)
    # Select top scenarios
    selected_scenarios = [scenario for scenario, score in evaluated_scenarios[:3]]
    # Generate final output
    generated_text = generate_final_output(text_input, selected_scenarios, tokenizer, model, config)
    # Print outputs
    print("\nTest Scenario 2 Output:")
    print("-" * 40)
    print(f"Input: {text_input}")
    print("\nGenerated Scenarios:")
    for idx, scenario in enumerate(selected_scenarios, 1):
        print(f"{idx}. {scenario}")
    print("\nFinal Output:")
    print(generated_text)

def main():
    """Main function to initialize components and run tests."""
    # Create configuration
    config = ISSConfig()

    # Initialize GPT-Neo for scenario generation and final text generation
    print("Loading GPT-Neo tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')  # Use smaller model for practicality
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M').to(config.device)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
    print("GPT-Neo tokenizer and model loaded.")

    # Initialize ISS system
    print("Initializing ISS system...")
    iss = ISS(config, tokenizer, model)
    print("ISS system initialized.")

    # Run Test Scenario 1
    test_scenario_1(iss, tokenizer, model, config)

    # Run Test Scenario 2
    test_scenario_2(iss, tokenizer, model, config)

if __name__ == "__main__":
    main()
