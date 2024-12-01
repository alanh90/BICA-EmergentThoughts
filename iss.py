# iss.py

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, LogitsProcessorList
import numpy as np
import re
from nltk.corpus import stopwords
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')


# --- ISS Module Implementation ---

class ISSModule:
    """
    Intrinsic Scenario Synthesis (ISS) Module
    """

    def __init__(self, tokenizer, device, llm, memory_buffer=None, memory_size=100):
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = tokenizer.vocab_size
        self.llm = llm  # Access to the LLM
        self.memory_buffer = memory_buffer if memory_buffer else []
        self.memory_size = memory_size  # Maximum number of scenarios to store
        self.stopwords = set(stopwords.words('english'))

    def process_input(self, input_ids):
        # Layer 1: Noise and Memory Integration
        related_token_ids = self.layer1_random_noise_and_memory_integration(input_ids)

        # Layers 2-4: Scenario Generation
        scenarios = self.generate_scenarios(input_ids, related_token_ids)
        print("\nLayers 2-4 - Generated Scenarios:")
        for idx, scenario in enumerate(scenarios):
            print(f"Scenario {idx + 1}: {scenario}")

            # Quality check (simple example: minimum length)
            if len(scenario.split()) >= 5:
                self.memory_buffer.append(scenario)

        # Maintain memory buffer size
        if len(self.memory_buffer) > self.memory_size:
            self.memory_buffer = self.memory_buffer[-self.memory_size:]

        # Layer 5: Prepare Influence
        influence_vector = self.layer5_prepare_influence(scenarios)
        self.print_top_influence_tokens(influence_vector)

        return influence_vector

    def layer1_random_noise_and_memory_integration(self, input_ids, num_random_tokens=30, num_memory_tokens=10):
        # Sample random tokens from the entire vocabulary, excluding input tokens
        input_token_ids = input_ids[0].tolist()
        excluded_token_ids = set(input_token_ids)

        # Sample random tokens
        random_token_ids = set()
        while len(random_token_ids) < num_random_tokens:
            token_id = torch.randint(0, self.vocab_size, (1,)).item()
            if token_id not in excluded_token_ids and token_id not in random_token_ids:
                random_token_ids.add(token_id)

        # Sample memory tokens: from memory buffer
        memory_token_ids = set()
        if self.memory_buffer:
            # Flatten memory buffer and sample unique tokens
            memory_tokens = ' '.join(self.memory_buffer)
            memory_token_ids = set(self.tokenizer.encode(memory_tokens, add_special_tokens=False))
            # Exclude input tokens and already sampled random tokens
            memory_token_ids = {tid for tid in memory_token_ids if tid not in excluded_token_ids and tid not in random_token_ids}
            # Limit to num_memory_tokens
            memory_token_ids = set(list(memory_token_ids)[:num_memory_tokens])

        # Combine random and memory tokens
        combined_token_ids = random_token_ids.union(memory_token_ids)

        return combined_token_ids

    def generate_scenarios(self, input_ids, influence_token_ids, num_scenarios=10, max_new_tokens=50):
        scenarios = []
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Decode influence tokens into meaningful words
        influence_tokens_text = self.tokenizer.decode(list(influence_token_ids), skip_special_tokens=True)
        # Extract unique, meaningful words
        influence_keywords = list(set(influence_tokens_text.split()))

        for idx in range(num_scenarios):
            # Create a refined prompt with clear instructions
            prompt = (
                f"Instruction: Given the input: \"{input_text}\", and considering factors like \"{', '.join(influence_keywords)}\", "
                "generate a coherent and relevant scenario. Scenario:"
            )

            # Generate a scenario using the LLM
            generated_scenario = self.generate_text_from_prompt(prompt, max_new_tokens=max_new_tokens)
            scenarios.append(generated_scenario)

        return scenarios

    def generate_text_from_prompt(self, prompt, max_new_tokens=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # Remove the prompt from the generated text
        generated_text = generated_text[len(prompt):].strip()
        return generated_text

    def layer5_prepare_influence(self, scenarios, top_n=20):
        """
        Constructs an influence vector based on the frequency of meaningful tokens in the generated scenarios.
        """
        influence_vector = torch.zeros(self.vocab_size, device=self.device)
        token_frequencies = {}
        for scenario in scenarios:
            scenario_ids = self.tokenizer.encode(scenario, add_special_tokens=False)
            # Filter tokens
            scenario_ids = [
                tid for tid in scenario_ids
                if self.is_meaningful_token(tid)
            ]
            for token_id in scenario_ids:
                token_frequencies[token_id] = token_frequencies.get(token_id, 0) + 1.0

        # Normalize frequencies
        if token_frequencies:
            max_freq = max(token_frequencies.values())
            for token_id, freq in token_frequencies.items():
                influence_vector[token_id] = freq / max_freq

        # Keep only the top N tokens to focus influence
        top_values, top_indices = torch.topk(influence_vector, k=top_n)
        focused_influence = torch.zeros(self.vocab_size, device=self.device)
        focused_influence[top_indices] = top_values

        return focused_influence

    def is_meaningful_token(self, token_id):
        """
        Determines whether a token is meaningful by checking if it's a word, meets length criteria, and is not a stopword.
        """
        token = self.tokenizer.decode([token_id]).strip()
        # Exclude tokens that are punctuation, single characters, subword tokens, or stopwords
        if not token:
            return False
        if re.fullmatch(r'[^\w\s]', token):
            return False
        if len(token) <= 1:
            return False
        if token.startswith('Ġ'):
            return False
        if token.lower() in self.stopwords:
            return False
        return True

    def print_top_influence_tokens(self, influence_vector, top_k=10):
        print("\nLayer 5 - Influence Vector (Top Tokens):")
        top_values, top_indices = torch.topk(influence_vector, k=top_k)
        top_tokens = self.tokenizer.convert_ids_to_tokens(top_indices.tolist())
        for value, token in zip(top_values.tolist(), top_tokens):
            print(f"Token: {token}, Value: {value}")


# --- Custom Logits Processor ---

class InfluenceLogitsProcessor(LogitsProcessor):
    def __init__(self, influence_vector: torch.Tensor, influence_weight: float):
        """
        Initializes the processor with the influence vector and weight.
        """
        super().__init__()
        self.influence_vector = influence_vector
        self.influence_weight = influence_weight

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Modifies the scores by adding the influence vector scaled by the influence weight.
        """
        # Apply a softmax to the influence vector to ensure it's a probability distribution
        influence = torch.softmax(self.influence_vector, dim=-1)
        # Scale the influence vector
        influence = influence * self.influence_weight
        # Add the influence to the logits
        scores += influence
        return scores


# --- Main LLM with ISS Integration ---

class MainLLMWithISS(nn.Module):
    def __init__(self, llm, iss_module, tokenizer, influence_weight=10.0):
        """
        Initializes the main LLM with ISS integration.
        """
        super().__init__()
        self.llm = llm
        self.iss_module = iss_module
        self.influence_weight = influence_weight
        self.tokenizer = tokenizer

    def generate_with_iss(self, input_ids, attention_mask=None, max_new_tokens=50):
        """
        Generates text influenced by the ISS module.
        """
        # Process input to get influence vector
        influence_vector = self.iss_module.process_input(input_ids)

        # Create custom logits processor
        logits_processor = LogitsProcessorList([
            InfluenceLogitsProcessor(influence_vector, self.influence_weight)
        ])

        # Generate text with ISS influence
        generated_ids = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            logits_processor=logits_processor,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Remove the input prompt from the generated text
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        generated_text = generated_text[len(input_text):].strip()
        return generated_text


# --- Testing Code ---

if __name__ == "__main__":
    # Initialize components
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize LLM and ISS module
    llm = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    iss_module = ISSModule(tokenizer, device, llm)
    model_with_iss = MainLLMWithISS(llm, iss_module, tokenizer, influence_weight=10.0).to(device)  # Reduced weight
    model_with_iss.eval()

    # Main LLM without ISS for comparison
    model_without_iss = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model_without_iss.eval()

    # User interaction loop
    print("Welcome to the ISS-enabled language model. Type 'exit' to quit.")
    while True:
        user_input = input("\nUser Input: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)

        # Generate output with ISS
        with torch.no_grad():
            generated_text_with_iss = model_with_iss.generate_with_iss(input_ids, attention_mask=attention_mask, max_new_tokens=50)

        # Generate output without ISS
        with torch.no_grad():
            generated_ids_without_iss = model_without_iss.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_text_without_iss = tokenizer.decode(generated_ids_without_iss[0], skip_special_tokens=True)
            # Remove the input prompt from the generated text
            generated_text_without_iss = generated_text_without_iss[len(user_input):].strip()

        # Print outputs
        print("\n=== Output with ISS ===")
        print(generated_text_with_iss)

        print("\n=== Output without ISS ===")
        print(generated_text_without_iss)
