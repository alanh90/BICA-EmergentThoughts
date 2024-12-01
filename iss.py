# iss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

# --- ISS Module Implementation ---

class ISSModule:
    """
    Intrinsic Scenario Synthesis (ISS) Module
    """
    def __init__(self, tokenizer, device, llm):
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = tokenizer.vocab_size
        self.llm = llm  # Access to the LLM

    def process_input(self, input_ids):
        # Layer 1: Random Noise and Memory Integration
        noise_vector = self.layer1_random_noise_and_memory_integration(input_ids)
        print("Layer 1 - Noise Vector (After Memory Integration):")
        print(noise_vector)

        # Layers 2-4: Scenario Generation
        scenarios = self.generate_scenarios(noise_vector)
        print("\nLayers 2-4 - Generated Scenarios:")
        for idx, scenario in enumerate(scenarios):
            print(f"Scenario {idx+1}: {scenario}")

        # Layer 5: Prepare Influence
        influence_vector = self.layer5_prepare_influence(scenarios)
        print("\nLayer 5 - Influence Vector:")
        print(influence_vector)

        return influence_vector

    def layer1_random_noise_and_memory_integration(self, input_ids):
        # Generate random noise
        random_noise = torch.rand(self.vocab_size, device=self.device)
        random_noise = random_noise / random_noise.sum()

        # Find related tokens using embeddings
        related_token_ids = self.find_related_tokens(input_ids)

        # Increase spike values for related tokens
        random_noise[related_token_ids] += 0.1  # Adjust as needed

        # Normalize again
        random_noise = random_noise / random_noise.sum()

        return random_noise

    def find_related_tokens(self, input_ids, top_k=50):
        with torch.no_grad():
            input_embeddings = self.llm.transformer.wte(input_ids).mean(dim=1)
            input_embeddings = input_embeddings / input_embeddings.norm(dim=1, keepdim=True)

            token_ids = torch.arange(self.vocab_size, device=self.device)
            token_embeddings = self.llm.transformer.wte(token_ids)
            token_embeddings = token_embeddings / token_embeddings.norm(dim=1, keepdim=True)

            similarities = torch.matmul(input_embeddings, token_embeddings.T).squeeze(0)
            _, top_indices = torch.topk(similarities, top_k)

        print("\nLayer 1 - Related Tokens (Memory Integration):")
        related_tokens_text = self.tokenizer.decode(top_indices)
        print(f"Related Token IDs: {top_indices}")
        print(f"Related Tokens Text: {related_tokens_text}")

        return top_indices

    def generate_scenarios(self, noise_vector):
        num_scenarios = 5
        sequence_length = 5

        scenarios = []
        print("\nLayers 2-4 - Generating Scenarios:")
        for idx in range(num_scenarios):
            sampled_token_ids = torch.multinomial(noise_vector, num_samples=sequence_length, replacement=True)
            sampled_tokens_text = self.tokenizer.decode(sampled_token_ids)
            print(f"Scenario {idx+1} - Sampled Tokens: {sampled_tokens_text}")

            generated_scenario = self.generate_scenario_from_tokens(sampled_token_ids)
            print(f"Scenario {idx+1} - Generated Text: {generated_scenario}")
            scenarios.append(generated_scenario)

        return scenarios

    def generate_scenario_from_tokens(self, token_ids):
        prompt_ids = token_ids.unsqueeze(0)
        attention_mask = torch.ones_like(prompt_ids)

        with torch.no_grad():
            outputs = self.llm.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_length=prompt_ids.size(1) + 20,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def layer5_prepare_influence(self, scenarios):
        influence_vector = torch.zeros(self.vocab_size, device=self.device)
        for scenario in scenarios:
            scenario_ids = self.tokenizer.encode(scenario, add_special_tokens=False, return_tensors='pt').to(self.device)
            influence_vector.scatter_add_(0, scenario_ids.squeeze(0), torch.ones_like(scenario_ids.squeeze(0), dtype=torch.float, device=self.device))
        influence_vector = influence_vector / influence_vector.sum()
        return influence_vector

# --- Main LLM with ISS Integration ---

class MainLLMWithISS(nn.Module):
    def __init__(self, llm, iss_module):
        super().__init__()
        self.llm = llm
        self.iss_module = iss_module
        self.influence_weight = 10.0  # Adjust as needed

    def forward(self, input_ids, attention_mask=None):
        influence_vector = self.iss_module.process_input(input_ids)
        outputs = self.llm(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        logits[:, -1, :] += self.influence_weight * influence_vector

        return {'logits': logits}

# --- Testing Code ---

if __name__ == "__main__":
    # Initialize components
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize LLM and ISS module
    llm = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    iss_module = ISSModule(tokenizer, device, llm)
    model_with_iss = MainLLMWithISS(llm, iss_module).to(device)
    model_with_iss.eval()

    # Main LLM without ISS for comparison
    model_without_iss = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model_without_iss.eval()

    # Input from user
    user_input = "The market is showing signs of volatility amid recent geopolitical tensions. What should be our investment strategy?"
    input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)

    # Generate output with ISS
    with torch.no_grad():
        outputs_with_iss = model_with_iss(input_ids, attention_mask=attention_mask)
        generated_ids_with_iss = outputs_with_iss['logits'].argmax(dim=-1)
        generated_text_with_iss = tokenizer.decode(generated_ids_with_iss[0], skip_special_tokens=True)

    # Generate output without ISS
    with torch.no_grad():
        outputs_without_iss = model_without_iss(input_ids, attention_mask=attention_mask)
        generated_ids_without_iss = outputs_without_iss.logits.argmax(dim=-1)
        generated_text_without_iss = tokenizer.decode(generated_ids_without_iss[0], skip_special_tokens=True)

    # Print outputs
    print("\n=== Output with ISS ===")
    print(generated_text_with_iss)

    print("\n=== Output without ISS ===")
    print(generated_text_without_iss)
