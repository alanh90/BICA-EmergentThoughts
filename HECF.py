import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Callable, Optional
import logging
import time
import numpy as np
import json
import os
import openai
from dotenv import load_dotenv
import re
import datasets
import csv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class HECFConfig:
    """Configuration for the Hungry Emergent Cognition Framework."""
    def __init__(self,
                 hidden_size=768,
                 latent_dim=256,
                 importance_dim=32,
                 batch_size=1,
                 max_memories=1000,
                 temporal_decay=0.1,
                 importance_threshold=0.3,
                 scan_size=64,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.importance_dim = importance_dim
        self.batch_size = batch_size
        self.max_memories = max_memories
        self.temporal_decay = temporal_decay
        self.importance_threshold = importance_threshold
        self.scan_size = scan_size
        self.device = device


class Layer1(nn.Module):
    """Layer 1: Noisy Memory Activation and Contextual Integration."""

    def __init__(self, config: HECFConfig, tokenizer, device):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.memory_bank = []  # list of {"memory" : str, "embedding": tensor,  "timestamp" : float}

    def forward(self, input_data: str):
        """Forward pass for Layer 1."""
        start = time.time()
        # 1. Input Data & Knowledge Base
        x = self.tokenizer.encode(input_data, return_tensors='pt').to(self.device).float()
        k = [m["embedding"] for m in self.memory_bank] if len(
            self.memory_bank) else None  # use memory embeddings as part of knowledge base
        logger.info(f"Layer 1 (Noise & Memory): Input tokenized in {time.time() - start:.4f}s")
        start = time.time()
        # 2. Memory Activation
        if k:
            A = []
            for m in self.memory_bank:
                sim_m_x = F.cosine_similarity(x, m["embedding"], dim=-1)
                if sim_m_x > 0.3:
                    A.append(m)
        else:
            A = []  # if no memories then its just an empty array

        K_prime = []  # set up for calculating the new weighted matrix
        for mem in A:  # weigh the knowledge base based on all activated memories
            for k_i in k:
                sim_k_mem = F.cosine_similarity(k_i, mem["embedding"], dim=-1)
                for x_i in x:
                    sim_k_x = F.cosine_similarity(k_i, x_i, dim=-1)
                    K_prime.append(k_i + sim_k_mem + sim_k_x)

        logger.info(f"Layer 1 (Noise & Memory): Memory activated in {time.time() - start:.4f}s")

        # 3. Importance Weighing
        W = []
        if K_prime:
            start = time.time()
            for k_prime_i in K_prime:
                sim_k_x = []  # to keep track of similarity
                sim_k_mem = []
                for x_i in x:  # weigh all of its token embeddings
                    sim_k_x.append(F.cosine_similarity(k_prime_i, x_i, dim=-1))
                for mem in A:  # weigh all of its memories
                    sim_k_mem.append(F.cosine_similarity(k_prime_i, mem["embedding"], dim=-1))
                W_i = k_prime_i + torch.mean(torch.stack(sim_k_x)) + torch.mean(torch.stack(sim_k_mem))
                W.append(W_i)
            logger.info(f"Layer 1 (Noise & Memory): Importance weighted in {time.time() - start:.4f}s")
        else:  # if there are no memories, just start by creating a zero array with the same length as the input
            W = [torch.zeros(x.shape[1]).to(self.device) for x_i in x]

        # 4. Noise Injection
        start = time.time()
        W_prime = []
        for w_i in W:
            noise = torch.randn(w_i.size(), device=self.device) * 0.1  # use standard deviation of 0.1 for initial tests
            W_prime.append(w_i + noise)
        logger.info(f"Layer 1 (Noise & Memory): Noise injected in {time.time() - start:.4f}s")

        # 5. Residual integration
        start = time.time()
        O_prev = torch.zeros(x.shape[1]).to(self.device)  # set default for when there is no previous output
        O_1 = [w_prime_i + 0.1 * O_prev for w_prime_i in W_prime]  # set alpha to 0.1 for this first iteration
        logger.info(f"Layer 1 (Noise & Memory): Residual integrated in {time.time() - start:.4f}s")

        return O_1

class Layer2(nn.Module):
    """Layer 2: Significant Data Element Extraction."""
    def __init__(self, config: HECFConfig, tokenizer, device):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, O_1: List[torch.Tensor], input_text:str):
        """Forward pass for Layer 2."""
        start = time.time()
        # 1. Peak and Valley Identification
        scores = [torch.norm(x).item() for x in O_1]  # Calculate the norm as a stand in value
        peaks = []
        valleys = []
        peak_threshold = np.mean(scores) + np.std(scores) # use a dynamic threshold for peaks
        valley_threshold = np.mean(scores) - np.std(scores) # use a dynamic threshold for valleys
        for idx, score in enumerate(scores):
            if score > peak_threshold:
                peaks.append(O_1[idx])
            elif score < valley_threshold:
                 valleys.append(O_1[idx])

        # 2. Controlled Randomness Selection
        k = 2 #  for this test I'll choose only two valleys for randomness
        if len(valleys) < k:
            selected_valleys = valleys
        else:
             selected_valleys = [valleys[i] for i in np.random.choice(len(valleys), size = k, replace=False)]

        # 3. Concept Abstraction

        prompt_concepts = (
                f"What are the 2 core, most abstract concepts behind this: \"{input_text}\"?"
                f" Only use one word per concept and return each concept in separate lines with no numbering."
            )
        response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_concepts}],
                temperature=0.7,
                top_p=0.9,
                n=1,
            )
        generated_concepts = response.choices[0].message.content
        concepts = [line.strip() for line in generated_concepts.split('\n') if line.strip()]
        logger.info(f"Layer 2 (Data Extraction): Abstract Concepts are: {concepts}")

        # 4. Output
        O_2 = peaks + selected_valleys
        logger.info(f"Layer 2 (Data Extraction): significant elements extracted in {time.time() - start:.4f}s")
        return O_2, concepts

class Layer3(nn.Module):
    """Layer 3: Hypothetical Scenario Generation using OpenAI."""
    def __init__(self, config: HECFConfig, tokenizer, device):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, O_2: List[torch.Tensor], concepts:List[str], input_text: str, num_scenarios=3) -> List[str]:
        """Generates scenarios using OpenAI's gpt-3.5-turbo."""
        start = time.time()
        scenarios = []

        # 2. Scenario Generation:
        prompt = (
            f"Based on the following situation: \"{input_text}\", and the abstract concepts: {concepts}, "
            "create 3 short, hypothetical scenarios of what could happen next in a story, it needs to be logical and creative. "
             "The scenarios should be around one sentence and they should explore all of those concepts. "
            " Do not include any information that does not belong to the scenario or any numbers or numbering, just output the sentence itself as separate lines."
            "\n"
        )
        logger.info(f"Layer 3 (Scenario Generator): Getting Scenarios Using OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.9,
            n=1,
        )
        generated_text = response.choices[0].message.content
        # Extract scenarios from the generated text
        lines = generated_text.split('\n')
        scenarios = [line.strip() for line in lines if line.strip()]  # clean up and filter empty spaces
        scenarios = scenarios[:num_scenarios]  # ensure we have num_scenarios
        O_3 = scenarios
        logger.info(f"Layer 3 (Scenario Generator): Scenarios Generated in {time.time() - start:.4f}s")
        return O_3


class Layer4(nn.Module):
    """Layer 4: Scenario Evaluation and Ranking."""
    def __init__(self, config: HECFConfig, device):
         super().__init__()
         self.config = config
         self.device = device
         from transformers import pipeline
        # Specify the zero-shot classification model
         self.classifier = pipeline(
             'zero-shot-classification',
             model='facebook/bart-large-mnli',
             device=0 if device == 'cuda' else -1
         )

    def forward(self, scenarios: List[str]) -> List[Tuple[str, float]]:
        """Forward pass for Layer 4."""
        start = time.time()
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
        O_4 = evaluated_scenarios
        logger.info(f"Layer 4 (Scenario Evaluator): Evaluated in {time.time() - start:.4f}s")
        return O_4

class Layer5(nn.Module):
    """Layer 5: Surface-Level Scenario Selection."""
    def __init__(self, config: HECFConfig):
        super().__init__()
        self.config = config

    def forward(self, O_4: List[Tuple[str, float]], num_selected=3) -> List[str]:
        """Forward pass for Layer 5."""
        start = time.time()
        selected_scenarios = [scenario for scenario, score in O_4[:num_selected]]
        O_5 = selected_scenarios
        logger.info(f"Layer 5 (Scenario Selection): Scenarios Selected in {time.time() - start:.4f}s")
        return O_5

class HECF(nn.Module):
    """Hungry Emergent Cognition Framework."""
    def __init__(self, config: HECFConfig, tokenizer, device):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.layer1 = Layer1(config, tokenizer, device)
        self.layer2 = Layer2(config, tokenizer, device)
        self.layer3 = Layer3(config, tokenizer, device)
        self.layer4 = Layer4(config, device)
        self.layer5 = Layer5(config)


    def forward(self, input_text: str) -> List[str]:
        """Forward pass of the HECF system."""
         # Layer 1
        O_1 = self.layer1(input_text)
        # Layer 2:
        O_2, concepts = self.layer2(O_1, input_text)
        # Layer 3
        O_3 = self.layer3(O_2, concepts, input_text)
         # Layer 4
        O_4 = self.layer4(O_3)
         # Layer 5
        O_5 = self.layer5(O_4)

        return O_5

def generate_final_output(input_text: str, selected_scenarios: List[str], tokenizer, model, config, concepts:List[str]) -> str:
    """Generates the final output for the combined system using OpenAI."""
    prompt = f"Given the story, \"{input_text}\", the following concepts: {concepts}, and these possible scenarios that could happen next:\n"
    for idx, scenario in enumerate(selected_scenarios, 1):
        prompt += f"{idx}. {scenario}\n"
    prompt += "\nWrite a coherent continuation of the story that incorporates at least one or more of these scenarios and the abstract concepts provided, and write only a single paragraph that is a direct continuation of the story using a natural and creative style, do not include any additional information, or numbering, and focus on being creative, while following the story."
    response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.9,
                n=1,
            )
    output_text = response.choices[0].message.content.strip()
    return output_text


def load_rocstories_data(file_path, num_samples=1):
    """Loads data from a ROCStories file and gets only the first few items, and returns the data and the answer"""
    stories = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as file:
         csv_reader = csv.reader(file)
         next(csv_reader)  # skip the header
         for i, row in enumerate(csv_reader):
              if i > num_samples-1:
                 break
              stories.append(" ".join(row[:5])) # the first five are the sentences of the stories.
              answers.append(row[7])
    return stories, answers

def main():
     # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Load config
    config = HECFConfig(device=device)
    # Initialize tokenizer and model (GPT-Neo for tokenization only)
    print("Loading GPT-Neo tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M').to(device) # this is not being used in layer 3 or output anymore
    tokenizer.pad_token = tokenizer.eos_token
    print("GPT-Neo tokenizer loaded.")
    # Initialize HECF
    hecf = HECF(config, tokenizer, device)
     # Load the ROCStories dataset
    file_path = "test/StoryClozeTest/cloze_test_val__spring2016 - cloze_test_ALL_val.csv" # place file in the folder of the python script
    roc_stories, answers = load_rocstories_data(file_path, num_samples = 1) # use the first story in the set

    # Take the first item as an example
    text_input = roc_stories[0]
    correct_answer = answers[0]
     # Process the data through HECF and get top results
    selected_scenarios = hecf(text_input)
    # generate the final output by passing top results into the LLM to produce final output
    prompt_concepts = (
                f"What are the 2 core, most abstract concepts behind this: \"{text_input}\"?"
                f" Only use one word per concept and return each concept in separate lines with no numbering."
            )
    response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_concepts}],
                temperature=0.7,
                top_p=0.9,
                n=1,
            )
    generated_concepts = response.choices[0].message.content
    concepts = [line.strip() for line in generated_concepts.split('\n') if line.strip()]  # extracts the generated concepts
    final_output = generate_final_output(text_input, selected_scenarios, tokenizer, model, config, concepts)

    # Print output
    print(f"\nInput: {text_input}")
    print("\nGenerated Scenarios:")
    for idx, scenario in enumerate(selected_scenarios, 1):
        print(f"{idx}. {scenario}")
    print(f"\nAbstract Concepts: {concepts}")
    print("\nFinal Output:")
    print(final_output)
    print(f"\nExpected Output: The correct answer is ending number {correct_answer} in the dataset. The correct story is in the dataset but will not be shown here since our purpose is to evaluate the generated text")


if __name__ == "__main__":
    main()