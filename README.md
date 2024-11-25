# Artificial-Subconscious

![Cover Image](media/cover_img.png)

## Overview
**Artificial-Subconscious** is an innovative system designed to simulate a subconscious-like process within an AI. Inspired by human cognition, this model processes raw inputs (noise), extracts meaningful insights, generates hypothetical scenarios, and evaluates them for relevance and risk. The ultimate goal is to enhance decision-making and adaptability in artificial intelligence systems, bridging the gap between human-like intuition and computational precision.

## Key Features

### **1. Layered Sub-Conscious Processing Architecture**
A multi-layered approach to simulate subconscious thought processes:

#### **Layer 1: Noise Generation**
- Generates foundational noise for subconscious simulation.
- Combines:
  - Random noise.
  - Relevant memory data related to the current situation.

   $$
  N(x, y, z) = R(x, y, z) + M(x, y, z)
  $$


- Noise includes **valleys** and **peaks** to prioritize:
  - **Peaks:** Previous data with successful outcomes or benefits.
  - **Valleys:** Data associated with negative consequences or risks.

#### **Layer 2: Identification of Useful Data**
- Selects semi-random data points from Layer 1, including:
  - The highest peaks.
  - Random noise around the peaks (for creativity).
  - The lowest valleys (for risk awareness).
- Introduces new random data related to the selected points as background noise.

#### **Layer 3: Hypothetical Scenario Creation**
- Creates scenarios based on insights from Layer 2:
  - **High-benefit scenarios**
  - **High-risk scenarios** (to explore potential dangers).
  - **Random scenarios** (for creative problem-solving).

  Each Scenario is weighed during runtime.

#### **Layer 4: Probability-Based Sorting**
- Consolidates scenarios with common patterns.
- Selects the **top N scenarios** and the **worst high-risk scenarios**.
- Integrates **memory** to apply a probability-based sorting mechanism using past experiences.

#### **Layer 5: Final Scenario Selection**
- Outputs the top N scenarios for the rest of the AI system to process and act upon.

### **2. Dynamic Scenario Generation**
- Generates multiple possible futures or outcomes based on current inputs.
- Evaluates scenarios to optimize benefits for the AI's current objectives.

### **3. Risk and Benefit Analysis**
- Integrates risky or unconventional scenarios to expand decision-making options.
- Assesses scenarios based on probability and historical occurrence for adaptive responses.

## Use Cases
- **Autonomous Systems**: Enhancing adaptability and risk awareness in real-time decision-making.
- **Strategic AI**: Simulating multiple outcomes for complex problem-solving.
- **Creative AI**: Generating imaginative or unconventional scenarios to support innovation.
- **Risk Management**: Balancing opportunities and dangers in critical applications like finance or disaster response.

## Why Artificial-Subconscious?
This project aims to provide AI systems with a "subconscious" layer that operates beneath conscious decision-making, offering:
- **Improved Adaptability**: Processes complex and unstructured inputs to uncover meaningful insights.
- **Enhanced Creativity**: Simulates diverse scenarios, including high-risk possibilities.
- **Human-Like Intuition**: Mimics subconscious processing for better alignment with human-like reasoning.
