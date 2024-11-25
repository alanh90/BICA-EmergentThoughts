# Artificial-Subconscious

![Cover Image](media/cover_img.png)

## Overview
**Artificial-Subconscious** is an innovative system designed to simulate a subconscious-like process within an AI. Inspired by human cognition, this model processes raw inputs (noise), extracts meaningful insights, generates hypothetical scenarios, and evaluates them for relevance and risk. The ultimate goal is to enhance decision-making and adaptability in artificial intelligence systems, bridging the gap between human-like intuition and computational precision.

## Key Features

### **1. Layered Sub-Conscious Processing Architecture**
A multi-layered approach to simulate subconscious thought processes:

#### **Layer 1: Noise Generation**
- Generates foundational noise for subconscious simulation:
$$
N(x, y, z) = R(x, y, z) + M(x, y, z)
$$
  where:
  - \( R(x, y, z) \): Random noise sampled from a uniform distribution \( U(a, b) \).
  - \( M(x, y, z) \): Memory-related data prioritized based on past experiences.

- Noise includes **valleys** and **peaks** to prioritize:
  - **Peaks:** Data associated with previous success, determined using:
$$
S_{\text{peak}} = \text{max}(N(x, y, z))
$$
  - **Valleys:** Data with past negative outcomes, represented as:
$$
S_{\text{valley}} = \text{min}(N(x, y, z))
$$

#### **Layer 2: Identification of Useful Data**
- Selects semi-random data points from Layer 1:
  - Highest peaks and lowest valleys are prioritized based on a scoring function:
$$
w_i = \frac{e^{\beta S_i}}{\sum_{j} e^{\beta S_j}}
$$
    where:
    - \( S_i \): Success (or failure) score of each data point.
    - \( \beta \): Scaling factor to emphasize significant values.

- Introduces new random data \( \epsilon \sim \mathcal{N}(0, \sigma^2) \) around selected peaks and valleys for diversity.

#### **Layer 3: Hypothetical Scenario Creation**
- Creates scenarios using selected data points:
$$
S_i = \{ x_j + \epsilon_j : x_j \in D, \epsilon_j \sim \mathcal{N}(0, \sigma^2) \}
$$
  where:
  - \( D \): Selected data points from Layer 2.
  - \( \epsilon_j \): Gaussian noise added for creativity.

- Each scenario is weighed using a **benefit-risk function**:
$$
B_r(S_i) = \alpha \cdot B(S_i) - \gamma \cdot R(S_i)
$$
  where:
  - \( B(S_i) \): Expected benefit of scenario \( S_i \).
  - \( R(S_i) \): Associated risk.
  - \( \alpha, \gamma \): Weighting factors.

#### **Layer 4: Probability-Based Sorting**
- Sorts scenarios based on their benefit-risk scores:
$$
P(S_i) = \frac{e^{\lambda B_r(S_i)}}{\sum_{j} e^{\lambda B_r(S_j)}}
$$
  where:
  - \( \lambda \): Scaling factor to control sensitivity to \( B_r(S_i) \).

- Common scenarios are consolidated using clustering algorithms to avoid redundancy.

#### **Layer 5: Final Scenario Selection**
- Outputs top \( N \) scenarios based on probability:
$$
S_{\text{final}} = \text{TopN}(S, P(S))
$$

---

### **2. Dynamic Scenario Generation**
- Generates multiple possible futures \( x_{t+1} \) from the current state \( x_t \):
$$
x_{t+1} = f(x_t) + \eta_t
$$
  where:
  - \( f(x_t) \): Transition function representing system dynamics.
  - \( \eta_t \): Random noise introduced for uncertainty modeling.

- Evaluates scenarios using Monte Carlo simulations:
$$
E(O) = \frac{1}{N} \sum_{i=1}^N f(S_i, X)
$$
  where:
  - \( S_i \): Generated scenarios.
  - \( X \): Contextual factors influencing the outcome.

---

### **3. Risk and Benefit Analysis**
- Conducts Bayesian analysis to refine scenario probabilities:
$$
P(O | S_i) = \frac{P(S_i | O) \cdot P(O)}{P(S_i)}
$$
  where:
  - \( O \): Observed outcome.
  - \( S_i \): Scenario under evaluation.

- Integrates adaptive risk-benefit adjustments:
$$
R_b(S_i) = \frac{B(S_i)}{1 + R(S_i)}
$$
  where:
  - \( R_b(S_i) \): Risk-benefit score balancing opportunity and danger.

---

## Use Cases

### **1. Autonomous Systems**
- Enhances adaptability using **stochastic differential equations**:
$$
dx_t = f(x_t, t) dt + g(x_t, t) dW_t
$$
  where:
  - \( dx_t \): Change in system state at time \( t \).
  - \( f(x_t, t) \): Drift term (deterministic dynamics).
  - \( g(x_t, t) \): Diffusion term (random environmental effects).
  - \( W_t \): Wiener process (randomness).

### **2. Strategic AI**
- Simulates outcomes using scenario modeling and Monte Carlo methods.

### **3. Creative AI**
- Generates novel scenarios using **variational autoencoders (VAEs)**:
$$
p(S | z) = p(S) \cdot q(z | S)
$$
  where:
  - \( z \): Latent variable capturing compressed features.

### **4. Risk Management**
- Balances risk and benefit in critical applications using dynamic adjustments to priorities.

---

## Why Artificial-Subconscious?
This project aims to provide AI systems with a "subconscious" layer that operates beneath conscious decision-making, offering:
- **Improved Adaptability**: Processes complex and unstructured inputs to uncover meaningful insights.
- **Enhanced Creativity**: Simulates diverse scenarios, including high-risk possibilities.
- **Human-Like Intuition**: Mimics subconscious processing for better alignment with human-like reasoning.
