# Artificial-Subconscious

![Cover Image](media/cover_img.png)

## Overview
The **Artificial-Subconscious** system simulates a subconscious-like process within AI. Its operation can be summarized as:
$$
\text{AI}_{\text{subconscious}} = f(\text{Noise}, \text{Memory}, \text{Scenarios}) \rightarrow \text{Decision}
$$
where:
- \( \text{Noise} \): Raw inputs processed to generate foundational randomness.
- \( \text{Memory} \): Relevant historical data extracted for context.
- \( \text{Scenarios} \): Hypothetical outcomes evaluated for relevance and risk.

The objective function:
$$
\text{Goal} = \text{Maximize Adaptability} + \text{Enhance Decision Precision}
$$

---

## Key Features

### **1. Layered Sub-Conscious Processing Architecture**

#### **Layer 1: Noise Generation**
The system generates foundational noise as:
$$
N(x, y, z) = R(x, y, z) + M(x, y, z)
$$
where:
- \( R(x, y, z) \sim U(a, b) \): Uniformly distributed random noise.
- \( M(x, y, z) \): Memory-retrieved data prioritized by:
$$
M(x, y, z) = w_i \cdot \text{Success}_{\text{past}} - v_i \cdot \text{Risk}_{\text{past}}
$$

Peaks and valleys in the noise field are defined as:
$$
S_{\text{peak}} = \text{max}(N(x, y, z)), \quad S_{\text{valley}} = \text{min}(N(x, y, z))
$$

#### **Layer 2: Identification of Useful Data**
Semi-random data points are extracted:
$$
D = \{ S_{\text{peak}}, S_{\text{valley}}, \epsilon \}
$$
where \( \epsilon \sim \mathcal{N}(0, \sigma^2) \) introduces randomness. Each point is weighted by:
$$
w_i = \frac{e^{\beta S_i}}{\sum_{j} e^{\beta S_j}}
$$
where:
- \( \beta \): Sensitivity scaling parameter.
- \( S_i \): Relevance score.

#### **Layer 3: Hypothetical Scenario Creation**
Scenarios are generated as:
$$
S_i = \{ x_j + \epsilon_j : x_j \in D, \epsilon_j \sim \mathcal{N}(0, \sigma^2) \}
$$
and evaluated by the benefit-risk function:
$$
B_r(S_i) = \alpha \cdot B(S_i) - \gamma \cdot R(S_i)
$$
where:
- \( B(S_i) \): Expected benefit.
- \( R(S_i) \): Associated risk.
- \( \alpha, \gamma \): Importance weights.

#### **Layer 4: Probability-Based Sorting**
The sorting function ranks scenarios as:
$$
P(S_i) = \frac{e^{\lambda B_r(S_i)}}{\sum_{j} e^{\lambda B_r(S_j)}}
$$
where \( \lambda \) controls the impact of \( B_r(S_i) \).

#### **Layer 5: Final Scenario Selection**
The top \( N \) scenarios are selected as:
$$
S_{\text{final}} = \text{TopN}(S, P(S))
$$

---

### **2. Dynamic Scenario Generation**
Scenarios evolve based on the Markov process:
$$
x_{t+1} = f(x_t) + \eta_t
$$
where:
- \( f(x_t) \): Transition function for system state.
- \( \eta_t \sim \mathcal{N}(0, \sigma^2) \): Randomness at time \( t \).

Expected outcomes are calculated using Monte Carlo methods:
$$
E(O) = \frac{1}{N} \sum_{i=1}^N f(S_i, X)
$$

---

### **3. Risk and Benefit Analysis**
Probabilities are refined through Bayesian updates:
$$
P(O | S_i) = \frac{P(S_i | O) \cdot P(O)}{P(S_i)}
$$
and the final risk-benefit ratio is given by:
$$
R_b(S_i) = \frac{B(S_i)}{1 + R(S_i)}
$$

---

## Use Cases

### **1. Autonomous Systems**
Adaptability in real-time decision-making is modeled as:
$$
dx_t = f(x_t, t) dt + g(x_t, t) dW_t
$$
where:
- \( dx_t \): Change in state.
- \( g(x_t, t) \): Diffusion term introducing environmental randomness.
- \( dW_t \): Wiener process.

### **2. Strategic AI**
Simulated outcomes follow:
$$
O = \frac{1}{N} \sum_{i=1}^N f(S_i, X)
$$

### **3. Creative AI**
Scenarios are generated using variational autoencoders:
$$
p(S | z) = p(S) \cdot q(z | S)
$$
where:
- \( z \): Latent space representation.

### **4. Risk Management**
Dynamic risk balancing adjusts weights as:
$$
\text{Risk}_{\text{new}} = \text{Risk}_{\text{current}} \cdot (1 + \eta)
$$
where \( \eta \sim \mathcal{N}(0, \sigma^2) \).

---

## Why Artificial-Subconscious?
The project enhances AI adaptability, creativity, and intuition by incorporating a layered mathematical framework:
$$
\text{AI}_{\text{adaptive}} = f(\text{SubConsciousLayer})
$$
This bridges the gap between human-like reasoning and computational precision.
