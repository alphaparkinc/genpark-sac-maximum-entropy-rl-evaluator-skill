# genpark-sac-maximum-entropy-rl-evaluator-skill

Agent Skill implementing the **Soft Actor-Critic (SAC) Maximum Entropy Reinforcement Learning Formulation**, incentivizing policy exploration through Shannon entropy regularization.

## Architectural Overview
```mermaid
flowchart TD
    QValues["Action-Value Q(s, a)"] & Probs["Action Probabilities pi(a|s)"] --> Entropy["Compute Entropy H(pi) = - sum pi log pi"]
    QValues & Probs & Entropy --> SoftVal["Soft State Value: V(s) = sum pi(a|s)[Q(s, a) - alpha * log pi(a|s)]"]
    SoftVal & Reward["Reward r"] --> SoftTarget["Soft Bellman Target: y = r + gamma * V(s')"]
```
