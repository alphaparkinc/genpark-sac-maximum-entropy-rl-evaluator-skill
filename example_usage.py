from client import SACMaximumEntropyEvaluator

sac = SACMaximumEntropyEvaluator(alpha=0.2, gamma=0.99)
res = sac.soft_value([2.0, 4.0], [0.5, 0.5])
target = sac.soft_bellman_target(1.0, res['soft_value'], done=False)

print(f"Soft Value: {res['soft_value']:.4f}, Shannon Entropy: {res['entropy']:.4f}")
print(f"Soft Bellman Target: {target:.4f}")
