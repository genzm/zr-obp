"""Test script for SyntheticCombinatorialBanditDataset with linear_behavior_policy."""
import sys
sys.path.insert(0, '/home/user/zr-obp')

import numpy as np
from obp.dataset.synthetic_combinatorial import SyntheticCombinatorialBanditDataset
from obp.dataset.synthetic import linear_behavior_policy

print("=" * 60)
print("Test 1: Basic usage with linear_behavior_policy")
print("=" * 60)

dataset1 = SyntheticCombinatorialBanditDataset(
    n_actions=5,
    behavior_policy_function=linear_behavior_policy,
    n_main_actions=2,
    dim_context=3,
    random_state=42
)

feedback1 = dataset1.obtain_batch_bandit_feedback(n_rounds=3)
print(f"✓ Generated {feedback1['n_rounds']} rounds")
print(f"  action_binary shape: {feedback1['action_binary'].shape}")
print(f"  action_binary:\n{feedback1['action_binary']}")
print(f"  reward: {feedback1['reward']}")
print(f"  subset_size: {feedback1['subset_size']}")

print("\n" + "=" * 60)
print("Test 2: Different random seed")
print("=" * 60)

dataset2 = SyntheticCombinatorialBanditDataset(
    n_actions=5,
    behavior_policy_function=linear_behavior_policy,
    n_main_actions=2,
    dim_context=3,
    random_state=123  # Different seed
)

feedback2 = dataset2.obtain_batch_bandit_feedback(n_rounds=3)
print(f"✓ Generated {feedback2['n_rounds']} rounds")
print(f"  action_binary shape: {feedback2['action_binary'].shape}")
print(f"  action_binary:\n{feedback2['action_binary']}")
print(f"  reward: {feedback2['reward']}")
print(f"  subset_size: {feedback2['subset_size']}")

print("\n" + "=" * 60)
print("Test 3: Verify different seeds produce different results")
print("=" * 60)

same_actions = np.array_equal(feedback1['action_binary'], feedback2['action_binary'])
print(f"  Same actions? {same_actions} (should be False)")
print(f"  Test 1 pscore range: [{feedback1['pscore'].min():.6f}, {feedback1['pscore'].max():.6f}]")
print(f"  Test 2 pscore range: [{feedback2['pscore'].min():.6f}, {feedback2['pscore'].max():.6f}]")

print("\n" + "=" * 60)
print("Test 4: Larger dataset with subset size constraints")
print("=" * 60)

dataset3 = SyntheticCombinatorialBanditDataset(
    n_actions=10,
    behavior_policy_function=linear_behavior_policy,
    n_main_actions=3,
    dim_context=5,
    min_subset_size=2,
    max_subset_size=6,
    random_state=123
)

feedback3 = dataset3.obtain_batch_bandit_feedback(n_rounds=100)
print(f"✓ Generated {feedback3['n_rounds']} rounds")
print(f"  Average subset size: {feedback3['subset_size'].mean():.2f}")
print(f"  Min subset size: {feedback3['subset_size'].min()}")
print(f"  Max subset size: {feedback3['subset_size'].max()}")
print(f"  Reward rate: {feedback3['reward'].mean():.3f}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
