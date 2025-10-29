# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic Logged Bandit Data for Combinatorial Bandits."""
from dataclasses import dataclass
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from ..types import BanditFeedback
from ..utils import check_array
from ..utils import sigmoid
from .base import BaseBanditDataset


@dataclass
class SyntheticCombinatorialBanditDataset(BaseBanditDataset):
    """Class for synthesizing combinatorial bandit dataset.

    Note
    -----
    This class generates logged bandit feedback for Contextual Combinatorial Bandits (CCB),
    where the action space consists of all possible subsets of a candidate action set.

    Unlike slate/ranking problems:
    1. Order doesn't matter (subset selection, not ranking)
    2. Single binary reward for the entire combination
    3. Factored action space with binary indicators

    The reward function is decomposed into:
    - Main effect: q_main(x, s_main) from main actions
    - Residual effect: q_residual(x, s) from all actions + interactions

    Parameters
    -----------
    n_actions: int
        Number of candidate actions. Action space size = 2^n_actions.

    n_main_actions: int, default=None
        Number of main actions. If None, defaults to n_actions // 2.

    dim_context: int, default=1
        Number of dimensions of context vectors.

    main_effect_weight: float, default=0.7
        Weight of main effect in [0, 1].
        q(x,s) = α*q_main + (1-α)*q_residual

    interaction_strength: float, default=0.3
        Strength of pairwise interaction effects in [0, 1].

    base_reward_function: Callable, default=None
        Function f: X × A → R for individual action rewards.
        If None, uses context-independent uniform random values.

    behavior_policy_function: Callable
        Function that returns logits for behavior policy: (context, action_context) → logits.
        Example: linear_behavior_policy from obp.dataset
        Required parameter.
        Each action is sampled independently using Bernoulli trials with probabilities
        derived from sigmoid(logits).

    min_subset_size: int, default=1
        Minimum number of actions to select.

    max_subset_size: int, default=None
        Maximum number of actions to select. If None, = n_actions.

    random_state: int, default=12345
        Random seed.

    dataset_name: str, default='synthetic_combinatorial_bandit_dataset'
        Dataset name.

    Examples
    ----------
    >>> from obp.dataset import SyntheticCombinatorialBanditDataset, linear_behavior_policy
    >>> dataset = SyntheticCombinatorialBanditDataset(
            n_actions=5,
            n_main_actions=2,
            dim_context=3,
            behavior_policy_function=linear_behavior_policy,
            random_state=12345
        )
    >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=100)

    References
    ------------
    Yuta Saito and Thorsten Joachims.
    "Off-Policy Evaluation for Large Action Spaces via Embeddings", 2022.

    """

    n_actions: int
    behavior_policy_function: Callable[[np.ndarray, np.ndarray], np.ndarray]
    n_main_actions: Optional[int] = None
    dim_context: int = 1
    main_effect_weight: float = 0.7
    interaction_strength: float = 0.3
    base_reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    min_subset_size: int = 1
    max_subset_size: Optional[int] = None
    random_state: int = 12345
    dataset_name: str = "synthetic_combinatorial_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.dim_context, "dim_context", int, min_val=1)

        if self.n_main_actions is None:
            self.n_main_actions = self.n_actions // 2
        check_scalar(self.n_main_actions, "n_main_actions", int, min_val=1, max_val=self.n_actions)

        check_scalar(self.main_effect_weight, "main_effect_weight", float, min_val=0.0, max_val=1.0)
        check_scalar(self.interaction_strength, "interaction_strength", float, min_val=0.0, max_val=1.0)
        check_scalar(self.min_subset_size, "min_subset_size", int, min_val=1, max_val=self.n_actions)

        if self.max_subset_size is None:
            self.max_subset_size = self.n_actions
        check_scalar(self.max_subset_size, "max_subset_size", int, min_val=self.min_subset_size, max_val=self.n_actions)

        self.random_ = check_random_state(self.random_state)
        self.action_context = np.eye(self.n_actions, dtype=int)

        # Randomly select main actions
        self.main_action_indices = self.random_.choice(
            self.n_actions, size=self.n_main_actions, replace=False
        )

        # Generate interaction weight matrix
        self.interaction_weights = self._generate_interaction_weights()

    def _generate_interaction_weights(self) -> np.ndarray:
        """Generate symmetric interaction weight matrix between actions.

        Returns
        --------
        interaction_weights: array-like, shape (n_actions, n_actions)
            Symmetric matrix where W[i,j] represents interaction strength between actions i and j.
        """
        # Generate random symmetric matrix
        base_matrix = self.random_.normal(
            scale=self.interaction_strength,
            size=(self.n_actions, self.n_actions)
        )
        # Make it symmetric
        symmetric_matrix = (base_matrix + base_matrix.T) / 2
        # Zero out diagonal (no self-interaction)
        np.fill_diagonal(symmetric_matrix, 0)
        return symmetric_matrix

    def calc_individual_expected_rewards(self, context: np.ndarray) -> np.ndarray:
        """Calculate expected reward for each individual action given context.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors.

        Returns
        --------
        individual_rewards: array-like, shape (n_rounds, n_actions)
            Expected reward for each individual action.
        """
        check_array(array=context, name="context", expected_dim=2)

        n_rounds = context.shape[0]

        if self.base_reward_function is None:
            # Context-independent rewards
            base_rewards = self.random_.uniform(0.3, 0.7, size=self.n_actions)
            individual_rewards = np.tile(base_rewards, (n_rounds, 1))
        else:
            # Context-dependent rewards
            individual_rewards = self.base_reward_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state
            )

        return individual_rewards

    def calc_expected_reward_for_combination(
        self,
        context: np.ndarray,
        action_binary: np.ndarray,
    ) -> np.ndarray:
        """Calculate expected reward for given action combinations.

        This implements the decomposition:
        q(x, s) = main_effect_weight * q_main(x, s_main) + (1 - main_effect_weight) * q_residual(x, s)

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors.

        action_binary: array-like, shape (n_rounds, n_actions)
            Binary matrix indicating which actions are included (1) or not (0) for each round.

        Returns
        --------
        expected_reward: array-like, shape (n_rounds,)
            Expected reward for each combination.
        """
        check_array(array=context, name="context", expected_dim=2)
        check_array(array=action_binary, name="action_binary", expected_dim=2)

        n_rounds = context.shape[0]

        # Get individual expected rewards
        individual_rewards = self.calc_individual_expected_rewards(context)

        # Calculate main effect (from main actions only)
        main_action_mask = np.zeros(self.n_actions)
        main_action_mask[self.main_action_indices] = 1.0

        # Main actions that are selected
        main_selected = action_binary * main_action_mask
        main_effect = (individual_rewards * main_selected).sum(axis=1)

        # Normalize by number of main actions selected (avoid division by zero)
        n_main_selected = main_selected.sum(axis=1)
        n_main_selected = np.maximum(n_main_selected, 1.0)  # Avoid division by zero
        main_effect = main_effect / n_main_selected

        # Calculate residual effect (including all actions + interactions)
        # Base: sum of all selected actions
        residual_base = (individual_rewards * action_binary).sum(axis=1)

        # Add interaction effects
        interaction_effect = np.zeros(n_rounds)
        for i in range(n_rounds):
            selected_actions = np.where(action_binary[i] == 1)[0]
            # Sum pairwise interactions for selected action pairs
            for j, action_j in enumerate(selected_actions):
                for action_k in selected_actions[j+1:]:
                    interaction_effect[i] += self.interaction_weights[action_j, action_k]

        # Normalize residual by number of selected actions
        n_selected = action_binary.sum(axis=1)
        n_selected = np.maximum(n_selected, 1.0)  # Avoid division by zero
        residual_effect = (residual_base + interaction_effect) / n_selected

        # Combine main and residual effects
        expected_reward = (
            self.main_effect_weight * main_effect +
            (1.0 - self.main_effect_weight) * residual_effect
        )

        # Apply sigmoid to get probabilities in [0, 1]
        expected_reward = sigmoid(expected_reward)

        return expected_reward

    def sample_reward_given_expected_reward(
        self,
        expected_reward: np.ndarray,
    ) -> np.ndarray:
        """Sample binary rewards from Bernoulli distribution.

        Parameters
        -----------
        expected_reward: array-like, shape (n_rounds,)
            Expected rewards (probabilities in [0, 1]).

        Returns
        --------
        reward: array-like, shape (n_rounds,)
            Binary rewards (0 or 1).
        """
        check_array(array=expected_reward, name="expected_reward", expected_dim=1)
        return self.random_.binomial(n=1, p=expected_reward)

    def sample_action_and_obtain_pscore(
        self,
        context: np.ndarray,
        n_rounds: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample action combinations using behavior policy.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors.

        n_rounds: int
            Number of rounds.

        Returns
        --------
        action_binary: array-like, shape (n_rounds, n_actions)
            Binary indicators of selected actions.

        pscore: array-like, shape (n_rounds,)
            Propensity score for each selected combination.

        pscore_factorized: array-like, shape (n_rounds,)
            Factorized propensity score (product of marginal probabilities).
        """
        action_binary = np.zeros((n_rounds, self.n_actions), dtype=int)
        pscore = np.zeros(n_rounds)
        pscore_factorized = np.zeros(n_rounds)

        # Calculate behavior policy logits
        behavior_logits = self.behavior_policy_function(
            context=context,
            action_context=self.action_context,
            random_state=self.random_state
        )
        inclusion_probs = sigmoid(behavior_logits)

        # Each action is selected independently using Bernoulli trials
        # p(a_l = 1) = sigmoid(behavior_logits[i, l])
        for i in range(n_rounds):
            for j in range(self.n_actions):
                # Sample whether to include this action
                include = self.random_.binomial(1, inclusion_probs[i, j])
                action_binary[i, j] = include

            # Ensure subset size constraints
            subset_size = action_binary[i].sum()
            if subset_size < self.min_subset_size or subset_size > self.max_subset_size:
                # Resample to satisfy constraints
                valid_size = self.random_.randint(self.min_subset_size, self.max_subset_size + 1)
                selected_indices = self.random_.choice(
                    self.n_actions, size=valid_size, replace=False,
                    p=inclusion_probs[i] / inclusion_probs[i].sum()
                )
                action_binary[i] = 0
                action_binary[i, selected_indices] = 1

            # Calculate factorized propensity score
            pscore_factorized[i] = np.prod(
                inclusion_probs[i, action_binary[i] == 1]
            ) * np.prod(
                1 - inclusion_probs[i, action_binary[i] == 0]
            )
            pscore[i] = pscore_factorized[i]

        # Ensure no zero probabilities (numerical stability)
        pscore = np.maximum(pscore, 1e-10)
        pscore_factorized = np.maximum(pscore_factorized, 1e-10)

        return action_binary, pscore, pscore_factorized

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedback:
        """Obtain batch logged bandit data for combinatorial bandits.

        Parameters
        ----------
        n_rounds: int
            Data size of the synthetic logged bandit data.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Synthesized combinatorial bandit logged data with the following keys:
            - n_rounds: Number of samples
            - n_actions: Number of candidate actions
            - context: Context vectors (n_rounds, dim_context)
            - action_context: Action feature vectors (n_actions, n_actions)
            - action: Indices of selected actions for each round (variable length list)
            - action_binary: Binary indicators (n_rounds, n_actions)
            - subset_size: Number of actions selected in each round (n_rounds,)
            - main_action_flags: Binary indicators for main actions (n_actions,)
            - reward: Observed rewards (n_rounds,)
            - expected_reward: Expected rewards for selected combinations (n_rounds,)
            - pscore: Propensity scores (n_rounds,)
            - pscore_factorized: Factorized propensity scores (n_rounds,)
        """
        check_scalar(n_rounds, "n_rounds", int, min_val=1)

        # Sample contexts
        context = self.random_.normal(size=(n_rounds, self.dim_context))

        # Sample actions using behavior policy
        action_binary, pscore, pscore_factorized = self.sample_action_and_obtain_pscore(
            context=context,
            n_rounds=n_rounds,
        )

        # Calculate expected reward for selected combinations
        expected_reward = self.calc_expected_reward_for_combination(
            context=context,
            action_binary=action_binary,
        )

        # Sample rewards
        reward = self.sample_reward_given_expected_reward(expected_reward)

        # Extract subset sizes
        subset_size = action_binary.sum(axis=1)

        # Create main action flags
        main_action_flags = np.zeros(self.n_actions)
        main_action_flags[self.main_action_indices] = 1

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=context,
            action_context=self.action_context,
            action_binary=action_binary,
            subset_size=subset_size,
            main_action_flags=main_action_flags,
            reward=reward,
            expected_reward=expected_reward,
            pscore=pscore,
            pscore_factorized=pscore_factorized,
        )


def logistic_combinatorial_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic reward function for combinatorial bandits.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors.

    action_context: array-like, shape (n_actions, dim_action_context)
        Action feature vectors.

    random_state: int, default=None
        Random state for generating coefficients.

    Returns
    --------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward for each action.
    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)

    random_ = check_random_state(random_state)

    # Generate random coefficients
    context_coef = random_.uniform(-1, 1, size=context.shape[1])
    action_coef = random_.uniform(-1, 1, size=action_context.shape[1])

    # Calculate logits
    logits = context @ context_coef[:, np.newaxis] + action_context @ action_coef
    expected_reward = sigmoid(logits.T)

    return expected_reward
