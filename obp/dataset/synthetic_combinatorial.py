# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic Logged Bandit Data for Combinatorial Bandits."""
from dataclasses import dataclass
from itertools import combinations
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.special import comb
from scipy.stats import truncnorm
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from ..types import BanditFeedback
from ..utils import check_array
from ..utils import sigmoid
from ..utils import softmax
from .base import BaseBanditDataset
from .reward_type import RewardType


@dataclass
class SyntheticCombinatorialBanditDataset(BaseBanditDataset):
    """Class for synthesizing combinatorial bandit dataset.

    Note
    -----
    This class generates logged bandit feedback for Contextual Combinatorial Bandits (CCB),
    where the action space consists of all possible subsets of a candidate action set.

    Unlike slate/ranking problems where position matters and individual slot rewards are observed,
    CCB involves:
    1. Selecting a subset of actions (order doesn't matter)
    2. Observing a single reward for the entire combination
    3. Using factored action space: binary indicators m_l ∈ {∅, a_l} for each action

    The reward function is decomposed into:
    - Main effect: q_main(x, s_main) from main actions
    - Residual effect: Δq_residual(x, s, s_main) from auxiliary actions

    Parameters
    -----------
    n_actions: int
        Number of candidate actions in the base action set A.
        The combinatorial action space will have size 2^n_actions.

    n_main_actions: int, default=None
        Number of main actions that contribute most to the reward.
        If None, defaults to n_actions // 2.

    dim_context: int, default=1
        Number of dimensions of context vectors.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary', rewards are sampled from the Bernoulli distribution.
        When 'continuous', rewards are sampled from the truncated Normal distribution.

    main_effect_weight: float, default=0.7
        Weight of main effect in total reward (between 0 and 1).
        Total reward = main_effect_weight * main_effect + (1 - main_effect_weight) * residual_effect.

    interaction_strength: float, default=0.3
        Strength of interaction effects between actions (between 0 and 1).
        Higher values mean stronger synergistic or antagonistic effects.

    base_reward_function: Callable, default=None
        Function defining the expected reward for each individual action given context,
        i.e., f: X × A → R.
        If None, context-independent rewards will be sampled from uniform distribution.

    behavior_policy_type: str, default='independent'
        Type of behavior policy for selecting combinations:
        - 'independent': Each action is included independently based on its marginal probability
        - 'epsilon_greedy': Epsilon-greedy policy based on expected rewards
        - 'boltzmann': Boltzmann exploration with temperature parameter

    epsilon: float, default=0.1
        Exploration parameter for epsilon-greedy behavior policy (only used when behavior_policy_type='epsilon_greedy').

    temperature: float, default=1.0
        Temperature parameter for Boltzmann exploration (only used when behavior_policy_type='boltzmann').

    min_subset_size: int, default=0
        Minimum size of selected subset (number of actions to include).

    max_subset_size: int, default=None
        Maximum size of selected subset. If None, defaults to n_actions.

    random_state: int, default=12345
        Controls the random seed in sampling synthetic combinatorial bandit data.

    dataset_name: str, default='synthetic_combinatorial_bandit_dataset'
        Name of the dataset.

    Examples
    ----------

    .. code-block:: python

        >>> from obp.dataset import SyntheticCombinatorialBanditDataset

        # Generate synthetic combinatorial bandit feedback
        >>> dataset = SyntheticCombinatorialBanditDataset(
                n_actions=5,
                n_main_actions=2,
                dim_context=3,
                reward_type='binary',
                main_effect_weight=0.7,
                interaction_strength=0.3,
                behavior_policy_type='independent',
                random_state=12345
            )
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=100)
        >>> bandit_feedback.keys()
        dict_keys(['n_rounds', 'n_actions', 'context', 'action_context',
                   'action', 'action_binary', 'subset_size', 'main_action_flags',
                   'reward', 'expected_reward', 'pscore', 'pscore_factorized'])

    References
    ------------
    Yuta Saito and Thorsten Joachims.
    "Off-Policy Evaluation for Large Action Spaces via Embeddings", 2022.

    """

    n_actions: int
    n_main_actions: Optional[int] = None
    dim_context: int = 1
    reward_type: str = RewardType.BINARY.value
    main_effect_weight: float = 0.7
    interaction_strength: float = 0.3
    base_reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    behavior_policy_type: str = "independent"
    epsilon: float = 0.1
    temperature: float = 1.0
    min_subset_size: int = 0
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
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0, max_val=1.0)
        check_scalar(self.temperature, "temperature", float, min_val=0.0)
        check_scalar(self.min_subset_size, "min_subset_size", int, min_val=0, max_val=self.n_actions)

        if self.max_subset_size is None:
            self.max_subset_size = self.n_actions
        check_scalar(self.max_subset_size, "max_subset_size", int, min_val=self.min_subset_size, max_val=self.n_actions)

        if self.reward_type not in ["binary", "continuous"]:
            raise ValueError(
                f"`reward_type` must be either 'binary' or 'continuous', but {self.reward_type} is given."
            )

        if self.behavior_policy_type not in ["independent", "epsilon_greedy", "boltzmann"]:
            raise ValueError(
                f"`behavior_policy_type` must be one of 'independent', 'epsilon_greedy', or 'boltzmann', "
                f"but {self.behavior_policy_type} is given."
            )

        self.random_ = check_random_state(self.random_state)

        # Set reward bounds for continuous rewards
        if self.reward_type == "continuous":
            self.reward_min = 0
            self.reward_max = 1e10
            self.reward_std = 1.0

        # One-hot encoding for each action
        self.action_context = np.eye(self.n_actions, dtype=int)

        # Randomly select which actions are "main" actions
        self.main_action_indices = self.random_.choice(
            self.n_actions, size=self.n_main_actions, replace=False
        )

        # Generate interaction weight matrix for action pairs
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

        # Apply sigmoid for binary rewards to get probabilities in [0, 1]
        if self.reward_type == "binary":
            expected_reward = sigmoid(expected_reward)
        else:
            # Clip to ensure non-negative
            expected_reward = np.maximum(expected_reward, 0)

        return expected_reward

    def sample_reward_given_expected_reward(
        self,
        expected_reward: np.ndarray,
    ) -> np.ndarray:
        """Sample rewards given expected rewards.

        Parameters
        -----------
        expected_reward: array-like, shape (n_rounds,)
            Expected rewards.

        Returns
        --------
        reward: array-like, shape (n_rounds,)
            Sampled rewards.
        """
        check_array(array=expected_reward, name="expected_reward", expected_dim=1)

        if self.reward_type == "binary":
            reward = self.random_.binomial(n=1, p=expected_reward)
        elif self.reward_type == "continuous":
            mean = expected_reward
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            reward = truncnorm.rvs(
                a=a, b=b, loc=mean, scale=self.reward_std,
                random_state=self.random_state
            )
        else:
            raise NotImplementedError

        return reward

    def sample_action_and_obtain_pscore(
        self,
        context: np.ndarray,
        individual_rewards: np.ndarray,
        n_rounds: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample action combinations and calculate propensity scores.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors.

        individual_rewards: array-like, shape (n_rounds, n_actions)
            Expected reward for each individual action.

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

        if self.behavior_policy_type == "independent":
            # Each action is selected independently with probability based on its expected reward
            # p(a_l = 1) = sigmoid(reward_l)
            inclusion_probs = sigmoid(individual_rewards)

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

        elif self.behavior_policy_type == "epsilon_greedy":
            # Epsilon-greedy: with probability epsilon, select random subset;
            # otherwise select greedy subset based on expected combination reward

            for i in range(n_rounds):
                if self.random_.random() < self.epsilon:
                    # Random subset
                    subset_size = self.random_.randint(self.min_subset_size, self.max_subset_size + 1)
                    selected_indices = self.random_.choice(
                        self.n_actions, size=subset_size, replace=False
                    )
                    action_binary[i, selected_indices] = 1

                    # Uniform probability over all valid subsets
                    n_valid_subsets = sum([
                        comb(self.n_actions, k, exact=True)
                        for k in range(self.min_subset_size, self.max_subset_size + 1)
                    ])
                    pscore[i] = self.epsilon / n_valid_subsets
                else:
                    # Greedy: select top actions based on individual rewards
                    greedy_size = min(self.max_subset_size, max(self.min_subset_size, self.n_main_actions))
                    top_indices = np.argsort(individual_rewards[i])[-greedy_size:]
                    action_binary[i, top_indices] = 1
                    pscore[i] = 1.0 - self.epsilon

                # Factorized pscore (independent assumption)
                inclusion_probs = sigmoid(individual_rewards[i])
                pscore_factorized[i] = np.prod(
                    inclusion_probs[action_binary[i] == 1]
                ) * np.prod(
                    1 - inclusion_probs[action_binary[i] == 0]
                )

        elif self.behavior_policy_type == "boltzmann":
            # Boltzmann exploration over subsets
            # This is computationally expensive for large action spaces

            for i in range(n_rounds):
                # For computational efficiency, sample subset size first
                subset_size = self.random_.randint(self.min_subset_size, self.max_subset_size + 1)

                # Calculate scores for all possible subsets of this size
                # (This is expensive - in practice, use approximations)
                if subset_size <= 5 and self.n_actions <= 10:
                    # Enumerate all subsets
                    all_subsets = list(combinations(range(self.n_actions), subset_size))
                    subset_scores = []

                    for subset in all_subsets:
                        temp_binary = np.zeros(self.n_actions)
                        temp_binary[list(subset)] = 1
                        score = (individual_rewards[i] * temp_binary).sum()
                        subset_scores.append(score)

                    subset_scores = np.array(subset_scores)
                    subset_probs = softmax(subset_scores / self.temperature)

                    # Sample subset
                    selected_subset_idx = self.random_.choice(len(all_subsets), p=subset_probs)
                    selected_subset = all_subsets[selected_subset_idx]
                    action_binary[i, list(selected_subset)] = 1
                    pscore[i] = subset_probs[selected_subset_idx]
                else:
                    # Approximation: sample actions with Boltzmann probabilities
                    action_probs = softmax(individual_rewards[i] / self.temperature)
                    selected_indices = self.random_.choice(
                        self.n_actions, size=subset_size, replace=False, p=action_probs
                    )
                    action_binary[i, selected_indices] = 1
                    pscore[i] = np.prod(action_probs[selected_indices])

                # Factorized pscore
                inclusion_probs = sigmoid(individual_rewards[i])
                pscore_factorized[i] = np.prod(
                    inclusion_probs[action_binary[i] == 1]
                ) * np.prod(
                    1 - inclusion_probs[action_binary[i] == 0]
                )

        # Ensure no zero probabilities (for numerical stability)
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

        # Calculate individual expected rewards
        individual_rewards = self.calc_individual_expected_rewards(context)

        # Sample actions and calculate propensity scores
        action_binary, pscore, pscore_factorized = self.sample_action_and_obtain_pscore(
            context=context,
            individual_rewards=individual_rewards,
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
