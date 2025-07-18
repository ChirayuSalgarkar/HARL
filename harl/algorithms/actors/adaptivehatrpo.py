"""Adaptive HATRPO algorithm."""

import numpy as np
import torch

from harl.algorithms.actors.hatrpo import HATRPO
from harl.utils.trpo_util import kl_divergence
from harl.utils.envs_tools import check


class AdaptiveHATRPO(HATRPO):
    """
    AdaptiveHATRPO extends the HATRPO algorithm by implementing an adaptive
    [cite_start]trust region, as inspired by your paper[cite: 1]. The trust region radius (KL threshold)
    is dynamically adjusted for each agent based on the policy drift of its teammates.
    This is intended to improve stability during training by being more conservative
    when other agents' policies are changing rapidly.
    """

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """
        Initializes the AdaptiveHATRPO algorithm.
        It calls the parent HATRPO constructor and adds new hyperparameters
        for the adaptive mechanism.

        Args:
            args: (dict) arguments, expected to contain new keys for adaptive KL.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        # Call the constructor of the parent class (HATRPO)
        super(AdaptiveHATRPO, self).__init__(args, obs_space, act_space, device)

        # --- NEW HYPERPARAMETERS FOR ADAPTIVE KL ---
        # Documenting the new parameters that will be controlled from a config file.

        # Flag to enable or disable the adaptive KL feature for easy experimentation.
        # Default: True
        self.use_adaptive_kl = args.get("use_adaptive_kl", True)

        # The constant 'C' in the adaptive radius formula: delta = C / (sum_kl + eps).
        # Default: 1.0
        self.adaptive_kl_C = args.get("adaptive_kl_C", 1.0)

        # The small epsilon value to prevent division by zero.
        # Default: 1e-8
        self.adaptive_kl_epsilon = args.get("adaptive_kl_epsilon", 1e-8)

    def train(self, actor_buffer, advantages, state_type, adaptive_kl_threshold=None):
        """
        Perform a training update. This method is overridden to accept the
        `adaptive_kl_threshold` calculated by the runner.

        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer with training data.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
            adaptive_kl_threshold: (float, optional) The dynamically calculated KL
                                   threshold. If None, falls back to fixed threshold.

        Returns:
            train_info: (dict) contains training update information.
        """
        train_info = {}
        train_info["kl"] = 0
        train_info["dist_entropy"] = 0
        train_info["loss_improve"] = 0
        train_info["expected_improve"] = 0
        train_info["ratio"] = 0

        # --- NEW: Add the calculated adaptive KL to the log ---
        if self.use_adaptive_kl and adaptive_kl_threshold is not None:
            train_info["adaptive_kl_threshold"] = adaptive_kl_threshold

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self.use_recurrent_policy:
            data_generator = actor_buffer.recurrent_generator_actor(
                advantages, 1, self.data_chunk_length
            )
        elif self.use_naive_recurrent_policy:
            data_generator = actor_buffer.naive_recurrent_generator_actor(advantages, 1)
        else:
            data_generator = actor_buffer.feed_forward_generator_actor(advantages, 1)

        for sample in data_generator:
            # --- MODIFIED: Pass the adaptive_kl_threshold to the update method ---
            kl, loss_improve, expected_improve, dist_entropy, imp_weights = self.update(
                sample, adaptive_kl_threshold
            )

            train_info["kl"] += kl
            train_info["loss_improve"] += loss_improve.item()
            train_info["expected_improve"] += expected_improve
            train_info["dist_entropy"] += dist_entropy.item()
            train_info["ratio"] += imp_weights.mean()

        num_updates = 1

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def update(self, sample, adaptive_kl_threshold=None):
        """
        Update actor network. This is overridden to use the adaptive KL threshold.

        Args:
            sample: (Tuple) contains data batch with which to update networks.
            adaptive_kl_threshold: (float, optional) The dynamically calculated KL
                                   threshold from the runner.

        Returns:
            kl, loss_improve, expected_improve, dist_entropy, ratio
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        # --- NEW: Determine which KL threshold to use ---
        # This logic uses the passed adaptive_kl_threshold if the feature is enabled
        # and a value is provided. Otherwise, it falls back to the fixed threshold
        # from the original HATRPO config.
        if self.use_adaptive_kl and adaptive_kl_threshold is not None:
            kl_threshold = adaptive_kl_threshold
        else:
            kl_threshold = self.kl_threshold

        # The rest of this method is identical to HATRPO's update, but uses the
        # dynamically determined `kl_threshold` instead of `self.kl_threshold`.

        # Note: The following code is from hatrpo.py, with self.kl_threshold
        # replaced by our local `kl_threshold` variable.

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch, rnn_states_batch, actions_batch, masks_batch,
            available_actions_batch, active_masks_batch
        )

        ratio = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1, keepdim=True
        )

        loss = self.compute_loss(ratio, factor_batch, adv_targ, active_masks_batch)

        loss_grad = torch.autograd.grad(loss, self.actor.parameters(), allow_unused=True)
        loss_grad = flat_grad(loss_grad)

        step_dir = self.compute_step_direction(loss_grad, obs_batch, rnn_states_batch, actions_batch, masks_batch,
                                               available_actions_batch, active_masks_batch)

        # --- MODIFIED: Use the dynamic kl_threshold ---
        shs = 0.5 * (step_dir * fvp).sum(0, keepdim=True)
        step_size = torch.sqrt(kl_threshold / shs)
        full_step = step_size * step_dir

        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)

        # Backtracking line search
        new_loss, kl, flag = self.line_search(loss, full_step, expected_improve, obs_batch, rnn_states_batch,
                                              actions_batch, masks_batch, available_actions_batch, active_masks_batch,
                                              old_action_log_probs_batch, factor_batch, adv_targ, kl_threshold)

        loss_improve = new_loss - loss.data.cpu().numpy()

        if not flag:
            self.actor.load_state_dict(self.old_actor.state_dict())
            print("Policy update does not improve the surrogate.")

        return kl, loss_improve, expected_improve, dist_entropy, ratio