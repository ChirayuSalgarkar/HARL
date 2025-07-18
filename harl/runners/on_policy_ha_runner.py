"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch

# --- NEW: Import necessary utilities for policy management and KL calculation ---
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.utils.trans_tools import _t2n
from harl.utils.trpo_util import flat_params, update_model
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyHARunner(OnPolicyBaseRunner):
    """
    Runner for on-policy HA algorithms.
    This runner is modified to support adaptive trust regions by managing
    old policies and calculating inter-agent policy drift (KL divergence).
    """

    # --- REMOVED: The custom __init__ method has been removed to fix the error. ---
    # Initialization of old_policies will now happen in the train() method.

    def train(self):
        """
        Train the model. This method is overridden to implement the sequential
        update logic for HA algorithms and to calculate/pass the adaptive
        KL threshold to the training function.
        """

        # --- NEW: Lazy initialization of old_policies ---
        # This check ensures that old_policies are created only once, on the first
        # call to train(). This guarantees all attributes from the base runner's
        # __init__ (like self.agent_ids and self.actor) are available.
        if not hasattr(self, "old_policies"):
            self.old_policies = {
                agent_id: StochasticPolicy(
                    self.algo_args[agent_id]["train"],
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    self.device,
                )
                for agent_id in self.agent_ids
            }
            # Initialize old policies with the current (initial) policy parameters.
            for agent_id in self.agent_ids:
                # The .actor attribute is the actual network within the algorithm object.
                update_model(
                    self.old_policies[agent_id], flat_params(self.actor[agent_id].actor)
                )

        actor_train_infos = []

        # Factor is used for considering updates made by previous agents in HATRPO
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # Compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                         :-1
                         ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                    self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # Normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # Determine agent update order
        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())

        # --- MODIFIED: Main training loop with adaptive KL calculation ---
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(factor)

            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                     .available_actions[:-1]
                     .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # Compute action log probs for the actor before update
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id].obs[:-1].reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id].rnn_states[0:1].reshape(-1,
                                                                    *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(-1, *self.actor_buffer[agent_id].actions.shape[2:]),
                self.actor_buffer[agent_id].masks[:-1].reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id].active_masks[:-1].reshape(-1,
                                                                      *self.actor_buffer[agent_id].active_masks.shape[
                                                                       2:]),
            )

            # --- NEW: Calculate the adaptive KL threshold ---
            adaptive_kl_threshold = None
            if self.algo_args[agent_id]["train"].get("use_adaptive_kl", False):
                teammate_kl_divergence = 0.0
                num_teammates = 0

                for teammate_id in self.agent_ids:
                    if teammate_id != agent_id:
                        kl_div = self.actor[teammate_id].kl_divergence(
                            self.actor_buffer[teammate_id], self.old_policies[teammate_id]
                        ).mean()
                        teammate_kl_divergence += kl_div.item()
                        num_teammates += 1

                if num_teammates > 0:
                    avg_teammate_kl = teammate_kl_divergence / num_teammates
                    C = self.algo_args[agent_id]["train"].get("adaptive_kl_C", 1.0)
                    eps = self.algo_args[agent_id]["train"].get("adaptive_kl_epsilon", 1e-8)
                    adaptive_kl_threshold = C / (avg_teammate_kl + eps)

            # --- MODIFIED: Pass the threshold to the trainer's train method ---
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages.copy(), "EP",
                    adaptive_kl_threshold=adaptive_kl_threshold
                )
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP",
                    adaptive_kl_threshold=adaptive_kl_threshold
                )

            # Compute action log probs for the updated agent
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id].obs[:-1].reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id].rnn_states[0:1].reshape(-1,
                                                                    *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(-1, *self.actor_buffer[agent_id].actions.shape[2:]),
                self.actor_buffer[agent_id].masks[:-1].reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id].active_masks[:-1].reshape(-1,
                                                                      *self.actor_buffer[agent_id].active_masks.shape[
                                                                       2:]),
            )

            # Update factor for the next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # --- NEW: Update old policies after all agents have been trained ---
        for agent_id in self.agent_ids:
            # The .actor attribute is the actual network within the algorithm object.
            update_model(
                self.old_policies[agent_id], flat_params(self.actor[agent_id].actor)
            )

        # Update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info