"""TRPO utility functions."""
import torch


def flat_grad(grads):
    """Flatten the gradients."""
    grad_flatten = []
    for grad in grads:
        if grad is None:
            continue
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    """Flatten the hessians."""
    hessians_flatten = []
    for hessian in hessians:
        if hessian is None:
            continue
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    """Flatten the parameters."""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    """Update the model parameters."""
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index : index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def kl_approx(p, q):
    """KL divergence between two distributions."""
    r = torch.exp(q - p)
    kl = r - 1 - q + p
    return kl


def _kl_normal_normal(p, q):
    """KL divergence between two normal distributions.
    adapted from https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
    """
    var_ratio = (p.scale.to(torch.float64) / q.scale.to(torch.float64)).pow(2)
    t1 = (
        (p.loc.to(torch.float64) - q.loc.to(torch.float64)) / q.scale.to(torch.float64)
    ).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def kl_divergence(
    obs,
    rnn_states,
    action,
    masks,
    available_actions,
    active_masks,
    new_actor,
    old_actor,
):
    """KL divergence between two distributions."""
    _, _, new_dist = new_actor.evaluate_actions(
        obs, rnn_states, action, masks, available_actions, active_masks
    )
    with torch.no_grad():
        _, _, old_dist = old_actor.evaluate_actions(
            obs, rnn_states, action, masks, available_actions, active_masks
        )
    if new_dist.__class__.__name__ == "FixedCategorical":  # discrete action
        new_logits = new_dist.logits
        old_logits = old_dist.logits
        kl = kl_approx(old_logits, new_logits)
    else:  # continuous action
        kl = _kl_normal_normal(old_dist, new_dist)

    if len(kl.shape) > 1:
        kl = kl.sum(1, keepdim=True)
    return kl


# pylint: disable-next=invalid-name
def conjugate_gradient(
    actor,
    obs,
    rnn_states,
    action,
    masks,
    available_actions,
    active_masks,
    b,
    nsteps,
    device,
    residual_tol=1e-10,
):
    """Conjugate gradient algorithm.
    # refer to https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    """
    x = torch.zeros(b.size()).to(device=device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        _Avp = fisher_vector_product(
            actor, obs, rnn_states, action, masks, available_actions, active_masks, p
        )
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def fisher_vector_product(
    actor, obs, rnn_states, action, masks, available_actions, active_masks, p
):
    """Fisher vector product."""
    with torch.backends.cudnn.flags(enabled=False):
        p.detach()
        kl = kl_divergence(
            obs,
            rnn_states,
            action,
            masks,
            available_actions,
            active_masks,
            new_actor=actor,
            old_actor=actor,
        )
        kl = kl.mean()
        kl_grad = torch.autograd.grad(
            kl, actor.parameters(), create_graph=True, allow_unused=True
        )
        kl_grad = flat_grad(kl_grad)  # check kl_grad == 0
        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(
            kl_grad_p, actor.parameters(), allow_unused=True
        )
        kl_hessian_p = flat_hessian(kl_hessian_p)
        return kl_hessian_p + 0.1 * p

# --- New methods required for W-MATRPO ---
def wasserstein_distance_1d(
    policy1_state, policy2_state, obs_batch, rnn_states_batch, masks_batch
):
    """
    Calculates the 1-Wasserstein distance between two policies.
    For 1D Gaussian distributions, W1 = |mu1 - mu2| + |sigma1 - sigma2|.
    This will need to be adapted for multivariate Gaussians.

    Args:
        policy1_state: (dict) State dictionary of the first policy.
        policy2_state: (dict) State dictionary of the second policy.
        obs_batch: (torch.Tensor) Batch of observations for generating distributions.
        rnn_states_batch: (torch.Tensor) Batch of recurrent states.
        masks_batch: (torch.Tensor) Batch of masks.

    Returns:
        (torch.Tensor): The 1-Wasserstein distance.
    """
    # TODO:
    # 1. Create two temporary StochasticPolicy models.
    # 2. Load the state_dicts into the models.
    # 3. Pass the data batch through both models to get their action distributions.
    # 4. Calculate and return the 1-Wasserstein distance between the distributions.
    #    This will likely require accessing the .mean and .stddev of the distributions.
    pass


def calculate_adaptive_trust_region(
    other_agents_policy_histories, c_const, eps, obs_batch, rnn_states_batch, masks_batch
):
    """
    Calculates the adaptive trust region delta for the current agent based on the
    policy changes of the other agents.

    Args:
        other_agents_policy_histories: (list of deques) A list where each element
                                       is a deque containing the last 2 policy
                                       state_dicts for one of the other agents.
        c_const: (float) The 'C' constant from the algorithm.
        eps: (float) A small epsilon for numerical stability.
        obs_batch: (torch.Tensor) Batch of observations.
        rnn_states_batch: (torch.Tensor) Batch of recurrent states.
        masks_batch: (torch.Tensor) Batch of masks.

    Returns:
        (float): The calculated adaptive trust region delta.
    """
    # TODO:
    # 1. Initialize a denominator sum to zero.
    # 2. Iterate through each of the other agents' policy histories.
    # 3. For each other agent, get their policy at step k-1 and k-2.
    # 4. Calculate the W1 distance between these two historical policies using
    #    the wasserstein_distance_1d function.
    # 5. Add this distance to the denominator sum.
    # 6. Calculate delta = c_const / (denominator_sum + eps).
    # 7. Return delta.
    pass
