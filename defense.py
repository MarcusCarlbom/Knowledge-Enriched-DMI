import torch
import torch.nn.functional as F


def defend_output(logits, noise_std=0.0, top_k=0, truncate_decimals=0):
    """
    Combined output defense against model inversion attacks.

    Applies three optional and composable defenses to the model's output logits
    before they are returned to the caller:

      1. Gaussian output perturbation (noise_std > 0):
         Adds calibrated Gaussian noise to all logits, degrading the gradient
         signal the attacker uses to guide GAN image generation. A legitimate
         user's top-1 prediction is unaffected since the correct class scores
         far higher than any other class, so small noise never flips the ranking.

      2. Top-k masking (top_k > 0):
         Zeroes out all logits except the top-k highest scoring classes.
         A legitimate user only needs the top-1 (or top-5) prediction, so
         masking the remaining 990+ classes costs them nothing. The attacker
         however relies on the full distribution to compute meaningful gradients
         -- losing 99%+ of the signal makes optimization extremely difficult.

      3. Confidence truncation (truncate_decimals > 0):
         Rounds all logits to a fixed number of decimal places before returning
         them. The attacker relies on fine-grained floating point differences
         between classes to guide optimization -- truncation destroys this
         precision while leaving the top-1 ranking completely intact for
         legitimate users. First proposed as a countermeasure by Fredrikson
         et al. (CCS 2015), the original paper on confidence-based model
         inversion attacks.

    Defenses are applied in order: noise first, then top-k masking, then
    truncation. Any combination can be used independently or together.

    Args:
        logits            (Tensor): Raw output logits from the classifier [B x n_classes]
        noise_std          (float): Std of Gaussian noise to add. 0.0 disables noise. (default: 0.0)
        top_k              (int):   Number of top classes to keep. 0 disables masking. (default: 0)
        truncate_decimals  (int):   Decimal places to round to. 0 disables truncation. (default: 0)

    Returns:
        Tensor: Defended logits [B x n_classes]

    Example:
        # noise only
        out = defend_output(out, noise_std=0.02)

        # top-k only
        out = defend_output(out, top_k=10)

        # truncation only
        out = defend_output(out, truncate_decimals=2)

        # all three combined
        out = defend_output(out, noise_std=0.02, top_k=10, truncate_decimals=2)
    """

    # --- Defense 1: Gaussian output perturbation ---
    # Add noise to every logit to corrupt the fine-grained gradient signal
    # the attacker uses to steer image generation toward a target identity.
    if noise_std > 0.0:
        noise = torch.randn_like(logits) * noise_std
        logits = logits + noise

    # --- Defense 2: Top-k masking ---
    # Keep only the top-k logits and set all others to a large negative value
    # so they become effectively zero after softmax. This removes the vast
    # majority of the distribution that the attacker needs for optimization
    # while leaving the top predictions intact for legitimate users.
    if top_k > 0 and top_k < logits.size(1):
        # Find the threshold value at position k
        threshold, _ = torch.topk(logits, top_k, dim=1)
        # Get the smallest value among the top-k
        threshold = threshold[:, -1].unsqueeze(1)
        # Mask everything below the threshold by setting to large negative
        mask = logits >= threshold
        logits = logits * mask + (~mask) * (-1e9)

    # --- Defense 3: Confidence truncation ---
    # Round logits to a fixed number of decimal places, destroying the
    # fine-grained floating point precision the attacker needs to compute
    # meaningful gradient updates. Top-1 ranking is unaffected since
    # the correct class still scores highest after rounding.
    if truncate_decimals > 0:
        scale = 10 ** truncate_decimals
        logits = torch.round(logits * scale) / scale

    return logits