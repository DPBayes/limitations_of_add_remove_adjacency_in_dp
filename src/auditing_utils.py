"""
Utility functions for auditing
"""
import math
import torch
import logging
import numpy as np
from torch import nn
from scipy.stats import norm
from statsmodels.stats import proportion
from scipy.optimize import brentq
from typing import Tuple
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, roc_curve
import functools
from typing import Optional
import numpy as np

def flat_grad(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])

def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.flatten() for p in model.parameters() if p.requires_grad])

def no_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_per_sample_grads(gradients) -> torch.Tensor:
    reshaped_gradients = [
                        g.reshape(len(g), -1) for g in gradients
                    ]     
    return torch.concat(reshaped_gradients, dim=1)

def param_shapes(model: nn.Module) -> Tuple[Tuple[int, ...], ...]:
    param_shapes = []
    for p in model.parameters():
        param_shapes.append(p.shape)
    return tuple(param_shapes)


def find_index(target_index: int, shapes: Tuple[Tuple, ...]) -> Tuple[int, ...]:
    """
    Find the index of a flattened parameter vector in the original parameter vector.
    :param target_index: the index of the flattened parameter vector
    :param shapes: the shapes of the unflattened parameters
    :return: the multidimensional index of the unflattened parameters
    """
    current_index = 0

    for layer_idx, shape in enumerate(shapes):
        new_index = 1
        for shape_dim in shape:
            new_index *= shape_dim

        new_index += current_index

        if current_index <= target_index < new_index:
            idx = np.unravel_index((target_index - current_index), tuple(shape))
            return layer_idx, idx

        current_index = new_index

    raise ValueError


@torch.no_grad()
def poison_model( model: nn.Module, reduction: str,
                parameter_index, learning_rate, batch_size, 
                C) -> None:
    """
    Check if the current step should be poisoned. If yes,
    the targeted parameter has its magnitude changed as
    following.
    1. + lr/batch_size * C if loss_reduction == "mean"
    2. + lr * C if loss_reduction == "sum"

    :param cfg: The hyperparameters of the experiment
    :param model: The attacked model
    :param step: The current step of optimization
    :return: Nothing, the poisoning is done in-place.
    """
    logging.info(f"Poisoning model with a parameter index {parameter_index}")

    layer_target_idx, idx = find_index(parameter_index, param_shapes(model))
    for layer_idx, p in enumerate(model.parameters()):
        if layer_idx == layer_target_idx:
            logging.info(f"Previous poisoned value: {p[idx].item():.2f}")

            if reduction == "mean":
                magnitude = learning_rate * C / batch_size
            elif reduction == "sum":
                magnitude = learning_rate * C
            else:
                raise ValueError("Unknown loss reduction")

            p[idx].sub_(magnitude) 

            logging.info(f"After poisoned value: {p[idx].item():.2f}")
            return

    raise ValueError("Poisoning Failed")


def get_poison_attack_output(model: nn.Module, anomaly_gradient_feature: int) -> float:
    """
    Retrieve the value of the poisoned parameter.
    :param model: the attacked model
    :param anomaly_gradient_feature:
    :return:
    """
    layer_target_idx, idx = find_index(anomaly_gradient_feature, param_shapes(model))

    for layer_idx, p in enumerate(model.parameters()):
        if layer_idx == layer_target_idx:
            return float(p[idx])

    raise ValueError("No layer found")


def get_bounds_gdp(
    fpr: float,
    fnr: float,
    eps_tol=1e-12,
    delta: float = 1e-5
):
    mean = norm.ppf(1.0 - fpr) - norm.ppf(fnr)
    def search_fn(eps):
        return (
                delta
                - norm.cdf(-eps / mean + mean / 2)
                + np.exp(eps) * norm.cdf(-eps / mean - mean / 2)
        )

    return brentq(search_fn, a=-25, b=200, xtol=eps_tol)


def get_bounds_clopper(
    fnr: float,
    fpr: float,
    delta: np.float64
) -> np.float64:
    b1_lb = np.log((1 - delta - fpr) / fnr)
    b2_lb = np.log((1 - delta - fnr) / fpr)

    results = np.array([0, b1_lb, b2_lb])
    return np.nanmax(results[np.isfinite(results)])


def get_privacy_profile(
    delta, outputs: np.ndarray, gt: np.ndarray, alpha=0.05, method="beta"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def alpha_beta_single(threshold: float) -> Tuple[float, float, float]:
        predictions = outputs > threshold
        tn, fp, fn, tp = confusion_matrix(gt, predictions).ravel()

        sum = tn + tp + fp + fn

        if (tn + tp) / sum < (fp + fn) / sum:
            tn, fp = fp, tn
            fn, tp = tp, fn

        _, p0_ub = proportion.proportion_confint(fp, tn + fp, alpha, method=method)
        _, p1_ub = proportion.proportion_confint(fn, tp + fn, alpha, method=method)

        if p0_ub < delta or \
                p1_ub < delta or\
                p0_ub > 1 - delta or \
                p1_ub > 1 - delta or \
                p0_ub + p1_ub > 1:
            return 0., 0., 0., 0.

        
        try:
            eps_lb_gdp = get_bounds_gdp(fpr=p0_ub, fnr=p1_ub, delta=delta)
        except Exception as e:
            logging.info(str(e))
            eps_lb_gdp = 0

        try:
            eps_lb_cp = get_bounds_clopper(fpr=p0_ub, fnr=p1_ub, delta=delta)
        except Exception as e:
            logging.info(str(e))
            eps_lb_cp = 0

        return p0_ub, p1_ub, eps_lb_cp, eps_lb_gdp

    _, _, thresholds = roc_curve(gt, outputs)
    result = Parallel(n_jobs=8)(
        delayed(alpha_beta_single)(threshold) for threshold in thresholds
    )
    alphas_ub = np.array([x[0] for x in result])
    betas_ub = np.array([x[1] for x in result])
    epsilon_cp_lbs = np.array([x[2] for x in result])
    epsilon_gdp_lbs = np.array([x[3] for x in result])
    
    if np.all(epsilon_gdp_lbs == 0):
        logging.info("No valid data")
        return alphas_ub, betas_ub, epsilon_cp_lbs, epsilon_gdp_lbs

    mask_1 = np.logical_and(alphas_ub > delta, alphas_ub < 1 - delta)
    mask_2 = np.logical_and(betas_ub > delta, betas_ub < 1 - delta)
    mask = np.logical_and(mask_1, mask_2)
    alphas_ub = alphas_ub[mask]
    betas_ub = betas_ub[mask]
    epsilon_cp_lbs = epsilon_cp_lbs[mask]
    epsilon_gdp_lbs = epsilon_gdp_lbs[mask]
    return alphas_ub, betas_ub, epsilon_cp_lbs, epsilon_gdp_lbs


def get_empirical_mu_lower(
    fnr: float,
    fpr: float,
):
    return norm.ppf(1.0 - fnr) - norm.ppf(fpr)


def convert_logit_to_prob(logit: np.ndarray) -> np.ndarray:
  """Converts logits to probability vectors.

  Args:
    logit: n by c array where n is the number of samples and c is the number of
      classes.

  Returns:
    The probability vectors as n by c array
  """
  prob = logit - np.max(logit, axis=1, keepdims=True)
  prob = np.array(np.exp(prob), dtype=np.float64)
  prob = prob / np.sum(prob, axis=1, keepdims=True)
  return prob

def calculate_statistic(pred: np.ndarray,
                        labels: np.ndarray,
                        sample_weight: Optional[np.ndarray] = None,
                        is_logits: bool = True,
                        option: str = 'logit',
                        small_value: float = 1e-45):
  """Calculates the statistics of each sample.

  The statistics is:
    for option="conf with prob", p, the probability of the true class;
    for option="xe", the cross-entropy loss;
    for option="logit", log(p / (1 - p));
    for option="conf with logit", max(logits);
    for option="hinge", logit of the true class - max(logits of the other
    classes).

  Args:
    pred: the logits or probability vectors, depending on the value of is_logit.
      An array of size n by c where n is the number of samples and c is the
      number of classes
    labels: true labels of samples (integer valued)
    sample_weight: a vector of weights of shape (num_samples, ) that are
      assigned to individual samples. If not provided, then each sample is
      given unit weight. Only the LogisticRegressionAttacker and the
      RandomForestAttacker support sample weights.
    is_logits: whether pred is logits or probability vectors
    option: confidence using probability, xe loss, logit of confidence,
      confidence using logits, hinge loss
    small_value: a small value to avoid numerical issue

  Returns:
    the computed statistics as size n array
  """
  if option not in [
      'conf with prob', 'xe', 'logit', 'conf with logit', 'hinge'
  ]:
    raise ValueError(
        'option should be one of ["conf with prob", "xe", "logit", "conf with logit", "hinge"].'
    )
  if option in ['conf with logit', 'hinge']:
    if not is_logits:  # the input needs to be the logits
      raise ValueError('To compute statistics with option "conf with logit" '
                       'or "hinge", the input must be logits instead of '
                       'probability vectors.')
  elif is_logits:
    pred = convert_logit_to_prob(pred)

  n = labels.size  # number of samples
  if option in ['conf with prob', 'conf with logit']:
    return pred[range(n), labels]
  if option == 'xe':
    return log_loss(labels, pred, sample_weight=sample_weight)
  if option == 'logit':
    p_true = pred[range(n), labels]
    pred[range(n), labels] = 0
    p_other = pred.sum(axis=1)
    return np.log(p_true + small_value) - np.log(p_other + small_value)
  if option == 'hinge':
    l_true = pred[range(n), labels]
    pred[range(n), labels] = -np.inf
    return l_true - pred.max(axis=1)
  raise ValueError
