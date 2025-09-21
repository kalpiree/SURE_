import numpy as np
from typing import Dict

def compute_covariate_shift(
    losses_all_models: Dict[int, Dict[int, float]],
    model_id: int,
    t: int,
    t_prime: int,
    epsilon: float = 1e-8
) -> float:
    max_shift = 0.0
    L_i_t = losses_all_models[model_id][t]
    L_i_tprime = losses_all_models[model_id][t_prime]

    for j, loss_trace in losses_all_models.items():
        if j == model_id:
            continue
        L_j_t = loss_trace[t]
        L_j_tprime = loss_trace[t_prime]

        num = abs(L_i_tprime - L_j_tprime) + epsilon
        denom = abs(L_i_t - L_j_t) + epsilon
        shift = abs(np.log(num / denom))
        max_shift = max(max_shift, shift)

    return max_shift

def compute_concept_shift(
    losses_all_models: Dict[int, Dict[int, float]],
    model_id: int,
    t: int,
    t_prime: int,
    epsilon: float = 1e-8
) -> float:
    L_i_t = losses_all_models[model_id][t]
    L_i_tprime = losses_all_models[model_id][t_prime]

    min_L_t = min(losses_all_models[j][t] for j in losses_all_models)
    min_L_tprime = min(losses_all_models[j][t_prime] for j in losses_all_models)

    num = L_i_t + L_i_tprime
    denom = min_L_t + min_L_tprime + epsilon

    return np.log((num + epsilon) / denom)

def compute_total_shift(
    losses_all_models: Dict[int, Dict[int, float]],
    model_id: int,
    t: int,
    t_prime: int
) -> float:
    cov = compute_covariate_shift(losses_all_models, model_id, t, t_prime)
    concept = compute_concept_shift(losses_all_models, model_id, t, t_prime)
    total = cov + concept

    print(f"    Shift between [{t} → {t_prime}] | Covariate: {cov:.4f} | Concept: {concept:.4f} | Total: {total:.4f}")
    return total

def select_segment_start(
    losses_all_models: Dict[int, Dict[int, float]],
    model_id: int,
    current_step: int,
    prev_segment_start: int,
    alpha: float,
    gamma: float
) -> int:
    candidates = list(range(prev_segment_start, current_step + 1))
    numerators = []

    print(f"\n[Segment Shift Selection] for Model {model_id} at Step {current_step}")
    print(f"    Candidate starts: {candidates}")

    for t in candidates:
        shift = compute_total_shift(losses_all_models, model_id, t, current_step)
        weight = np.exp(-alpha * shift) * ((current_step - t + 1) ** gamma)
        numerators.append(weight)

        print(f"    Segment [{t} → {current_step}] | Weight = {weight:.6f}")

    denom = sum(numerators)
    probs = [w / denom for w in numerators]
    best_idx = int(np.argmax(probs))
    best_start = candidates[best_idx]

    print(f"Selected Best Segment Start = {best_start} with Probability = {probs[best_idx]:.6f}\n")

    return best_start
