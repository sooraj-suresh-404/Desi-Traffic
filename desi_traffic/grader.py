def grade_episode(total_wait_penalty, max_possible_penalty=-5000):
    """
    Returns a score strictly between 0.0 and 1.0.

    The validator rejects endpoint scores, so we keep a small margin away from
    both 0 and 1 while preserving the relative ranking of episodes.
    """
    # Map the penalty range to (0, 1) and keep a small epsilon margin.
    epsilon = 1e-6
    raw_score = 1.0 - (total_wait_penalty / max_possible_penalty)

    if raw_score <= 0.0:
        return epsilon
    if raw_score >= 1.0:
        return 1.0 - epsilon

    return max(epsilon, min(1.0 - epsilon, raw_score))
