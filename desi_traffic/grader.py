def grade_episode(total_wait_penalty, max_possible_penalty=-5000):
    """
    Returns a score between 0.0 and 1.0. 
    1.0 means no waiting penalty (perfect), 0.0 means extremely bad congestion or ambulance stuck.
    """
    # Simply mapping negative penalties to 0-1 range
    score = 1.0 - (total_wait_penalty / max_possible_penalty)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
