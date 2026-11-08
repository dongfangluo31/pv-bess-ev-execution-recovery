def linear_shortage(deficit0, deficit1):
    if min(deficit0, deficit1) > 0:
        return (0.5 * (deficit0 + deficit1), 1.0)
    if max(deficit0, deficit1) <= 0:
        return (0.0, 0.0)
    fraction = max(deficit0, deficit1) / abs(deficit1 - deficit0)
    return (0.5 * max(deficit0, deficit1) * fraction, fraction)
