"""Layout for the Any dataset page's overview: every episode checked, at a glance."""


def columns(episodes, max_cols: int = 300) -> list[list]:
    """Episode indices, sorted, split into at most `max_cols` runs of consecutive episodes of near-equal size."""
    eps = sorted(episodes)
    k = min(len(eps), max_cols)
    if k == 0:
        return []
    q, r = divmod(len(eps), k)
    out, i = [], 0
    for c in range(k):
        n = q + (c < r)
        out.append(eps[i:i + n])
        i += n
    return out
