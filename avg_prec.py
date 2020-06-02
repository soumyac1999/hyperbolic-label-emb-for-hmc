import numpy as np
from scipy import stats

def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """

#    print('y_score ', y_score)
    order = np.argsort(y_score)[::-1]
#    print(' order', order)
#    print('start y_true ', y_true)
    y_true = np.take(y_true, order[:k])
#    print("y_true ", y_true)

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    elif gains == "hops":
    	gains = 15-y_true
        # gains = np.log(1/y_true)
        # print(gains)
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    # discounts = np.arange(len(y_true)) + 2
    # print('np.sum(gains / discounts) ', np.sum(gains / discounts))
    # print(gains, discounts)
    return np.sum(gains / discounts)

def spearman_rank_correlation(y_true, y_score):

    x = stats.spearmanr(y_true, y_score)
    return x[0]


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

