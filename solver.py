from typing import List
import numpy as np

DATATYPE = np.float64
N_ITER = 1000
tol = 1e-8


def pdpinv(A: np.array):
    # Find P, D such that A = P D P^-1
    w, P = np.linalg.eig(A)
    D = np.diag(w)
    return P, D

# Connectedness loss function
def connectedness_loss(_weights):
    n = _weights.shape[0]
    return np.linalg.norm(
        np.linalg.matrix_power(_weights, n)
        - np.eye(n))
def d_connectedness_loss(_weights):
    n = _weights.shape[0]
    P, D = pdpinv(_weights)
    Pinv = np.linalg.inv(P)
    return (
        np.linalg.matrix_power(D, n) @ Pinv
        + n * P @ np.linalg.matrix_power(D, n-1) @ Pinv
        - P @ np.linalg.matrix_power(D, n) @ Pinv @ Pinv
    )

# One-ness loss function
def oneness_loss(_weights):
    return np.linalg.norm(_weights.sum(axis=0) - 1.) + np.linalg.norm(_weights.sum(axis=1) - 1.)
def d_oneness_loss(_weights):
    n = _weights.shape[0]
    _sum0 = _weights.sum(axis=0)
    _sum1 = _weights.sum(axis=1)
    return 2*(np.array([_sum0]*n) + np.array([_sum1]*n).T - 2.)

def solve(distances: np.array) -> List[int]:

    n = distances.shape[0]

    # Initialize random weights
    weights = np.random.random(size=(n, n)).astype(DATATYPE)

    _distances_ravel = distances.ravel()

    # Distance loss function
    def distance_loss(_weights):
        return _distances_ravel.dot(_weights.ravel())

    def d_distance_loss(_):
        return distances

    def loss(weights, i):
        return (distance_loss(weights) * 20 / (i + 1)
                + connectedness_loss(weights)
                + oneness_loss(weights)
                )
    def d_loss(weights, i):
        d_dist = d_distance_loss(weights)
        try:
            d_conn = d_connectedness_loss(weights)
        except np.linalg.LinAlgError:
            d_conn = 0.
        d_onen = d_oneness_loss(weights)
        return (
            d_dist * 20. / (i + 1) if np.isrealobj(d_dist) else 0.
            + d_conn / 40. if np.isrealobj(d_conn) else 0.
            + d_onen / 3. if np.isrealobj(d_onen) else 0.
        )

    lr = .05

    print(weights)

    @np.vectorize
    def put_10(_x: float) -> float:
        if _x < 0.:
            return 0.
        if _x > 1.:
            return 1.
        return _x
        # return (.5 + np.tanh(_x - .5))

    weights = put_10(weights)

    for i in range(N_ITER):
        weights -= lr * d_loss(weights, i)
        new_loss = loss(weights, i)

        # print(f"{distance_loss(weights)=}")
        # print(d_distance_loss(weights))

        # print(f"{connectedness_loss(weights)=}")
        # print(d_connectedness_loss(weights))

        # print(f"{oneness_loss(weights)=}")
        # print(d_oneness_loss(weights))

        # if i % int(N_ITER / 100) == 1:
        #     print(f"Loss: {new_loss:.4e}, delta: {last_loss - new_loss:.4e} | weights: " + ' '.join(f"{w:.4e}" for w in weights.ravel()))
        
        last_loss = new_loss

        weights = put_10(weights)

    
    # print("D_LOSS")
    # print(d_loss(weights, i))

    return weights


if __name__ == "__main__":

    N = 3

    mat = np.random.random(size=(N, N)).astype(DATATYPE)

    print(mat)

    weights = solve(mat)

    print("MAT")
    print(mat)
    print("WEIGHTS")
    print(weights)
