import flwr as fl
import numpy as np

def krum(weights, num_nearest=2):
    if len(weights) == 0:
        return None
    dists = np.zeros((len(weights), len(weights)))
    for i, w1 in enumerate(weights):
        for j, w2 in enumerate(weights):
            dists[i, j] = np.linalg.norm(w1 - w2)
    scores = np.sum(np.partition(dists, num_nearest, axis=1)[:, :num_nearest], axis=1)
    return weights[np.argmin(scores)]

def main():
    fl.server.start_server(
        server_address="localhost:8080",
        strategy=fl.server.strategy.FedAvg(aggregate_fn=krum)
    )

if __name__ == "__main__":
    main()
