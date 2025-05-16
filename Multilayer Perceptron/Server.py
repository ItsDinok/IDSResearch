import flwr as fl
from typing import Dict
import sys

# Define global values for client count and training rounds
CLIENTCOUNT = 3
ROUNDS = 1

# Return current round
def fit_config(server_round: int) -> Dict:
    config = {
        "server_round" : server_round,
    }
    return config


# Aggregate metrics and calculate weighted averages
def metrics_aggregate(results) -> Dict:
    if not results:
        return {}

    else:
        total_samples = 0

        # Collecting metrics
        aggregatedMetrics = {
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F1_Score": 0,
            "AUCROC": 0
        }

        # Extracting values from results
        for samples, metrics in results:
            for key, value in metrics.items():
                if key not in aggregatedMetrics:
                    aggregatedMetrics[key] = 0
                else:
                    aggregatedMetrics[key] += (value * samples)

            total_samples += samples

        # Compute weighted average for each metric
        for key in aggregatedMetrics.keys():
            aggregatedMetrics[key] = round(aggregatedMetrics[key] / total_samples, 6)

        return aggregatedMetrics


if __name__ == "__main__":
    print("Server:\n")
    if len(sys.argv) > 1:
        CLIENTCOUNT = int(sys.argv[1])

    # Build the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = CLIENTCOUNT,
        min_evaluate_clients = CLIENTCOUNT,
        min_available_clients = CLIENTCOUNT,
        on_fit_config_fn = fit_config,
        evaluate_metrics_aggregation_fn = metrics_aggregate,
        fit_metrics_aggregation_fn = metrics_aggregate
    )

    # Generate text file for server log
    fl.common.logger.configure(identifier='FL_Test', filename='log.txt')

    # Start the server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        server_address = "127.0.0.1:8080"
    )
