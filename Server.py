import flwr as fl
from typing import Dict

# Define global value for clients and training round
CLIENTCOUNT = 3
ROUNDS = 2

# Return current round
def FitConfig(serverRound) -> Dict:
    config = {
        "server_round": serverRound,
        }
    return config


# Aggregate metrics and calculate weighted average
def MetricsAggregate(results) -> Dict:
    if not results:
        return {}
    
    else:
        totalSamples = 0
        
        # Collecting Metrics
        aggregatedMetrics = {
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F1_Score": 0                                    
            }        

        # Extracting values from results
        for samples, metrics in results:
            for key, value in metrics.items:
                if key not in aggregatedMetrics:    
                    aggregatedMetrics[key] = 0
                else:
                    aggregatedMetrics[key] += (value * samples)

            totalSamples += samples

        # Compute weighted average for each metric
        for key in aggregatedMetrics.keys():
            aggregatedMetrics[key] = round(aggregatedMetrics[key] / totalSamples, 6)

        return aggregatedMetrics

if __name__ == "__main__":
    print(f"Server:\n")

    # Build a strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=CLIENTCOUNT,
        min_available_clients=CLIENTCOUNT,
        on_fit_config_fn=FitConfig,
        evaluate_metrics_aggregation_fn=MetricsAggregate,
        fit_metrics_aggregation_fn=MetricsAggregate                                                        
        )                                                                                                                                                                            

    # Generate text file for server log
    fl.common.logger.configure(identifier="FL_Test", filename="log.txt")

    # Start the server
    fl.server.start_server(
        config = fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy = strategy,
        server_address="LOCALHOST:8080"                
        )    
