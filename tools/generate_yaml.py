import yaml
import random

DIR = "config"

def generate_yaml_data():
    max_steps = random.randint(500000, 3000000)
    learning_rate = random.uniform(0.0001, 0.001)
    epochs = random.randint(1, 10)
    batch_size = random.randint(1000, 10000)
    buffer_size = random.randint(10000, 100000)

    data = {
        "behaviors": {
            "SoccerTwos": {
                "trainer_type": "poca",
                "hyperparameters": {
                    "batch_size": batch_size,
                    "buffer_size": buffer_size,
                    "learning_rate": learning_rate,
                    "beta": 0.005,
                    "epsilon": 0.2,
                    "lambd": 0.95,
                    "num_epoch": epochs,
                    "learning_rate_schedule": "constant"
                },
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 512,
                    "num_layers": 2,
                    "vis_encode_type": "simple"
                },
                "reward_signals": {
                    "extrinsic": {
                        "gamma": 0.99,
                        "strength": 1.0
                    }
                },
                "keep_checkpoints": 5,
                "max_steps": max_steps,
                "time_horizon": 1000,
                "summary_freq": 10000,
                "self_play": {
                    "save_steps": 50000,
                    "team_change": 200000,
                    "swap_steps": 2000,
                    "window": 10,
                    "play_against_latest_model_ratio": 0.5,
                    "initial_elo": 1200.0
                }
            }
        }
    }
    return data

def create_yaml_file():
    print("Creating yaml file...")
    data = generate_yaml_data()
    path = f"{DIR}/config.yaml"
    with open(path,"w",encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )
    print("Successfully created yaml file")

def main():
    create_yaml_file()

if __name__ == "__main__":
    main()
