import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# command to run python train.py --batch nnnn --buffer nnnn --steps nnnn --lr nnnn --epoch n
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1000000)
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--batch", type=int, default=2048)
parser.add_argument("--buffer", type=int, default=20480)

args = parser.parse_args()
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(script_dir, "..", "data", "SCT-data.csv")
FILE_PATH = os.path.normpath(FILE_PATH)

if not os.path.exists(FILE_PATH):
    print("No file")
    exit()

df = pd.read_csv(FILE_PATH)

if 'bath_size' in df.columns:
    df.rename(columns={'bath_size': 'batch_size'}, inplace=True)

df = df[df['total_steps'] >= 50000]

features = ['total_steps', 'learning_rate', 'epoch', 'batch_size', 'buffer_size']
X = df[features]

def train_predictor(target_name):
    if target_name not in df.columns: return None
    y = df[target_name]
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

model_entropy = train_predictor('mean_entropy')
model_length  = train_predictor('episode_length')

input_data = pd.DataFrame([{
    'total_steps': args.steps,
    'learning_rate': args.lr,
    'epoch': args.epoch,
    'batch_size': args.batch,
    'buffer_size': args.buffer
}])

print(f"Steps: {args.steps:,}")
print(f"LR: {args.lr}")
print(f"Batch size: {args.batch}")
print(f"Buffer size: {args.buffer}")

if model_entropy:
    pred_ent = model_entropy.predict(input_data)[0]
    print(f"\nMean entropy: {pred_ent:.3f}")
if model_length:
    pred_len = model_length.predict(input_data)[0]
    print(f"\nEpisode length: {pred_len:.1f}")