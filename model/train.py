import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# command to run python train.py --batch nnnn --buffer nnnn --steps nnnn --lr nnnn --epoch n
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1000000)
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--batch", type=int, default=2048)
parser.add_argument("--buffer", type=int, default=20480)

args = parser.parse_args()
FILE_PATH = "../data/SCT-data.csv"

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
    y = df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    print(f"MAE{mae:.4f}")
    print(f"MSE{mse:.4f}")
    print(f"RMSE{rmse:.4f}")
    print(f"R2{r2:.4f}")

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