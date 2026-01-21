import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score

class SoccerTwosAnalyzer:
    def __init__(self):
        self.df = self.load_data()
        self.models = {}
        
    def load_data(self):
        path = "../data/SCT-data.csv"
        df = pd.read_csv(path)
        df = df[df['total_steps'] >= 100000].copy()
        df['batch_buffer_ratio'] = df['batch_size'] / df['buffer_size']
        df['training_intensity'] = df['total_steps'] * df['learning_rate']
        return df

    def train_predictors(self):
        print("\nPre-Training")
        features = ['total_steps', 'learning_rate', 'batch_size', 'buffer_size', 
                    'batch_buffer_ratio', 'training_intensity']
        X = self.df[features]
        
        self._train_rf(X, self.df['training_time'], 'Training Time', 'time_model')

        self._train_rf(X, self.df['mean_entropy'], 'Mean Entropy', 'entropy_model')

    def _train_rf(self, X, y, name, model_key):
        model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        
        print(f"Model for '{name}':")
        print(f"R2 Score: {r2_scores.mean():.4f}")

        model.fit(X, y)
        self.models[model_key] = model

    def analyze_correlations(self):
        print("\nBehavioral Analysis")
        
        self._check_linear_relation(
            x_col='mean_entropy', 
            y_col='episode_length', 
            name="Entropy vs Length",
            xlabel="Entropy ",
            ylabel="Episode Length "
        )

        self._check_linear_relation(
            x_col='episode_length', 
            y_col='ELO', 
            name="Length vs Elo",
            xlabel="Episode Length ",
            ylabel="Agent Elo"
            )

    def _check_linear_relation(self, x_col, y_col, name, xlabel, ylabel):

        X = self.df[[x_col]]
        y = self.df[y_col]
        
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        r2 = r2_score(y, preds)
        corr = self.df[x_col].corr(self.df[y_col])
        
        print(f"Hypothesis:'{name}':")
        print(f"Correlation:{corr:.4f}")
        print(f"R2 Score:{r2:.4f}")
        

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=self.df[x_col], y=self.df[y_col], alpha=0.6, label='Data')
        plt.plot(self.df[x_col], preds, color='red', linewidth=2, label=f'Trend (R²={r2:.2f})')
        plt.title(name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        filename = f"{x_col}_vs_{y_col}.png"
        # plt.savefig(filename)
        # plt.show() 
    def predict_scenario(self, steps, lr, batch, buffer):
        print(f"Steps={steps:,}, LR={lr}, Batch={batch}")
        
        input_data = pd.DataFrame([{
            'total_steps': steps, 
            'learning_rate': lr, 
            'batch_size': batch, 
            'buffer_size': buffer,
            'batch_buffer_ratio': batch / buffer,
            'training_intensity': steps * lr
        }])
        

        pred_time = self.models['time_model'].predict(input_data)[0]
        print(f"Learning time: ~{pred_time/60:.1f} min")
            
       
        pred_ent = self.models['entropy_model'].predict(input_data)[0]
        print(f"Predicted entropy: {pred_ent:.3f}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--buffer", type=int, default=20480)
    args = parser.parse_args()
    
    app = SoccerTwosAnalyzer()
    app.train_predictors()      
    app.analyze_correlations()  
    app.predict_scenario(args.steps, args.lr, args.batch, args.buffer) 