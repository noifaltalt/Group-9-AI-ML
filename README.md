# Project Structure
```commandline
Root:.
|   AutoRunner.sh
|   Dockerfile
|   README.md
|   requirements.txt
+---config
|       config.yaml
|       SoccerTwos.yaml
+---data 
+---env             
|   +---Soccer Twos
|   |   |   UnityEnvironment.exe
|   |   |   UnityEnvironments.x86_64                   
+---model
|       evaluate_features_importance.ipynb
|       plot_data.ipynb
|       predict.py
|       prediction.ipynb
|       preprocess_data.ipynb
|       train.py            
+---tools
|       create_excel.py
|       generate_yaml.py    
\---training
    |   linux_train_model.py
    |   train_SCT.py
```

# How to start training?
## Setup environment
Setup virtual environment and install all required libs
```commandline
python -m venv .ml-agents-venv (I recommend to use version 3.10.1)
```
```commandline
.ml-agents-venv/Scripts/activate (for windows)
source .ml-agents-venv/bin/activate (for mac)
```
``` commandline
python -r install requirements.txt
```
If you want to use GPU instead of CPU (not for Appel M-series chip):
```commandline
pip install "torch==2.1.1+cu118" "torchvision==0.16.1+cu118" "torchaudio==2.1.1+cu118" -f https://download.pytorch.org/whl/torch_stable.html
```

## Start training
    
### Env: SoccerTwos
Open terminal and run this command
```commandline
python training/train_SCT.py --run-id [run id]
```
Argument
- --run-id: run id
- --command: "resume" or "force" (default: force)
- --base-port: set different port to run multi-processes (default: 5005)
