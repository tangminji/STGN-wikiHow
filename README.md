# STGN Experiment on wikiHow

Sub-experiment of ["STGN: an Implicit Regularization Method for Learning with Noisy Labels in Natural Language Processing"](https://aclanthology.org/2022.emnlp-main.515/) (EMNLP 2022) by Tingting Wu, Xiao Ding, Minji Tang, Hao Zhang, Bing Qin, Ting Liu.

Main experiment: [github](https://github.com/tangminji/STGN-sst)

## Data: wikiHow
Data Source: ["Reasoning about Goals, Steps, and Temporal Ordering with WikiHow"](https://github.com/zharry29/wikihow-goal-step) (EMNLP2020) by Zhang

You should extract the wikihow data to `dataset/wikihow` according to the instruction of "https://github.com/zharry29/wikihow-goal-step/blob/master/wikihow_train_eval_code.ipynb".

Here's an example:
```
cd dataset
mkdir wikihow
gdown https://drive.google.com/uc?id=1BEhjc8geCzCREJl2VyTbg9W_TFDz-wVI
unzip wikihow_goal_step_data.zip -d ./wikihow
```

## Run code
Here is a example shell.
```
#!/bin/bash

#SBATCH -J {task}_{method}
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100-sxm2-16gb:2
#SBATCH -t 10:00:00
#SBATCH --mem 20240
#SBATCH -o output/{task}_{method}.out
#SBATCH -e output/{task}_{method}.err

source ~/.bashrc
conda activate discrimloss_torch1.4

model_type=roberta
task={task}
method={method}

for i in 0 1 2
do
    python tm_train_hy_nruns.py \\
    --model_type $model_type \\
    --exp_name run/$task/$method \\
    --seed $i \\
    --params_path params_{task}_{method}.json \\
    --out_tmp out_tmp_{task}_{method}.json
done
```

Options:
+ task: GOAL, STEP, ORDER
+ method: base, STGN

You should run the tm_train_hy_nruns.py with proper params written in json files.
