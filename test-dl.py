import lightning as L
import pandas as pd
import numpy as np
# from lightning.pytorch.loggers import CSVLogger

from configs.dl import dl_best_hparams
from configs.experiments import experiments_configs
from configs.ml import ml_best_hparams
from datasets.loader.datamodule import EhrDataModule
from datasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline, MlPipeline


project_name = "pyehr"

def run_ml_experiment(config):
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})
    
    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # # logger
    # checkpoint_filename = f'{config["model"]}-fold{config["fold"]}-seed{config["seed"]}'
    # logger = CSVLogger(save_dir="logs", name=f'test/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)
    # L.seed_everything(config["seed"]) # seed for reproducibility
    
    # train/val/test
    pipeline = MlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
    trainer.test(pipeline, dm)
    perf = pipeline.test_performance
    out = pipeline.test_outputs
    return perf, out

def run_dl_experiment(config):
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # # logger
    # checkpoint_filename = f'{config["model"]}-fold{config["fold"]}-seed{config["seed"]}'
    # if "time_aware" in config and config["time_aware"] == True:
    #     checkpoint_filename+="-ta" # time-aware loss applied
    # logger = CSVLogger(save_dir="logs", name=f'test/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)
    
    # checkpoint
    checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints/best.ckpt'
    if "time_aware" in config and config["time_aware"] == True:
        checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}-ta/checkpoints/best.ckpt' # time-aware loss applied
    # train/val/test
    pipeline = DlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
    trainer.test(pipeline, dm, ckpt_path=checkpoint_path)
    perf = pipeline.test_performance
    out = pipeline.test_outputs
    return perf, out

if __name__ == "__main__":
    best_hparams = dl_best_hparams # [TO-SPECIFY]
    performance_table = {'dataset':[], 'task': [], 'model': [], 'fold': [], 'seed': [], 'accuracy': [], 'auroc': [], 'auprc': [], 'es': [], 'mae': [], 'mse': [], 'rmse': [], 'r2': [], 'osmae': []}
    for i in range(0, len(best_hparams)):
        
        config = best_hparams[i]
        print(f"Testing... {i}/{len(best_hparams)}")
        if config["model"] not in ["MCGRU"]: # [TO-SPECIFY]
            # print(f"----{config["model"]}----skip")
            continue
        run_func = run_ml_experiment if config["model"] in ["LR", "RF", "DT", "GBDT", "XGBoost", "CatBoost"] else run_dl_experiment
        if config["dataset"]=="cdsl":
            # continue
            seeds = [0, 43, 2024]
            folds = [2,3]
        else: # tjh dataset
            continue
            seeds = [0, 43, 2024]
            folds = [0,1]
        if config["task"] not in ["outcome"]:
            continue
        for fold in folds:
            config["fold"] = fold
            for seed in seeds:
                config["seed"] = seed
                perf, out = run_func(config)
                # print(type(out))
                # print(out)
                y_pred = out['preds']
                y_true = out['labels']
                if not isinstance(y_pred, np.ndarray):
                    y_pred = y_pred.numpy()
                if not isinstance(y_true, np.ndarray):
                    y_true = y_true.numpy()
                df = pd.DataFrame({'predictions': y_pred})
                for i in range(y_true.shape[1]):
                    df[f'label_{i}'] = y_true[:, i]
                
                print(f"{config}, Test Performance: {perf}")

                if "time_aware" in config and config["time_aware"] == True:
                    model_name = config['model']+"-ta"
                else:
                    model_name = config['model']

                df.to_csv(rf'logs/test/{config['dataset']}/{config['model']}-{config['task']}-seed{seed}-fold{fold}-predictions.csv', index=False)

                performance_table['dataset'].append(config['dataset'])
                performance_table['task'].append(config['task'])
                performance_table['model'].append(model_name)
                performance_table['fold'].append(config['fold'])
                performance_table['seed'].append(config['seed'])
                if config['task'] == 'outcome':
                    performance_table['accuracy'].append(perf['accuracy'])
                    performance_table['auroc'].append(perf['auroc'])
                    performance_table['auprc'].append(perf['auprc'])
                    performance_table['es'].append(perf['es'])
                    performance_table['mae'].append(None)
                    performance_table['mse'].append(None)
                    performance_table['rmse'].append(None)
                    performance_table['r2'].append(None)
                    performance_table['osmae'].append(None)
                elif config['task'] == 'los':
                    performance_table['accuracy'].append(None)
                    performance_table['auroc'].append(None)
                    performance_table['auprc'].append(None)
                    performance_table['es'].append(None)
                    performance_table['mae'].append(perf['mae'])
                    performance_table['mse'].append(perf['mse'])
                    performance_table['rmse'].append(perf['rmse'])
                    performance_table['r2'].append(perf['r2'])
                    performance_table['osmae'].append(None)
                else:
                    performance_table['accuracy'].append(perf['accuracy'])
                    performance_table['auroc'].append(perf['auroc'])
                    performance_table['auprc'].append(perf['auprc'])
                    performance_table['es'].append(perf['es'])
                    performance_table['mae'].append(perf['mae'])
                    performance_table['mse'].append(perf['mse'])
                    performance_table['rmse'].append(perf['rmse'])
                    performance_table['r2'].append(perf['r2'])
                    performance_table['osmae'].append(perf['osmae'])
    pd.DataFrame(performance_table).to_csv(rf'logs/test/{config['dataset']}/perf_experiments.csv', index=False) # [TO-SPECIFY]
    