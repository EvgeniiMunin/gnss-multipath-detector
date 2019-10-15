import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import json
import glob
import ast

#%% read all json files and write results to df

def json_to_df(path, ref=False):
    allFiles = glob.glob(path)
    
    df = pd.DataFrame()
    for i, file_ in enumerate(allFiles):
        print(file_)
        with open(file_, 'r') as json_log_file:
            # read all lines into list of strings / convert each liast item to a dict
            content = [ast.literal_eval(line) for line in json_log_file.readlines()[:-1]]
            # convert list of dicts in df 
            samples = []
            for sample in content:
                config = sample['config']
                
                if ref:
                    dict_sample = {'test_iter': sample['test_iter'], 
                           'val_acc': sample['val_acc'],
                           'val_precision': sample['val_precision'],
                           'val_recall': sample['val_recall']}
                    dict_sample.update(config)
                else:
                    history_last_epoch = {k:v[-1] for k,v in sample['history'].items()}
                    dict_sample = {'test_iter': sample['test_iter']}
                    dict_sample.update(history_last_epoch)
                
                dict_sample.update(config)
                samples.append(dict_sample)
            
            df_temp = pd.DataFrame.from_dict(samples)
            cn0_pos_str = file_.find('cn0-')
            df_temp['cn0'] =  int(file_[cn0_pos_str+4:cn0_pos_str+6])
            
        if i == 0:
            df = df_temp.copy(deep=True)
        else:
            df = pd.concat([df, df_temp], axis=0)
            
    df = df.reset_index(drop=True)
    if ref:
        df['model'] = 'ref'
    else:
        df['model'] = 'work'
    
    return df

#%% work/ ref models comparison by unit tests
df_work = json_to_df("logs10_work/*.json")
df_ref = json_to_df("logs_ref/*.json", ref=True)
df = pd.concat([df_ref, df_work[df_ref.columns]], axis=0)
ax = sns.boxplot(x='cn0', y='val_acc', hue='model', data=df)

# Boxplot
for config_var in ['delta_phase', 'cn0']:
    for metrics in ['val_acc', 'val_precision', 'val_recall']:
        plt.figure()
        sns_plot = sns.boxplot(x=config_var, y=metrics, hue='model', data=df)
        plt.savefig("result_imgs/boxplot_var-{}_metrics-{}.png".format(config_var, metrics))

        

#%% work/ ref models comparison by unit tests
df_work_combin = json_to_df("logs10_combin_work/*.json")
df_ref = json_to_df("logs_ref/*.json", ref=True)
df = pd.concat([df_ref, df_work_combin[df_ref.columns]], axis=0)
ax = sns.boxplot(x='cn0', y='val_acc', hue='model', data=df)

# Boxplot
for config_var in ['delta_phase', 'cn0']:
    for metrics in ['val_acc', 'val_precision', 'val_recall']:
        plt.figure()
        sns_plot = sns.boxplot(x=config_var, y=metrics, hue='model', data=df)
        plt.savefig("result_imgs/boxplot_var-{}_metrics-{}_combin.png".format(config_var, metrics))

        

#%% work models comparison by discretization level
dfs = []
folders = glob.glob("logs*_work")[:-1]
discrs = []
for folder in folders:
    df_temp = json_to_df(folder + "/*.json")
    
    if folder[5] == '0':
        discr_level = int(folder[4:6])
    else:
        discr_level = int(folder[4:5])
    
    df_temp['discr'] = int(folder[4:5])
    dfs.append(df_temp)

df = pd.concat(dfs, axis=0)
ax = sns.boxplot(x='cn0', y='val_acc', hue='discr', data=df)

# Boxplot
for config_var in ['delta_phase', 'cn0']:
    for metrics in ['val_acc', 'val_precision', 'val_recall']:
        plt.figure()
        sns_plot = sns.boxplot(x=config_var, y=metrics, hue='discr', data=df)
        plt.savefig("result_imgs/boxplot_var-{}_metrics-{}_combin.png".format(config_var, metrics))

        

        
        