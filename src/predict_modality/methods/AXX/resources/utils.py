import anndata as ad
import logging
import numpy as np
import os
import time
import pandas as pd
import yaml
from pathlib import Path
from collections import namedtuple
from const import PATH, OUT_PATH
#logging.basicConfig(level=logging.INFO)

try:
    import git
except:
    pass

def get_tasks(phase):
    assert phase in ['phase1v2','phase2']
    tasks = [
        "GEX2ADT",
        "ADT2GEX",
        "GEX2ATAC",
        "ATAC2GEX"
    ]
    task2name = {
        "ADT2GEX":f"openproblems_bmmc_cite_{phase}_mod2",
        "GEX2ADT":f"openproblems_bmmc_cite_{phase}_rna",
        "ATAC2GEX":f"openproblems_bmmc_multiome_{phase}_mod2",
        "GEX2ATAC":f"openproblems_bmmc_multiome_{phase}_rna"
    }
    return tasks, task2name

def get_y_dim(data_path):
    if '_cite_' in data_path:
        if 'mod2' in data_path:
            return 13953,"ADT2GEX"
        elif 'rna' in data_path:
            return 134,"GEX2ADT"
        else:
            assert 0
    elif '_multiome_' in data_path:
        if 'mod2' in data_path:
            return 13431,"ATAC2GEX"
        elif 'rna' in data_path:
            return 10000,"GEX2ATAC"
        else:
            assert 0

def get_par(path,phase):
    par = {
      "input_solution" : f"{path}/datasets_{phase}/predict_modality",
      "input_prediction" : f"{path}/predictions/predict_modality",
    }
    return par

def get_train_test_paths(name,phase,path = "./output"):
    par = get_par(path,phase)
    train_mod1 = f"{par['input_solution']}/{name}/{name}.censor_dataset.output_train_mod1.h5ad"
    train_mod2 = train_mod1.replace('mod1','mod2')
    test_mod1 = train_mod1.replace('train','test')
    test_mod2 = test_mod1.replace('mod1','mod2')
    assert os.path.exists(train_mod1) and os.path.exists(train_mod2)
    if phase == 'phase1v2':
        assert os.path.exists(test_mod1) and os.path.exists(test_mod2)
    return train_mod1,train_mod2,test_mod1,test_mod2

def get_data_paths(task,phase,data_type='train_test',path='./output'):
    assert data_type in ['train_test','gt_pred']
    tasks, task2name = get_tasks(phase)
    name = task2name[task]
    if data_type == 'train_test':
        return get_train_test_paths(name,phase,path)
    else:
        return get_gt_pred_paths(name,path)

def get_gt_pred_paths(name,path = "./output"):
    par = get_par(path,'phase1v2')
    gt = f"{par['input_solution']}/{name}/{name}.censor_dataset.output_test_mod2.h5ad"
    pred = f"{par['input_prediction']}/{name}/{name}.method.output.h5ad"
    print(gt)
    print(pred)
    assert os.path.exists(gt) and os.path.exists(pred)
    return gt, pred

def eval_one_file(name):
    gt, pred = get_gt_pred_paths(name)
    
    logging.info("Reading solution file")
    ad_sol = ad.read_h5ad(gt)
    
    logging.info("Reading prediction file")
    ad_pred = ad.read_h5ad(pred)

    logging.info("Check prediction format")
    if ad_sol.uns["dataset_id"] != ad_pred.uns["dataset_id"]:
        raise ValueError("Prediction and solution have differing dataset_ids")

    if ad_sol.shape != ad_pred.shape:
        raise ValueError("Dataset and prediction anndata objects should have the same shape / dimensions.")
        
    logging.info("Computing MSE metrics")

    tmp = ad_sol.X - ad_pred.X
    rmse = np.sqrt(tmp.power(2).mean())
    mae = np.abs(tmp).mean()
    
    return rmse

def eval_all():
    start = time.time()
    tasks, task2name = get_tasks(phase='phase1v2')
    s = 0
    res = {}
    for task in tasks:
        name = task2name[task]
        score = eval_one_file(name)
        s += score
        res[task] = score
    res['overall'] = s/len(tasks)
    print_res(res)
    duration = time.time() - start
    logging.critical(f" Total time: {duration:.1f} seconds")
    
def print_res(res):
    for i,j in res.items():
        logging.critical(f" {i} {j:.4f}")
    

def check_column_mean_var_all(path='./output',phase='phase2'):
    tasks, task2name = get_tasks(phase=phase)
    if phase == 'phase2':
        names = ['train_mod1', 'train_mod2']
    else:
        names = ['train_mod1', 'train_mod2', 'test_mod1', 'test_mod2']
    logging.info("[min, max, mean]")
    res = []
    ms = []
    ns = []
    for task in tasks:
        data_names = get_data_paths(task,phase=phase,path=path)
        logging.info(f"task:{task}")
        for d,n in zip(data_names, names):
            logging.info(n)
            data = ad.read_h5ad(d)
            msg,dd = check_column_mean_var(data)
            logging.info('\n'+msg)
            res.append(dd)
            ms.append(task)
            ns.append(n)
    dg = pd.DataFrame({'task':ms,'type':ns})
    res = np.array(res)
    c1 = ['mu','var']
    c2 = ['min','max','mean']
    df = pd.DataFrame(res,columns = [f'{i}_{j}' for i in c1 for j in c2]+['rows','cols'])
    df = pd.concat([dg,df],axis=1)
    return df
    
        
def check_column_mean_var(data):
    
    x = data.X
    mu = x.mean(axis=0)  
    
    msg = f"mean {mu.min():.3f}, {mu.max():.3f}, {mu.mean():.3f}\n"
    u2 = (x.multiply(x)).mean(axis=0)
    var = u2 - np.multiply(mu,mu)
    msg += f"var {var.min():.3f}, {var.max():.3f}, {var.mean():.3f}\n"
    
    d = [mu.min(),mu.max(),mu.mean(),var.min(),var.max(),var.mean(),x.shape[0],x.shape[1]]
    return msg,np.array(d)

def to_site_donor(data):
    df = data.obs['batch'].copy().to_frame().reset_index()
    df.columns = ['index','batch']
    df['site'] = df['batch'].apply(lambda x: x[:2])
    df['donor'] = df['batch'].apply(lambda x: x[2:]) 
    return df

def get_batch_count_df(data):
    df = to_site_donor(data)    

    ds = df[['site','count']].groupby('site').agg({'count':'sum'})
    ds = ds.reset_index()
    
    dd = df[['donor','count']].groupby('donor').agg({'count':'sum'})
    dd = dd.reset_index()
    
    return df.drop(['site','donor'],axis=1), ds, dd

def get_batch_count_df_all(path,phase='phase2'):
    tasks, task2name = get_tasks(phase=phase)
    names = ['train_mod1', 'train_mod2', 'test_mod1', 'test_mod2']
    if phase == 'phase2':
        names = ['train_mod1']
    else:
        names = ['train_mod1', 'test_mod1']
    res = []
    for task in tasks:
        data_names = get_data_paths(task,phase=phase,path=path)
        data_names = [data_names[0],data_names[2]]
        logging.info(f"task:{task}")
        for d,n in zip(data_names, names):
            data = ad.read_h5ad(d)
            dfs = get_batch_count_df(data)
            for i in dfs:
                i['type'] = n
                i['task'] = task
            res.append(dfs)
    df = pd.concat([i[0] for i in res],axis=0).set_index(['batch','type','task']).unstack(-1).fillna(0)
    ds = pd.concat([i[1] for i in res],axis=0).set_index(['site','type','task']).unstack(-1).fillna(0)
    du = pd.concat([i[2] for i in res],axis=0).set_index(['donor','type','task']).unstack(-1).fillna(0)
    return df,ds,du

    
def get_count_df(df,col):
    ds = df[col].value_counts().to_frame().reset_index()
    ds.columns = [col,'count']
    return ds
    
    
def get_tr_te_data(task, phase, path='./output', fold = 0):
    # 3 fold cv where each site is a fold
    assert fold in [0,1,2]
    tr1,tr2,te1,te2 = get_data_paths(task,phase=phase,path=path)
    tr1 = ad.read_h5ad(tr1)
    tr2 = ad.read_h5ad(tr2)
    return split(tr1, tr2, fold)
    

def split(tr1, tr2, fold):
    df = to_site_donor(tr1) 
    mask = df['site'] == f's{fold+1}'
    maskr = ~mask

    Xt = tr1[mask].X.toarray()
    X = tr1[maskr].X.toarray()

    yt = tr2[mask].X.toarray()
    y = tr2[maskr].X.toarray()

    print(f"{X.shape}, {y.shape}, {Xt.shape}, {yt.shape}")

    return X,y,Xt,yt

def align(y,ms):
    ym_n = np.mean(y[~ms],axis=0,keepdims=True)
    ym_o = np.mean(y[ms],axis=0,keepdims=True)
    y[ms] = y[ms] - ym_o + ym_n
    return y

def write_score_log(commit,scores,log_path,notes):
    if not os.path.exists(log_path):
        with open(log_path,'w') as f:
            f.write("commit,fold0,fold1,fold2,overall,notes\n")
    score = ','.join(scores)
    with open(log_path,'a') as f:
        f.write(f"{commit},{score},{notes}\n")


def get_last_commit():
    repo = git.Repo('../../')
    master = repo.head.reference
    commit = master.commit.hexsha
    msg = master.commit.message
    return commit,msg

def load_yaml(path):
    with open(path) as f:
        x = yaml.safe_load(f)
    res = {}
    for i in x:
        res[i] = x[i]['value']
    config = namedtuple('Config', res.keys())(**res)
    print(config)
    return config

def generate_pseudo_test_data(task, fold):
    path = f'{PATH}/output/pseudo_test/fold_{fold}'
    tasks, task2name = get_tasks(phase='phase2')
    name = task2name[task]

    Path(f"{path}/{name}").mkdir(parents=True, exist_ok=True)
    name = f"{path}/{name}/{name}.censor_dataset.output_test_mod1.h5ad"
    if os.path.exists(name):
        assert os.path.exists(name.replace('mod1','mod2'))
        return
    
    tr1,tr2,te1,te2 = get_data_paths(task,phase='phase2',path=f'{PATH}/output')
    
    tr1 = ad.read_h5ad(tr1)
    tr2 = ad.read_h5ad(tr2)
    print(task, fold)
    print(tr1.shape, tr2.shape)
    
    df = to_site_donor(tr1) 
    tr1.obs['site'] = df['site'].values
    tr2.obs['site'] = df['site'].values
    
    mask = tr1.obs['site'] == f's{fold+1}'
    tr1 = tr1[mask]
    
    mask = tr2.obs['site'] == f's{fold+1}'
    tr2 = tr2[mask]
    
    print(tr1.shape, tr2.shape)
    
    tr1.write_h5ad(name)
    tr2.write_h5ad(name.replace('mod1','mod2'))
    
def generate_pseudo_test_data_all():
    tasks, task2name = get_tasks(phase='phase2')
    for task in tasks:
        for fold in range(3):
            generate_pseudo_test_data(task=task, fold=fold)

if __name__ == '__main__':
    #generate_pseudo_test_data_all()
    eval_all()
    #check_column_mean_var_all()
    #get_tr_te_data(task='GEX2ADT', path='./output', fold = 0)
