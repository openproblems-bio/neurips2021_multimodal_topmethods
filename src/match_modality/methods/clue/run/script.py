import gc
import logging
import os
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy
import yaml

import scglue

logging.basicConfig(level=logging.INFO)

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.

# dataset_path = 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.'
# dataset_path = 'output/datasets_phase1v2/match_modality/openproblems_bmmc_multiome_phase1v2_rna/openproblems_bmmc_multiome_phase1v2_rna.censor_dataset.output_'
dataset_path = 'output/datasets_phase1v2/match_modality/openproblems_bmmc_cite_phase1v2_rna/openproblems_bmmc_cite_phase1v2_rna.censor_dataset.output_'

par = {
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'input_train_sol': f'{dataset_path}train_sol.h5ad',
    'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
    'input_test_mod2': f'{dataset_path}test_mod2.h5ad',
    'input_pretrain': 'path/to/model',
    'output': 'output.h5ad'
}
meta = {
    'resources_dir': '.',
    'functionality_name': 'clue'
}
## VIASH END


sys.path.append(meta['resources_dir'])
import utils

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_train_sol = ad.read_h5ad(par['input_train_sol'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
input_test_mod2 = ad.read_h5ad(par['input_test_mod2'])

input_train_mod1.X = input_train_mod1.X.astype(np.float32)
input_train_mod2.X = input_train_mod2.X.astype(np.float32)
input_test_mod1.X = input_test_mod1.X.astype(np.float32)
input_test_mod2.X = input_test_mod2.X.astype(np.float32)
input_train_mod1.layers["counts"] = input_train_mod1.layers["counts"].astype(np.float32)
input_train_mod2.layers["counts"] = input_train_mod2.layers["counts"].astype(np.float32)
input_test_mod1.layers["counts"] = input_test_mod1.layers["counts"].astype(np.float32)
input_test_mod2.layers["counts"] = input_test_mod2.layers["counts"].astype(np.float32)

dataset_id = {
    input_train_mod1.uns["dataset_id"],
    input_train_mod2.uns["dataset_id"],
    input_test_mod1.uns["dataset_id"],
    input_test_mod2.uns["dataset_id"]
}
assert len(dataset_id) == 1
dataset_id = dataset_id.pop()

logging.info("Unscrambling training cells...")
ord = input_train_sol.X.tocsr().indices
if "pairing_ix" in input_train_sol.uns:
    assert np.all(ord == np.argsort(input_train_sol.uns["pairing_ix"]))
input_train_mod2 = input_train_mod2[ord, :].copy()
input_train_mod2.obs_names = input_train_mod1.obs_names
input_train_mod1.obs["uid"] = [f"train-{i}" for i in range(input_train_mod1.shape[0])]
input_train_mod2.obs["uid"] = [f"train-{i}" for i in range(input_train_mod2.shape[0])]
input_test_mod1.obs["uid"] = [f"test-mod1-{i}" for i in range(input_test_mod1.shape[0])]
input_test_mod2.obs["uid"] = [f"test-mod2-{i}" for i in range(input_test_mod2.shape[0])]
assert np.all(input_train_mod1.obs["batch"] == input_train_mod2.obs["batch"])

mod1_feature_type = set(input_train_mod1.var["feature_types"])
mod2_feature_type = set(input_train_mod2.var["feature_types"])
assert len(mod1_feature_type) == len(mod2_feature_type) == 1
mod1_feature_type = mod1_feature_type.pop()
mod2_feature_type = mod2_feature_type.pop()

if {mod1_feature_type, mod2_feature_type} == {"GEX", "ATAC"}:
    omics = "multiome"
elif {mod1_feature_type, mod2_feature_type} == {"GEX", "ADT"}:
    omics = "cite"
else:
    raise RuntimeError("Unrecognized modality!")

logging.info('Concatenating training and test data...')
input_mod1 = ad.concat(
    {"train": input_train_mod1, "test": input_test_mod1},
    axis=0, join="outer", merge="same", label="group",
    fill_value=0, index_unique="-"
)
input_mod1.uns["feature_type"] = mod1_feature_type
del input_train_mod1, input_test_mod1
gc.collect()
input_mod2 = ad.concat(
    {"train": input_train_mod2, "test": input_test_mod2},
    axis=0, join="outer", merge="same", label="group",
    fill_value=0, index_unique="-"
)
input_mod2.uns["feature_type"] = mod2_feature_type
del input_train_mod2, input_test_mod2
gc.collect()

if mod1_feature_type == "GEX":
    gex, other = input_mod1, input_mod2
elif mod2_feature_type == "GEX":
    gex, other = input_mod2, input_mod1

logging.info('Reading preprocessors...')
with open(os.path.join(
        par['input_pretrain'], "prep.pickle"
), "rb") as f:
    prep = pickle.load(f)
    gex_prep = prep["gex_prep"]
    other_prep = prep["other_prep"]

logging.info('Preprocessing...')
if "starter" in dataset_id:
    gex_missing = set(gex_prep.params["features"]).difference(gex.var_names)
    gex = ad.concat([gex, ad.AnnData(
        X=scipy.sparse.csr_matrix((gex.n_obs, len(gex_missing)), dtype=gex.X.dtype),
        obs=gex.obs, var=pd.DataFrame(index=list(gex_missing)),
        layers={"counts": scipy.sparse.csr_matrix((gex.n_obs, len(gex_missing)), dtype=gex.layers["counts"].dtype)}
    )], axis=1, merge="same", uns_merge="first")
    other_missing = set(other_prep.params["features"]).difference(other.var_names)
    other = ad.concat([other, ad.AnnData(
        X=scipy.sparse.csr_matrix((other.n_obs, len(other_missing)), dtype=other.X.dtype),
        obs=other.obs, var=pd.DataFrame(index=list(other_missing)),
        layers={"counts": scipy.sparse.csr_matrix((other.n_obs, len(other_missing)), dtype=other.layers["counts"].dtype)}
    )], axis=1, merge="same", uns_merge="first")
    if input_mod1.uns["feature_type"] == "GEX":
        input_mod1, input_mod2 = gex, other
    else:  # input_mod2.uns["feature_type"] == "GEX":
        input_mod2, input_mod1 = gex, other
gex_prep.transform(gex)
other_prep.transform(other)

logging.info('Fine-tuning model...')
scglue.models.configure_dataset(
    gex, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="X_pca",
    use_batch="batch", use_uid="uid"
)
scglue.models.configure_dataset(
    other, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="X_lsi" if other.uns["feature_type"] == "ATAC" else "X_pca",
    use_batch="batch", use_uid="uid"
)

with open(os.path.join(
        par['input_pretrain'], "hyperparams.yaml"
), "r") as f:
    hyperparams = yaml.load(f, Loader=yaml.Loader)

logging.info('Building model...')
model = scglue.models.SCCLUEModel(
    {"gex": gex, "other": other},
    latent_dim=hyperparams["latent_dim"],
    x2u_h_depth=hyperparams["x2u_h_depth"],
    x2u_h_dim=hyperparams["x2u_h_dim"],
    u2x_h_depth=hyperparams["u2x_h_depth"],
    u2x_h_dim=hyperparams["u2x_h_dim"],
    du_h_depth=hyperparams["du_h_depth"],
    du_h_dim=hyperparams["du_h_dim"],
    dropout=hyperparams["dropout"],
    shared_batches=True,
    random_seed=hyperparams["random_seed"]
)
print(model.net)

logging.info('Adopting pretrained weights...')
model.adopt_pretrained_model(scglue.models.load_model(os.path.join(
    par['input_pretrain'], "pretrain.dill"
)))

logging.info('Compiling model...')
model.compile(
    lam_data=hyperparams["lam_data"],
    lam_kl=hyperparams["lam_kl"],
    lam_align=hyperparams["lam_align"],
    lam_cross=hyperparams["lam_cross"],
    lam_cos=hyperparams["lam_cos"],
    normalize_u=hyperparams["normalize_u"],
    domain_weight={"gex": 1, "other": 1},
    lr=1e-3  # TODO: Fine-tuning learning rate
)

logging.info('Training model...')
model.fit(
    {"gex": gex, "other": other},
    align_burnin=0, max_epochs=50 if "phase2" in dataset_id else 5,
    patience=8, reduce_lr_patience=3
)

logging.info('Projecting cell embeddings...')
gex.obsm["X_model"] = model.encode_data("gex", gex)
other.obsm["X_model"] = model.encode_data("other", other)

logging.info('Predicting pairing matrix...')
pairing_matrix = utils.split_matching(
    input_mod1[input_mod1.obs["group"] == "test"],
    input_mod2[input_mod2.obs["group"] == "test"],
    ("gex", "other") if input_mod1.uns["feature_type"] == "GEX" else ("other", "gex"),
    "batch", model, utils.snn_matching
)

logging.info('Writing prediction output...')
out = ad.AnnData(
    X=pairing_matrix,
    uns={
        "dataset_id": dataset_id,
        "method_id": "clue"
    }
)
out.write_h5ad(par['output'], compression = "gzip")
