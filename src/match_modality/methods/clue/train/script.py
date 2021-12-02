import os
import pickle

import anndata as ad
import numpy as np
import yaml
import sys

import scglue

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
dataset_path = 'output/datasets/match_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_'
par = {
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'input_train_sol': f'{dataset_path}train_sol.h5ad',
    'output_pretrain': 'output/pretrain/clue/openproblems_bmmc_cite_phase2_mod2.clue_train.output_pretrain/'
}
meta = { 'resources_dir': 'src/match_modality/methods/clue/resources' }
## VIASH END

sys.path.append(meta['resources_dir'])
import utils



print('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_train_sol = ad.read_h5ad(par['input_train_sol'])

input_train_mod1.X = input_train_mod1.X.astype(np.float32)
input_train_mod2.X = input_train_mod2.X.astype(np.float32)
input_train_mod1.layers["counts"] = input_train_mod1.layers["counts"].astype(np.float32)
input_train_mod2.layers["counts"] = input_train_mod2.layers["counts"].astype(np.float32)

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

if omics == "cite":
    n_genes = 5000
    latent_dim = 20
    x2u_h_depth = 2
    x2u_h_dim = 512
    u2x_h_depth = 1
    u2x_h_dim = 128
    du_h_depth = 2
    du_h_dim = 128
    dropout = 0.2
    lam_data = 1.0
    lam_kl = 1.0
    lam_align = 2.0
    lam_cross = 2.0
    lam_cos = 1.0
    normalize_u = True
    random_seed = 5
elif omics == "multiome":
    n_genes = 10000
    latent_dim = 50
    x2u_h_depth = 2
    x2u_h_dim = 512
    u2x_h_depth = 1
    u2x_h_dim = 256
    du_h_depth = 1
    du_h_dim = 256
    dropout = 0.2
    lam_data = 1.0
    lam_kl = 0.3
    lam_align = 0.02
    lam_cross = 1.0
    lam_cos = 0.02
    normalize_u = True
    random_seed = 2

os.makedirs(par['output_pretrain'], exist_ok=True)
with open(os.path.join(par['output_pretrain'], "hyperparams.yaml"), "w") as f:
    yaml.dump({
        "n_genes": n_genes,
        "latent_dim": latent_dim,
        "x2u_h_depth": x2u_h_depth,
        "x2u_h_dim": x2u_h_dim,
        "u2x_h_depth": u2x_h_depth,
        "u2x_h_dim": u2x_h_dim,
        "du_h_depth": du_h_depth,
        "du_h_dim": du_h_dim,
        "dropout": dropout,
        "lam_data": lam_data,
        "lam_kl": lam_kl,
        "lam_align": lam_align,
        "lam_cross": lam_cross,
        "lam_cos": lam_cos,
        "normalize_u": normalize_u,
        "random_seed": random_seed
    }, f)


print("Unscrambling training cells...")
ord = input_train_sol.X.tocsr().indices
if "pairing_ix" in input_train_sol.uns:
    assert np.all(ord == np.argsort(input_train_sol.uns["pairing_ix"]))
input_train_mod2 = input_train_mod2[ord, :].copy()
input_train_mod2.obs_names = input_train_mod1.obs_names
input_train_mod1.obs["uid"] = [f"train-{i}" for i in range(input_train_mod1.shape[0])]
input_train_mod2.obs["uid"] = [f"train-{i}" for i in range(input_train_mod2.shape[0])]
assert np.all(input_train_mod1.obs["batch"] == input_train_mod2.obs["batch"])

if mod1_feature_type == "GEX":
    gex = input_train_mod1
    other = input_train_mod2
else:
    gex = input_train_mod2
    other = input_train_mod1

print('Preprocessing GEX...')
gex_prep = utils.GEXPreprocessing(n_comps=100, n_genes=n_genes, merge_adt=omics == "cite")
gex_prep.fit_transform(gex)


if omics == "cite":
    print('Preprocessing ADT...')
    other_prep = utils.ADTPreprocessing(n_comps=100)
elif omics == "multiome":
    print('Preprocessing ATAC...')
    other_prep = utils.ATACPreprocessing(n_comps=100)


other_prep.fit_transform(other)


with open(os.path.join(par['output_pretrain'], "prep.pickle"), "wb") as f:
    pickle.dump({
        "gex_prep": gex_prep,
        "other_prep": other_prep
    }, f)


scglue.models.configure_dataset(
    gex, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="X_pca",
    use_batch="batch", use_uid="uid"
)
scglue.models.configure_dataset(
    other, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="X_pca",
    use_batch="batch", use_uid="uid"
)


print('Building model...')
model = scglue.models.SCCLUEModel(
    {"gex": gex, "other": other},
    latent_dim=latent_dim,
    x2u_h_depth=x2u_h_depth,
    x2u_h_dim=x2u_h_dim,
    u2x_h_depth=u2x_h_depth,
    u2x_h_dim=u2x_h_dim,
    du_h_depth=du_h_depth,
    du_h_dim=du_h_dim,
    dropout=dropout,
    shared_batches=True,
    random_seed=random_seed
)


print('Compiling model...')
model.compile(
    lam_data=lam_data, lam_kl=lam_kl, lam_align=lam_align,
    lam_cross=lam_cross, lam_cos=lam_cos, normalize_u=normalize_u,
    domain_weight={"gex": 1, "other": 1}
)


print('Training model...')
model.fit(
    {"gex": gex, "other": other}
)
model.save(os.path.join(par['output_pretrain'], "pretrain.dill"))