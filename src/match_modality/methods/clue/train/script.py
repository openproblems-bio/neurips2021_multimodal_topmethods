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
par = {
    'input_train_mod1': 'data/cite/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': 'data/cite/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_mod2.h5ad',
    'input_train_sol': 'data/cite/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_train_sol.h5ad',
    'output_pretrain': 'pretrain/cite'
}
## VIASH END

sys.path.append(meta['resources_dir'])
import utils

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


print('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_train_sol = ad.read_h5ad(par['input_train_sol'])


input_train_mod1.X = input_train_mod1.X.astype(np.float32)
input_train_mod2.X = input_train_mod2.X.astype(np.float32)
input_train_mod1.layers["counts"] = input_train_mod1.layers["counts"].astype(np.float32)
input_train_mod2.layers["counts"] = input_train_mod2.layers["counts"].astype(np.float32)


print("Unscrambling training cells...")
ord = input_train_sol.X.tocsr().indices
if "pairing_ix" in input_train_sol.uns:
    assert np.all(ord == np.argsort(input_train_sol.uns["pairing_ix"]))
input_train_mod2 = input_train_mod2[ord, :].copy()
input_train_mod2.obs_names = input_train_mod1.obs_names
input_train_mod1.obs["uid"] = [f"train-{i}" for i in range(input_train_mod1.shape[0])]
input_train_mod2.obs["uid"] = [f"train-{i}" for i in range(input_train_mod2.shape[0])]
assert np.all(input_train_mod1.obs["batch"] == input_train_mod2.obs["batch"])


gex = input_train_mod1
other = input_train_mod2


print('Preprocessing GEX...')
gex_prep = utils.GEXPreprocessing(n_comps=100, n_genes=n_genes, merge_adt=True)
gex_prep.fit_transform(gex)


print('Preprocessing ADT...')
other_prep = utils.ADTPreprocessing(n_comps=100)
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