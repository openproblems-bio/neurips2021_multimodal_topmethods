from os import path
import subprocess
import anndata as ad
from scipy.sparse import issparse

## VIASH START
# This code block will be replaced by viash at runtime.
meta = { 'functionality_name': 'foo' }
## VIASH END

method_id = meta['functionality_name']
command = "./" + method_id

# define some filenames
testpar = {
  "input_train_mod1": "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad",
  "input_train_mod2": "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad",
  "input_test_mod1": "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod1.h5ad",
  "input_test_mod2": "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod2.h5ad",
  "output": "output.h5ad"
}

print("> Running method")
out = subprocess.check_output([
  command,
  "--input_train_mod1", testpar['input_train_mod1'],
  "--input_train_mod2", testpar['input_train_mod2'],
  "--input_test_mod1", testpar['input_test_mod1'],
  "--output", testpar['output']
]).decode("utf-8")

print("> Checking whether output files were created")
assert path.exists(testpar['output'])

print("> Reading h5ad files")
ad_sol = ad.read_h5ad(testpar['input_test_mod2'])
ad_pred = ad.read_h5ad(testpar['output'])

print("> Checking dataset id")
assert ad_pred.uns['dataset_id'] == ad_sol.uns['dataset_id']

print("> Checking method id")
assert ad_pred.uns['method_id'] == method_id

print("> Checking X")
assert issparse(ad_pred.X)
assert ad_pred.n_obs == ad_sol.n_obs
assert ad_pred.n_vars == ad_sol.n_vars
assert all(ad_pred.obs_names == ad_sol.obs_names)
assert all(ad_pred.var_names == ad_sol.var_names)

print("> Test succeeded!")