import argparse
import io
import numpy as np
import random
import glob
import os
import re
import json
import logging
import pandas as pd
import dill as pickle

def valid_string(values):
    return f"Valid choices are: {list(values)}"


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _result_tag(results: dict) -> str:
    """Build a `dataset_dim={D}_rho={rho}` filename tag from a results dict.

    Including both `dim` and `rho` keeps every D x rho cell of the sweep in
    its own file rather than overwriting prior runs of the same dataset.
    """
    dataset_kwargs = results.get("dataset_kwargs", {})
    dim = dataset_kwargs.get("dim", "na")
    rho = dataset_kwargs.get("rho", "na")
    return f"{results['dataset']}_dim={dim}_rho={rho}"


def save_results(results: dict, results_dir: str):
    saveName = f"{results_dir}/{_result_tag(results)}.log"
    logging.info("Saving results in %s", saveName)
    if not os.path.exists(os.path.dirname(saveName)):
        os.makedirs(os.path.dirname(saveName))
    with open(saveName, "w") as f:
        f.write(json.dumps(results, indent=4))


def save_results_csv(results: dict, results_dir: str):
    saveName = f"{results_dir}/csv/{_result_tag(results)}.csv"
    logging.info("Saving results in %s", saveName)
    if not os.path.exists(os.path.dirname(saveName)):
        os.makedirs(os.path.dirname(saveName))
    dataset_kwargs = results.get("dataset_kwargs", {})
    dim = dataset_kwargs.get("dim", "na")
    rho = dataset_kwargs.get("rho", "na")
    with open(saveName, "w") as f:
        f.write(f"# dataset,{results['dataset']}\n")
        f.write(f"# dim,{dim}\n")
        f.write(f"# rho,{rho}\n")
        for model in results["models"]:
            f.write(str(model).upper() + "," + ",".join(results["model_perfs"][model]) + "\n")
            df = pd.read_json(io.StringIO(json.dumps(results["models"][model])))
            df = df.transpose()
            f.write(df.to_csv())
            f.write("\n\n")

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def save_experiment(experiment, checkpoint_dir: str, results: dict):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saveName = f"{checkpoint_dir}/{_result_tag(results)}.pkl"
    save_object(experiment, saveName)
