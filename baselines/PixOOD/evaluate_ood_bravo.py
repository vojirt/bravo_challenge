import os
import argparse
import matplotlib.image as mpimg
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from pprint import pprint
from easydict import EasyDict as edict

from pixood import PixOOD
from support import get_datasets, OODEvaluator



def save_dict(d, name):
    """
    Save the records into args.out_path. 
    Print the records to console if verbose=True
    """
    if args.verbose:
        pprint(d)
    store_path = os.path.join(args.out_path, name)
    Path(store_path).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(store_path, f'results.pkl'), 'wb') as f:
        pickle.dump(d, f)

def current_result_exists(model_name):
    """
    Check if the current results exist in the args.out_path
    """
    store_path = os.path.join(args.out_path, model_name)
    return os.path.exists(os.path.join(store_path, f'results.pkl'))

def run_evaluations(args, model, dataset, model_name, dataset_name, out_path):
    """
    Run evaluations for a particular model over all designated datasets.
    """

    evaluator = OODEvaluator(model)
    loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

    viz_path = os.path.join(f"segmentations/{model_name}/") if args.verbose else None  
    anomaly_score = evaluator.compute_anomaly_scores(loader=loader, out_path=out_path, viz_path=viz_path)

    if args.verbose:
        vis_path = os.path.join(f"anomaly_scores/{model_name}/{dataset_name}")
        os.makedirs(vis_path, exist_ok=True)
        for i in tqdm(range(0, len(anomaly_score)), desc=f"storing anomaly scores at {vis_path}"):
            mpimg.imsave(os.path.join(vis_path, f"score_{i}.png"), anomaly_score[i].squeeze(), cmap='magma')

def main(args):
    results = edict()

    DATASETS = get_datasets(os.path.abspath(args.datasets_folder))

    dataset_group = [(name, dataset) for (name, dataset) in DATASETS.items() ]

    print("Datasets to be evaluated:")
    [print(g[0]) for g in dataset_group]
    print("-----------------------")

    pixood_methods = {
        "PixOOD": "./configs/pixood.yaml",           # grood_logml_1000K_01adamw_tau10_resetthr1
        "PixOOD_Dec": "./configs/pixood_dec.yaml",   # grood_dinov2deeplabdecoder_logml_1000K_01adamw_tau10_resetthr1
        "PixOOD_Dec_CityBDD": "./configs/pixood_dec_citybdd.yaml",   # grood_dinov2deeplabdecoder_citybdd_logml_1000K_01adamw_tau10_resetthr1
        "PixOOD_R101DLv3": "./configs/pixood_r101dlv3.yaml",   # grood_dinov2deeplab_logml_1000K_01adamw_tau10_resetthr1
    }

    code_dir = os.path.join("./", "pixood_src")
    model_name = args.method
    model = PixOOD(config = pixood_methods[model_name], code_dir = code_dir, 
                   eval_labels = [], eval_scale_factor = 7, norm_scale = 0.2, norm_thr = 0.95)

    tmp = f"Evaluating method: {model_name}"
    print("\033[104m" + "="*len(tmp) + "\033[0m")
    print(f"\033[104m{tmp}\033[0m")
    print("\033[104m" + "="*len(tmp) + "\033[0m")

    args.out_path = os.path.join(args.out_path, model_name)

    for dataset_name, dataset in dataset_group:
        print(f'Testing dataset {dataset_name}')
        if dataset_name not in results:
            results[dataset_name] = edict()
        out_path = args.out_path
        Path(out_path).mkdir(exist_ok=True, parents=True)
        results[dataset_name] = run_evaluations(args, model, dataset, model_name, dataset_name, out_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OOD Evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help="If segmentation and anomaly score images should be saved")
    parser.add_argument('--datasets_folder', type=str, default='../../datasets/',
                        help='the path to the folder that contains all datasets for evaluation')
    parser.add_argument('--out_path', type=str, default='./results',
                        help='output file for saving the results as a pickel file')
    parser.add_argument('--method', type=str, choices=['PixOOD', 'PixOOD_Dec', 'PixOOD_Dec_CityBDD', 'PixOOD_R101DLv3'], default='PixOOD',
                        help='Method to be evaluated')
    args = parser.parse_args()

    main(args)
