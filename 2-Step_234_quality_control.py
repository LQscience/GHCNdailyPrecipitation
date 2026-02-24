import os
import math
import shutil
import numpy as np
import pandas as pd

import multiprocessing as mp
from multiprocessing import Pool, Manager

from pathlib import Path
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed

from QualityControl.checks.durre import Durre_qc_one_station
from QualityControl.checks.hamada import Hamada_qc_one_station
from QualityControl.checks.beck import Beck_qc_one_station

Merge_data_path = Path("/data6t/AIWP_TP_dataset/merge_data")
Merge_daily_all_data_path = Merge_data_path / "daily_100"

Durre_data_path = Path("/data6t/AIWP_TP_dataset/Durre2010_data")
Durre_daily_all_data_path = Durre_data_path / "daily_100"

Hamada_data_path = Path("/data6t/AIWP_TP_dataset/Hamada2011_data")
Hamada_daily_all_data_path = Hamada_data_path / "daily_100"

Beck_data_path = Path("/data6t/AIWP_TP_dataset/Beck2019_data")
Beck_daily_all_data_path = Beck_data_path / "daily_100"

station_info_df = pd.read_csv(Merge_data_path / "daily_100_station_info.csv")
station_info_df


def Durre_qc_multiprocessing(id_list, n_jobs=-1):
    """
    Perform quality control checks for all stations using multiprocessing.

    Parameters:
    -----------
    id_list : list
        List of station IDs
    n_jobs : int
        Number of parallel processes, -1 means using all available CPU cores
    """
    # Ensure output directory exists
    Durre_daily_all_data_path.mkdir(parents=True, exist_ok=True)
    
    # If n_jobs is -1, use all available CPU cores
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    print(f"Using {n_jobs} processes to handle {len(id_list)} stations...")
    
    # Prepare argument list
    args_list = [
        (station_id, Merge_daily_all_data_path, Durre_daily_all_data_path, station_info_df) 
        for station_id in id_list
    ]
    
    # Process using process pool
    with Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap_unordered(Durre_qc_one_station, args_list),
            total=len(id_list),
            desc="Processing progress"
        ))
    
    # Count results
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\nProcessing finished!")
    print(f"Success: {success_count}, Failed: {failed_count}")
    
    # Print failed stations
    if failed_count > 0:
        print("\nFailed stations:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['message']}")
    
    return results


def Hamada_qc_multiprocessing(id_list, n_jobs=-1):
    """
    Perform quality control checks for all stations using multiprocessing.

    Parameters:
    -----------
    id_list : list
        List of station IDs
    n_jobs : int
        Number of parallel processes, -1 means using all available CPU cores
    """
    # Ensure output directory exists
    Hamada_daily_all_data_path.mkdir(parents=True, exist_ok=True)
    
    # If n_jobs is -1, use all available CPU cores
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    print(f"Using {n_jobs} processes to handle {len(id_list)} stations...")
    
    # Prepare argument list
    args_list = [
        (station_id, Durre_daily_all_data_path, Hamada_daily_all_data_path, station_info_df) 
        for station_id in id_list
    ]
    
    # Process using process pool
    with Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap_unordered(Hamada_qc_one_station, args_list),
            total=len(id_list),
            desc="Processing progress"
        ))
    # Count results
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\nProcessing finished!")
    print(f"Success: {success_count}, Failed: {failed_count}")
    
    # Print failed stations
    if failed_count > 0:
        print("\nFailed stations:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['message']}")
    
    return results


def Beck_qc_multiprocessing(id_list, n_jobs=-1):
    """
    Perform quality control checks for all stations using multiprocessing.

    Parameters:
    -----------
    id_list : list
        List of station IDs
    n_jobs : int
        Number of parallel processes, -1 means using all available CPU cores
    """
    # Ensure output directory exists
    Beck_daily_all_data_path.mkdir(parents=True, exist_ok=True)
    
    # If n_jobs is -1, use all available CPU cores
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    print(f"Using {n_jobs} processes to handle {len(id_list)} stations...")
    
    # Prepare argument list
    args_list = [
        (station_id, Hamada_daily_all_data_path, Beck_daily_all_data_path) 
        for station_id in id_list
    ]
    
    # Process using process pool
    with Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap_unordered(Beck_qc_one_station, args_list),
            total=len(id_list),
            desc="Processing progress"
        ))

    # Count results
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\nProcessing finished!")
    print(f"Success: {success_count}, Failed: {failed_count}")
    
    # Print failed stations
    if failed_count > 0:
        print("\nFailed stations:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['message']}")
    
    return results


id_list = [f.name[:-4] for f in Merge_daily_all_data_path.iterdir() if f.is_file()]


results = Durre_qc_multiprocessing(id_list, n_jobs=-1)

results = Hamada_qc_multiprocessing(id_list, n_jobs=-1)

results = Beck_qc_multiprocessing(id_list, n_jobs=-1)
