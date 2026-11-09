"""BADDAT lookup functions; see PROVENANCE.md for snapshot details."""
from pathlib import Path
import numpy as np

_DATA = Path(__file__).resolve().parent / "data"
log_rho1_map = np.load(_DATA / "log_rho1_map.npy", allow_pickle=False)
log_rho1_std_map = np.load(_DATA / "log_rho1_std_map.npy", allow_pickle=False)
log_rho1_dev_map = np.load(_DATA / "log_rho1_dev_map.npy", allow_pickle=False)

def log_rho0_to_log_rho1(log_rho0, ncadence):
    ind_rho = np.clip(np.array((log_rho0 + 7 )//0.01, 'int'), 0, 1500)
    ind_ncadence = np.clip(np.array((np.log10(ncadence)-1)//0.15789474, 'int'), 0, 19)
    log_rho1_mean = log_rho1_map[ind_ncadence, ind_rho]
    log_rho1_std = log_rho1_std_map[ind_ncadence, ind_rho]
    return log_rho1_mean, log_rho1_std
    


def log_tau0_to_log_tau1(log_tau0, baseline, ncadence):
    log_rho0 = log_tau0 - np.log10(baseline)
    log_rho1_mean, log_rho1_std = log_rho0_to_log_rho1(log_rho0, ncadence)
    return log_rho1_mean + np.log10(baseline), log_rho1_std

def log_rho0_to_log_rho1_derivative(log_rho0, ncadence):
    ind_rho = np.clip(np.array((log_rho0 + 7 )//0.01, 'int'), 0, 1500)
    ind_ncadence = np.clip(np.array((np.log10(ncadence)-1)//0.15789474, 'int'), 0, 19)
    log_rho1_dev = log_rho1_dev_map[ind_ncadence, ind_rho]    
    return log_rho1_dev

def log_tau0_to_log_tau1_derivative(log_tau0, baseline, ncadence):
    log_rho0 = log_tau0 - np.log10(baseline)
    log_rho1_derivative = log_rho0_to_log_rho1_derivative(log_rho0, ncadence)
    return log_rho1_derivative
