#I AM HUMAN
# I AM ROBOT
# I AM GAIA
import xarray as xr
import numpy as np
import statsmodels.api as sm
import pandas as pd
import json 
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
from sklearn import linear_model
import glob
from scipy import signal
import netCDF4

def stand(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return anom

def main(config):
    """Run the diagnostic."""
    meta = group_metadata(config["input_data"].values(), "dataset")
    rd_list = []
    models = []
    for dataset, dataset_list in meta.items():
        #print(f"Computing index regression for {alias}\n")
        ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].mean(dim='time') for m in dataset_list if (m["variable_group"] != "u850") & 
                (m["variable_group"] != "tas") & (m["variable_group"] != "sst") & (m["variable_group"] != "psl") & (m["variable_group"] != "pr")}
        if 'gw' in ts_dict.keys():
            rd_list.append(ts_dict)
            models.append(dataset)
        else:
            print("There is no data for model "+dataset)
    
    print('rd_list',rd_list)
    #Across model regrssion - create data array
    regressor_names = list(rd_list[0].keys())
    regressor_names.remove("gw")

    regressors_scaled = {}
    regressors = {}
    for rd in regressor_names:
        print(len(rd_list))
        list_values = [rd_list[m][rd].values/rd_list[m]['gw'].values for m,model in enumerate(rd_list)]
        regressors_scaled[rd] = np.array(list_values)
        list_values = [rd_list[m][rd].values for m,model in enumerate(rd_list)]
        regressors[rd] = np.array(list_values)
        
    
    #Create directories to store results
    os.chdir(config["work_dir"])
    os.getcwd()
    os.makedirs("remote_drivers",exist_ok=True)
    df_stand = pd.DataFrame(regressors_scaled, index = models).apply(stand,axis=0)
    df_raw = pd.DataFrame(regressors, index = models)
    df_scaled = pd.DataFrame(regressors_scaled, index = models)
    df_stand.to_csv(config["work_dir"]+'/remote_drivers'+'/scaled_standardized_drivers.csv')
    df_raw.to_csv(config["work_dir"]+'/remote_drivers'+'/drivers.csv')
    df_scaled.to_csv(config["work_dir"]+'/remote_drivers'+'/scaled_drivers.csv')

if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                              
