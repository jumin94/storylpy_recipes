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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import matplotlib as mpl
import random

from climstory import spatial_MLR, stand


#Across models regression class
class spatial_MLR(object):
    def __init__(self):
        self.what_is_this = 'This performs a regression across models and plots everything'
    
    def regression_data(self,variable,regressors,regressor_names):
        """Define the regression target variable 
        this is here to be edited if some opperation is needed on the DataArray
        
        :param variable: DataArray
        :return: target variable for the regression  
        """
        self.target = variable
        regressor_indices = regressors
        self.regression_y = sm.add_constant(regressors.values)
        self.regressors = regressors.values
        self.rd_num = len(regressor_names)
        self.regressor_names = regressor_names

    #Regresion lineal
    def linear_regression(self,x):
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        returns = [res.params[i] for i in range(self.rd_num)]
        return tuple(returns)

    def linear_regression_pvalues(self,x):
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        returns = [res.pvalues[i] for i in range(self.rd_num)]
        return tuple(returns)
    
    def linear_regression_R2(self,x):
        y = self.regression_y
        res = sm.OLS(x,y).fit()
        return res.rsquared
    
    def linear_regression_relative_importance(self,x):
        y = self.regressors
        try:
            res = robjects.globalenv['rel_importance'](x,y)
            returns = [res[i] for i in range((len(res)))]
            #print(res)
            correct = True
        except:
            returns2 = [np.array([0.0]) for i in range(self.rd_num-1)]
            correct = False
        finally:
            if correct:
                #print('I should return ',returns)
                return tuple(returns)
            else:
                #print('I am returning 0')
                return tuple(returns2)


    def perform_regression(self,path,var): 
        """ Performs regression over all gridpoints in a map and returns and saves DataFrames
        
        :param path: saving path
        :return: none
        """
        
        target_var = xr.apply_ufunc(self.replace_nans_with_zero, self.target)
        results = xr.apply_ufunc(self.linear_regression,target_var,input_core_dims=[["model"]],
                                 output_core_dims=[[] for i in range(self.rd_num)],
                                 vectorize=True,
                                 dask="parallelized")
        results_pvalues = xr.apply_ufunc(self.linear_regression_pvalues,target_var,input_core_dims=[["model"]],
                                 output_core_dims=[[] for i in range(self.rd_num)],
                                 vectorize=True,
                                 dask="parallelized")
        results_R2 = xr.apply_ufunc(self.linear_regression_R2,target_var,input_core_dims=[["model"]],
                                 output_core_dims=[[]],
                                 vectorize=True,
                                 dask="parallelized")
        
        relative_importance = xr.apply_ufunc(self.linear_regression_relative_importance,target_var,input_core_dims=[["model"]],
                                 output_core_dims=[[] for i in range(self.rd_num-1)],
                                 vectorize=True,
                                 dask="parallelized")
      
        for i in range(self.rd_num):
            if i == 0:
                regression_coefs = results[0].to_dataset()
            else:
                regression_coefs[self.regressor_names[i]] = results[i]
                
        print('This is regressor_coefs:',regression_coefs)
        if var == 'ua':
            regression_coefs = regression_coefs.rename({'ua':self.regressor_names[0]})
        elif var == 'sst':
            regression_coefs = regression_coefs.rename({'tos':self.regressor_names[0]})
        elif var == 'tas':
            regression_coefs = regression_coefs.rename({'tas':self.regressor_names[0]})
        elif var == 'pr':
            regression_coefs = regression_coefs.rename({'pr':self.regressor_names[0]})
        else:
            regression_coefs = regression_coefs.rename({var:self.regressor_names[0]})
        regression_coefs.to_netcdf(path+'/'+var+'/regression_coefficients.nc')
        
        for i in range(self.rd_num):
            if i == 0:
                regression_coefs_pvalues = results_pvalues[0].to_dataset()
            else:
                regression_coefs_pvalues[self.regressor_names[i]] = results_pvalues[i]        
        if var == 'ua':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'ua':self.regressor_names[0]})
        elif var == 'sst':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'tos':self.regressor_names[0]})
        elif var == 'tas':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'tas':self.regressor_names[0]})
        elif var == 'pr':
            regression_coefs_pvalues = regression_coefs_pvalues.rename({'pr':self.regressor_names[0]})
        else:
            regression_coefs_pvalues = regression_coefs_pvalues.rename({var:self.regressor_names[0]})
        regression_coefs_pvalues.to_netcdf(path+'/'+var+'/regression_coefficients_pvalues.nc')
        
        for i in range(len(relative_importance)):
            if i == 0:
                relative_importance_values = relative_importance[0].to_dataset()
            else:
                relative_importance_values[self.regressor_names[1:][i]] = relative_importance[i]
                
        if var == 'ua':
            relative_importance_values = relative_importance_values.rename({'ua':self.regressor_names[1]})
        elif var == 'sst':
            relative_importance_values = relative_importance_values.rename({'tos':self.regressor_names[1]})
        elif var == 'tas':
            relative_importance_values = relative_importance_values.rename({'tas':self.regressor_names[1]})
        elif var == 'pr':
            relative_importance_values = relative_importance_values.rename({'pr':self.regressor_names[1]})
        else:
            relative_importance_values = relative_importance_values.rename({'ua':self.regressor_names[1]})
    
        relative_importance_values.to_netcdf(path+'/'+var+'/regression_coefficients_relative_importance.nc')
        results_R2.to_netcdf(path+'/'+var+'/R2.nc')
                     
        
    def create_x(self,i,j,dato):
        """ For each gridpoint creates an array and standardizes it 
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """    
        x = np.array([])
        for y in range(len(dato.time)):
            aux = dato.isel(time=y)
            x = np.append(x,aux[i-1,j-1].values)
        return stand(x)
    
    def replace_nans_with_zero(self,x):
        return np.where(np.isnan(x), random.random(), x)
          
def stand(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return anom





def main(config):
    """Run the diagnostic."""
    cfg=get_cfg(os.path.join(config["run_dir"],"settings.yml"))
    #print(cfg)
    meta = group_metadata(config["input_data"].values(), "dataset")
    rd_list = []
    target_wind_list = []; target_pr_list = [] 
    models = []
    count = 0
    for dataset, dataset_list in meta.items():
        #print(f"Computing index regression for {alias}\n")
        ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].mean(dim='time') for m in dataset_list if (m["variable_group"] != "u850") & 
                (m["variable_group"] != "tas") & (m["variable_group"] != "sst") & (m["variable_group"] != "psl") & (m["variable_group"] != "pr")}
        if 'gw' in ts_dict.keys():
            rd_list.append(ts_dict)
        else:
            print("There is no data for model "+dataset)
        target_pr = [xr.open_dataset(m["filename"])[m["short_name"]].mean(dim='time') for m in dataset_list if m["variable_group"] == "pr"]
        target_u850 = [xr.open_dataset(m["filename"])[m["short_name"]].mean(dim='time') for m in dataset_list if m["variable_group"] == "u850"]
        if len(target_pr) != 0:
            target_pr_list.append(target_pr[0]/ts_dict['gw'].values)
            target_wind_list.append(target_u850[0]/ts_dict['gw'].values)
            count +=1
            models.append(count)
        else:
            continue
    
    print('rd_list',rd_list)
    #Across model regrssion - create data array
    regressor_names = list(rd_list[0].keys())
    regressor_names.remove("gw")

    regressors = {}
    for rd in regressor_names:
        print(len(rd_list))
        list_values = [rd_list[m][rd].values/rd_list[m]['gw'].values for m,model in enumerate(rd_list)]
        regressors[rd] = np.array(list_values)
    
    #Create directories to store results
    os.chdir(config["work_dir"])
    os.getcwd()
    os.makedirs("regression_output",exist_ok=True)
    os.chdir(config["plot_dir"])
    os.getcwd()
    os.makedirs("regression_output",exist_ok=True)
    #Evalutate coefficients and make plots U850
    ua850_ens = xr.concat(target_wind_list,dim="model")
    var = 'ua'
    os.chdir(config["work_dir"]+'/regression_output')
    os.getcwd()
    os.makedirs(var,exist_ok=True)
    MLR = spatial_MLR()
    MLR.regression_data(ua850_ens,pd.DataFrame(regressors).apply(stand,axis=0),pd.DataFrame(regressors).keys().insert(0,'MEM'))
    MLR.perform_regression(config["work_dir"]+'/regression_output',var)
    #PR
    pr_ens = xr.concat(target_pr_list,dim="model")  
    var = 'pr'
    os.chdir(config["work_dir"]+'/regression_output')
    os.getcwd()
    os.makedirs(var,exist_ok=True)
    MLR = spatial_MLR()
    MLR.regression_data(pr_ens,pd.DataFrame(regressors).apply(stand,axis=0),pd.DataFrame(regressors).keys().insert(0,'MEM'))
    MLR.perform_regression(config["work_dir"]+'/regression_output',var)                                  
                  
       
if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                              
