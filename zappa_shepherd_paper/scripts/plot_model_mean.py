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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def test_mean_significance(sample_data):
    """Test if the mean of the sample_data is significantly different from zero.
    Args:
        sample_data (np.ndarray): Array of sample data.
    Returns:
        float: p-value from the t-test.
    """
    # Calculate the sample mean and standard deviation
    mean = np.mean(sample_data)
    std_dev = np.std(sample_data, ddof=1)  # Sample standard deviation
    n = len(sample_data)
    
    # Perform a one-sample t-test
    t_statistic, p_value = stats.ttest_1samp(sample_data, 0)
    
    return p_value

def clim_change(target,period1,period2,box='null'):
    """
    Calculate the difference in climatological mean between two periods for a specific geographical box.

    This function extracts data from a specific geographical region (defined by `box`), calculates the average 
    over two distinct time periods (`period1` and `period2`), and returns the difference between the two means. 
    It effectively computes how the climate has changed between two periods in a given region.

    Parameters:
    ----------
    target : xarray.DataArray
        The input data array containing the variable of interest (e.g., temperature, precipitation, etc.)
        with dimensions that include 'time', 'lat' (latitude), and 'lon' (longitude).

    box : list or tuple of four floats
        A list or tuple defining the geographical box [lon_min, lon_max, lat_min, lat_max]. 
        This selects the area of interest based on longitude and latitude ranges.
    
    period1 : list or tuple of two elements (strings or datetime-like objects)
        Defines the first time period to be averaged, in the form [start_time, end_time].
        The format should match the time format in the `target` data (e.g., 'YYYY-MM-DD' or datetime objects).
        
    period2 : list or tuple of two elements (strings or datetime-like objects)
        Defines the second time period to be averaged, in the form [start_time, end_time].
        The format should match the time format in the `target` data (e.g., 'YYYY-MM-DD' or datetime objects).

    Returns:
    -------
    xarray.DataArray
        A DataArray representing the difference between the climatological means of the two periods for the 
        selected geographical box. The returned data will have the same spatial dimensions (lat, lon) as the 
        input data but reduced in time (since time is averaged over).
    """
    if box == 'null':
        output = target.sel(time=slice(period2[0],period2[1])).mean(dim='time') - target.sel(time=slice(period1[0],period1[1])).mean(dim='time')
    else:
        output = target.sel(lon=slice(box[0],box[1])).sel(lat=slice(box[2],box[3]))
    return output

def plot_function(target_change, p_values, sig_level=0.05):
    """
    Plot target changes and add stippling where p-values indicate significance.

    Parameters:
    ----------
    target_change : xarray.DataArray
        The data array to plot (e.g., difference between two climatological periods).
    p_values : xarray.DataArray
        Data array with p-values, aligned with the dimensions of `target_change`.
    sig_level : float, optional
        Significance level for stippling. Defaults to 0.05 (5%).

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The resulting plot figure with stippling for significance.
    """
    fig, ax = plt.subplots()

    # Plot the main target change
    target_change.plot(ax=ax)

    return fig

def main(config):
    """Run the diagnostic.""" 
    cfg=get_cfg(os.path.join(config["run_dir"],"settings.yml"))
    print(cfg)
    meta_dataset = group_metadata(config["input_data"].values(), "dataset") # meta_datasets va a guardar la lista de datasets (MODELO)
    meta = group_metadata(config["input_data"].values(), "alias") # meta va a guardar la lista de alias (cada alias es MODELO_r#i#p#f#)
    os.chdir(config["work_dir"])
    os.getcwd()
    os.makedirs("target_change",exist_ok=True) # genero los directorios donde voy a guardar resultados de este diagnostico 
    os.chdir(config["work_dir"]+'/'+"target_change")
    os.chdir(config["plot_dir"]) # genero los directorios donde voy las figuras de este diagnostico 
    os.getcwd()
    os.makedirs("target_change",exist_ok=True)
    ensemble_changes = [] # genero una lista para concatenar los cambios en precipitacion
    #print(f"\n\n\n{meta}")
    for dataset, dataset_list in meta_dataset.items(): ####DATASET es el modelo
        model_changes = [] # genero una lista para concatenar los cambios en precipitacion
        print(f"Evaluate for {dataset}\n")
        for alias, alias_list in meta.items(): ###ALIAS son los miembros del ensemble para el modelo DATASET
            #print(f"Computing index regression for {alias}\n")
            # las listas abajo solo abren los datos que corresponden al modelo "DATASET", lo hago en una lista simplement porque asi puedo poner el condicional en una linea
            target_pr = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]] for m in alias_list if (m["dataset"] == dataset) & (m["variable_group"] == 'pr')} 
            target_gw = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]] for m in alias_list if (m["dataset"] == dataset) & (m["variable_group"] == 'gw')} 
            try: 
                target_pr = target_pr['pr']*86400 #change units from kg m-2 s-1 to mm day-1
                target_gw = target_gw['gw']
                print(f"Computing climatological change in model {alias}\n")
                # calculo al cambio en precipiation entre dos periodos y recorto el area que quiero graficar 
                target_pr_change = clim_change(target_pr,['1950','1979'],['2070','2099'],box=[2,293.75,-38.75,-26.25])
                target_gw_change = clim_change(target_gw,['1950','1979'],['2070','2099']) 
                # guardo en la lista
                model_changes.append(target_pr.mean(dim='time') /target_gw.mean(dim='time') )
            except KeyError:
                continue
            
        # armo un xarray con todos los miembros del modelo dataset
        model_changes_ar = xr.concat(model_changes,dim='ensemble')
        # calculo la media del ensemble
        model_mean = model_changes_ar.mean(dim='ensemble')
        # calculo la significancia estadistica
        model_maps_pval = xr.apply_ufunc(test_mean_significance,model_changes_ar,input_core_dims=[["ensemble"]],output_core_dims=[[]],vectorize=True,dask="parallelized")
        # guardo la media de este modelo en la lista que contiene todas las medias de cada modelo
        ensemble_changes.append(model_mean)
        # guardo una figura por modelo
        fig = plot_function(model_mean, model_maps_pval)
        fig.savefig(config["plot_dir"]+"/target_change/"+dataset+"_precip_change.png")

    # hago lo mismo que hice con el ensemble por modelo pero para el ensemble de modelos
    ensemble_changes_ar = xr.concat(ensemble_changes,dim='model')
    ensemble_mean = ensemble_changes_ar.mean(dim='model')
    # calculo la significancia estadistica con un t-test
    ensemble_maps_pval = xr.apply_ufunc(test_mean_significance,ensemble_changes_ar,input_core_dims=[["model"]],output_core_dims=[[]],vectorize=True,dask="parallelized")
    # hago la figura
    fig = plot_function(ensemble_mean, ensemble_maps_pval)
    fig.savefig(config["plot_dir"]+"/target_change/ensemble_mean_precip_change.png")

if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                                    
