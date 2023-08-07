#! /usr/bin/env python
# Created: Fri Jun 30, 2023  02:08pm
# <Last modified>: Fri Jun 30, 2023  02:08pm
#
# (C) 2023  Bedartha Goswami <bedartha.goswami@uni-tuebingen.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License Version 3 for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------
import os
import xarray as xr
import numpy as np
from scipy import signal, fft

PATH = os.path.dirname(os.path.abspath(__file__))


def anomalies(da):
    """
    Remove the climatological means for each measurement (hour of year anomalies).

    Input
    -----
    dataarr : xarray.DataArray
              Input data array which has the dimension 'time' which in
              turn has the attribute 'dt'
    """
    def hour_of_year(time):
        return (time.dt.dayofyear - 1) * 24 + time.dt.hour

    grouped = da.groupby(hour_of_year(da.time))
    anom = da.copy(data=grouped - grouped.mean())
    return anom.rename(f"{da.name}_anom")


def filter(da, ftype="LF"):
    if ftype == "LF":
        fc = 1. / (250 * 4)                     # periods > 250d
        btype = "lowpass"
    elif ftype == "HF":
        fc = [1. / (250 * 4), 1. / (5 * 4)]     # 5d < periods < 250d
        btype = "bandpass"
    b, a = signal.butter(N=4, Wn=fc, btype=btype, analog=False)
    y = signal.filtfilt(b, a, da.to_numpy(), axis=0)
    y = da.copy(data=y)
    return y.rename(f"{da.name}_{ftype}")



if __name__ == "__main__":
    # load data
    print("load data ...")
    FN = PATH + "/../data/reanalysis/6-hourly/ERA5/era5_u10m_1940_2021_5N5S_130E80W_1deg.nc"
    ds = xr.open_dataset(FN)
    u10 = ds.u10

    # get anomalies
    print("anomalies ...")
    u10_anom = anomalies(u10)

    # filter anomalies to get LF and HF components
    print("filter anomalies ...")
    print("low pass ...")
    u10_anom_LF = filter(u10_anom, ftype="LF")
    print("band pass high frequencies ...")
    u10_anom_HF = filter(u10_anom, ftype="HF")

    # # sanity check plots
    # print("plot ...")
    # import matplotlib.pyplot as pl
    # u10_anom_LF.std(dim="time").plot.pcolormesh(figsize=[12, 3])
    # u10_anom_HF.std(dim="time").plot.pcolormesh(figsize=[12, 3])
    # pl.show()

    # save output
    print("save to disk ...")
    FO = FN.replace("u10m", u10_anom_LF.name)
    u10_anom_LF.to_netcdf(FO)
    print(f"saved to {FO}")
    FO = FN.replace("u10m", u10_anom_HF.name)
    u10_anom_HF.to_netcdf(FO)
    print(f"saved to {FO}")

