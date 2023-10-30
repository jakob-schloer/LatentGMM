"""Util functions for analyzing ENSO types."""
import os
import cftime
import numpy as np
import pandas as pd
import xarray as xr
from scipy import linalg
import scipy.stats as stats
from sklearn.decomposition import PCA
import scipy.spatial.distance as dist
from tqdm import tqdm

from latgmm.utils import preproc, utdata, utstats
from latgmm.utils.eof import SpatioTemporalPCA
from latgmm.utils.dataset import SpatialData

PATH = os.path.dirname(os.path.abspath(__file__))

def get_nino_indices(ssta, time_range=None, monthly=False, antimeridian=False):
    """Returns the time series of the Nino 1+2, 3, 3.4, 4.

    Args:
        ssta (xr.dataarray): Sea surface temperature anomalies.
        time_range (list, optional): Select only a certain time range.
        monthly (boolean): Averages time dimensions to monthly. 
                            Default to True.

    Returns:
        [type]: [description]
    """
    da = ssta.copy()
    lon_range = [-90, -80] if antimeridian is False else preproc.get_antimeridian_coord([-90, -80])
    nino12, nino12_std = preproc.get_mean_time_series(
        da, lon_range=lon_range,
        lat_range=[-10, 0], time_roll=0
    )
    nino12.name = 'nino12'
    lon_range = [-150, -90] if antimeridian is False else preproc.get_antimeridian_coord([-150, -90])
    nino3, nino3_std = preproc.get_mean_time_series(
        da, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino3.name = 'nino3'
    lon_range = [-170, -120] if antimeridian is False else preproc.get_antimeridian_coord([-170, -120])
    nino34, nino34_std = preproc.get_mean_time_series(
        da, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino34.name = 'nino34'
    lon_range = [160, -150] if antimeridian is False else preproc.get_antimeridian_coord([160, -150])
    nino4, nino4_std = preproc.get_mean_time_series(
        da, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino4.name = 'nino4'

    nino_idx = xr.merge([nino12, nino3, nino34, nino4])

    if monthly:
        nino_idx = nino_idx.resample(time='M', label='left' ).mean()
        nino_idx = nino_idx.assign_coords(
            dict(time=nino_idx['time'].data + np.timedelta64(1, 'D'))
        )

    # Cut time period
    if time_range is not None:
        nino_idx = nino_idx.sel(time=slice(
            np.datetime64(time_range[0], "M"), 
            np.datetime64(time_range[1], "M")
        ))
    return nino_idx


def get_nino_indices_NOAA(
        fname="https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
        time_range=None, time_roll=0, group='month'):
    """Compute Nino indices from NOAA file.

    Args:
        fname ([type]): NOAA Nino calculation. Default to link.
        time_range ([type], optional): [description]. Defaults to None.
        time_roll (int, optional): [description]. Defaults to 3.
        base_period (list, optional): Base period to compute climatology.
            If None whole range is used. Default None.

    Returns:
        [type]: [description]
    """
    df = pd.read_csv(
        fname, skiprows=0, header=0, delim_whitespace=True
    )
    time = []
    for i, row in df.iterrows():
        time.append(np.datetime64(
            '{}-{:02d}'.format(int(row['YR']), int(row['MON'])), 'D'))

    nino_regions = xr.merge([
        xr.DataArray(data=df['NINO1+2'], name='nino12', coords={
                     "time": np.array(time)}, dims=["time"]),
        xr.DataArray(data=df['NINO3'], name='nino3', coords={
                     "time": np.array(time)}, dims=["time"]),
        xr.DataArray(data=df['NINO4'], name='nino4', coords={
                     "time": np.array(time)}, dims=["time"]),
        xr.DataArray(data=df['NINO3.4'], name='nino34', coords={
                     "time": np.array(time)}, dims=["time"]),
    ])

    # Choose 30 year climatology every 5 years

    min_date = np.array(nino_regions.time.data.min(), dtype='datetime64[M]')
    max_date = np.array('2020-12', dtype='datetime64[M]')
    time_steps = np.arange(min_date, max_date,
                           np.timedelta64(60, 'M'))
    nino_anomalies = []
    for ts in time_steps:
        if ts < (min_date + np.timedelta64(15*12, 'M')):
            base_period = np.array(
                [min_date, min_date + np.timedelta64(30*12-1, 'M')])
        elif ts > (max_date - np.timedelta64(15*12, 'M')):
            base_period = np.array(
                [max_date - np.timedelta64(30*12-1, 'M'), max_date])
        else:
            base_period = np.array([
                ts - np.timedelta64(15*12, 'M'),
                ts + np.timedelta64(15*12-1, 'M')
            ])

        buff = tut.compute_anomalies(
            nino_regions, group=group, base_period=base_period,
            verbose=False)
        buff = buff.sel(time=slice(ts, ts+np.timedelta64(5*12-1, 'M')))
        nino_anomalies.append(buff)

    nino_indices = xr.concat(nino_anomalies, dim='time')

    if time_roll > 0:
        nino_indices = nino_indices.rolling(
            time=time_roll, center=True).mean(skipna=True)
    if time_range is not None:
        nino_indices = nino_indices.sel(time=slice(np.datetime64(time_range[0], "M"),
                                                   np.datetime64(time_range[1], "M")))

    return nino_indices

def get_oni_index_noaa(fname, time_range=None):
    """Get ONI index from oni.acii.txt file."""
    df = pd.read_csv(
        fname, skiprows=1, header=0, delim_whitespace=True
    )
    # create time
    df['MON'] = df.index % 12 + 1
    time = []
    for i, row in df.iterrows():
        time.append(np.datetime64(
            '{}-{:02d}'.format(int(row['YR']), int(row['MON'])), 'M'))

    oni = xr.DataArray(data=df['ANOM'], name='oni',
                       coords={"time": np.array(time)}, dims=["time"])

    if time_range is not None:
        oni = oni.sel(time=slice(np.datetime64(time_range[0], "M"),
                                 np.datetime64(time_range[1], "M")))
    return oni

def get_tni_index(ssta):
    """ TNI index (Trenberth & Stepaniak, 2001)"""
    nino_idx = get_nino_indices(ssta)
    n12 = nino_idx['nino12']/ nino_idx['nino12'].std()
    n4 = nino_idx['nino4']/ nino_idx['nino4'].std()
    tni = n12 - n4
    tni.name = 'tni'
    return tni


def get_emi_index(ssta):
    """El Niño Modoki index (EMI; Ashok et al., 2007)."""
    central, central_std = preproc.get_mean_time_series(
        ssta, lon_range=[-165, 140],
        lat_range=[-10, 10]
    )
    eastern, eastern_std =  preproc.get_mean_time_series(
        ssta, lon_range=[-110, -70],
        lat_range=[-15, 5]
    )
    western, western_std =  preproc.get_mean_time_series(
        ssta, lon_range=[125, 145],
        lat_range=[-10, 20]
    )
    emi = central - 0.5 * (eastern + western)
    emi.name = 'emi'
    return emi


def get_epcp_index(ssta):
    """EPnew–CPnew indices (Sullivan et al., 2016)."""
    nino_idx = get_nino_indices(ssta)
    n3 = nino_idx['nino3']/ nino_idx['nino3'].std()
    n4 = nino_idx['nino4']/ nino_idx['nino4'].std()

    ep_idx = n3 - 0.5 * n4
    cp_idx = n4 - 0.5 * n3
    ep_idx.name = 'EPnew'
    cp_idx.name = 'CPnew'
    return ep_idx, cp_idx


def EC_indices(ssta, pc_sign=[1, 1], time_range=None):
    """E and C indices (Takahashi et al., 2011).

    Args:
        ssta (xr.DataArray): Dataarray of SSTA in the region
            lat=[-10,10] and lon=[120E, 70W].
        pc_sign (list, optional): Sign of principal components which can be switched
            for consistency with e.g. Nino-indices. See ambigouosy of sign of PCA.
            For:
                ERA5 data set to [1,-1].
                CMIP6 models set to[-1,-1]
            Defaults to [1,1].

    Returns:
        e_index (xr.Dataarray)
        c_index (xr.Dataarray)
    """

    # Flatten and remove NaNs
    buff = ssta.stack(z=('lat', 'lon'))
    ids = ~np.isnan(buff.isel(time=0).data)
    X = buff.isel(z=ids)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(X.data)

    # Modes
    ts_modes = []
    for i, comp in enumerate(pca.components_):
        ts = stats.zscore(X.data @ comp, axis=0)
        # Flip sign of mode due to ambiguousy of sign
        ts = pc_sign[i] * ts
        ts_modes.append(
            xr.DataArray(data=ts,
                         name=f'eof{i+1}',
                         coords={"time": X.time},
                         dims=["time"])
        )
    ts_mode = xr.merge(ts_modes)

    # Compute E and C index
    # Changed sign of eof2 due to sign flip of it
    e_index = (ts_mode['eof1'] - ts_mode['eof2']) / np.sqrt(2)
    e_index.name = 'E'

    c_index = (ts_mode['eof1'] + ts_mode['eof2']) / np.sqrt(2)
    c_index.name = 'C'

    # Cut time period
    if time_range is not None:
        e_index = e_index.sel(time=slice(
            np.datetime64(time_range[0], "M"), np.datetime64(
                time_range[1], "M")
        ))
        c_index = c_index.sel(time=slice(
            np.datetime64(time_range[0], "M"), np.datetime64(
                time_range[1], "M")
        ))

    return e_index, c_index


def get_pdo_index(ssta):
    """Get time-series of PDO. 
    
    The PDO is defined as the first leading EOF of the Northern Pacific, 
    i.e. 20-70N.

    Args:
        datapath (str): Path to data.

    Returns:
        ts_pdo (xr.Dataarray): Time series of PDO
    """
    ssta_cut = preproc.cut_map(ssta, lon_range=[120, -71], lat_range=[20, 70])
    dataset, loader = utdata.data2dataset(ssta,
        SpatialData, data_2d=True, shuffle=False,
    )
    # PCA
    pca = SpatioTemporalPCA(dataset, n_components=2)
    ts = pca.get_principal_components().T
    ts_pdo = xr.DataArray(
        ts[:, 0], dims=['time'],
        coords={'time': dataset.time},
        name='pdo'
    )
    return ts_pdo 



#########################################################################################
# ENSO classification
#########################################################################################
def get_enso_flavors_N3N4(nino_indices,
                          month_range=[12, 2],
                          mean=True, threshold=0.5,
                          offset=0.0,
                          min_diff=0.0,
                          drop_volcano_year=False):
    """Get nino flavors from Niño‐3–Niño‐4 approach (Kug et al., 2009; Yeh et al.,2009).

    Parameters:
    -----------
        min_diff (float): min_diff between nino3 and nino4 to get only the
                            extreme EP or CP
        threshold (float, str): Threshold to define winter as El Nino or La Nina,
                                A float or 'std' are possible.
                                Default: 0.5.
    """
    if offset > 0.0:
        print("Warning! A new category of El Nino and La Ninas are introduced." )

    if threshold == 'std':
        threshold_nino3 = float(nino_indices['nino3'].std(skipna=True))
        threshold_nino4 = float(nino_indices['nino4'].std(skipna=True))
    else:
        threshold_nino3 = float(threshold)
        threshold_nino4 = float(threshold)
    
    def is_datetime360(time):
        return isinstance(time, cftime._cftime.Datetime360Day)

    def is_datetime(time):
        return isinstance(time, cftime._cftime.DatetimeNoLeap)

    # Identify El Nino and La Nina types
    enso_classes = []
    sd, ed = np.array([nino_indices.time.data.min(), nino_indices.time.data.max()])
    if is_datetime360(nino_indices.time.data[0]) or is_datetime(nino_indices.time.data[0]):
        times = xr.cftime_range(start=sd,
                                end=ed,
                                freq='Y')
    else:
        times = np.arange(
            np.array(sd, dtype='datetime64[Y]'),
            np.array(ed, dtype='datetime64[Y]')
        )
    for y in times:
        if is_datetime360(nino_indices.time.data[0]):
            y = y.year
            y_end = y+1 if month_range[1] < month_range[0] else y
            time_range = [cftime.Datetime360Day(y, month_range[0], 1),
                          cftime.Datetime360Day(y_end, month_range[1]+1, 1)]
        elif is_datetime(nino_indices.time.data[0]):
            y = y.year
            y_end = y+1 if month_range[1] < month_range[0] else y
            time_range = [cftime.DatetimeNoLeap(y, month_range[0], 1),
                          cftime.DatetimeNoLeap(y_end, month_range[1]+1, 1)]
        else:
            y_end = y+1 if month_range[1] < month_range[0] else y
            time_range = [np.datetime64(f"{y}-{month_range[0]:02d}-01", "D"),
                          np.datetime64(f"{y_end}-{month_range[1]+1:02d}-01", "D")-1]

        # Select time window
        nino34 = nino_indices['nino34'].sel(time=slice(time_range[0], time_range[1]))
        nino3 = nino_indices['nino3'].sel(time=slice(time_range[0], time_range[1]))
        nino4 = nino_indices['nino4'].sel(time=slice(time_range[0], time_range[1]))

        # Choose mean or min
        if mean:
            nino34 = nino34.mean(dim='time', skipna=True)
            nino3 = nino3.mean(dim='time', skipna=True)
            nino4 = nino4.mean(dim='time', skipna=True)
        else:
            nino34 = nino34.min(dim='time', skipna=True)
            nino3 = nino3.min(dim='time', skipna=True)
            nino4 = nino4.min(dim='time', skipna=True)

        # El Nino years
        if ((nino3.data >= threshold_nino3) or (nino4.data >= threshold_nino4)):
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nino', 'N3': float(nino3), 'N4': float(nino4)}
            buff_dic['N3-N4'] = nino3.data - nino4.data

            Nino_EP_label = 'Nino_EP_weak' if offset > 0 else 'Nino_EP'
            Nino_CP_label = 'Nino_CP_weak' if offset > 0 else 'Nino_CP'

            # EP type if DJF nino3 > 0.5 and nino3 > nino4
            if (nino3.data - min_diff) > nino4.data:
                buff_dic['type'] = Nino_EP_label
            # CP type if DJF nino4 > 0.5 and nino3 < nino4
            elif (nino4.data - min_diff) > nino3.data:
                buff_dic['type'] = Nino_CP_label

            # Strong El Ninos
            if offset > 0.0:
                if (nino3.data >= threshold_nino3 + offset) and (nino3.data - min_diff) > nino4.data:
                    buff_dic['type'] = "Nino_EP_strong"
                elif (nino4.data >= threshold_nino4 + offset) and (nino4.data - min_diff) > nino3.data:
                    buff_dic['type'] = 'Nino_CP_strong'

            enso_classes.append(buff_dic)

        # La Nina years
        elif ((nino3.data <= -threshold_nino3) or (nino4.data <= -threshold_nino4)):
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nina', 'N3': float(nino3), 'N4': float(nino4)}
            buff_dic['N3-N4'] = nino3.data - nino4.data

            Nina_EP_label = 'Nina_EP_weak' if offset > 0 else 'Nina_EP'
            Nina_CP_label = 'Nina_CP_weak' if offset > 0 else 'Nina_CP'

            # EP type if DJF nino3 < -0.5 and nino3 < nino4
            if (nino3.data + min_diff) < nino4.data:
                buff_dic['type'] = Nina_EP_label
            # CP type if DJF nino4 < -0.5 and nino3 > nino4
            elif (nino4.data + min_diff) < nino3.data:
                buff_dic['type'] = Nina_CP_label

            # Strong La Nina
            if offset > 0.0:
                if (nino3.data <= -threshold_nino3 - offset) and (nino3.data + min_diff) < nino4.data:
                    buff_dic['type'] = "Nina_EP_strong"
                elif (nino4.data <= -threshold_nino4 - offset) and (nino4.data + min_diff) < nino3.data:
                    buff_dic['type'] = 'Nina_CP_strong'

            enso_classes.append(buff_dic)

        # standard years
        else:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Normal', 'N3': float(nino3), 'N4': float(nino4)}
            buff_dic['N3-N4'] = nino3.data - nino4.data
            enso_classes.append(buff_dic)

    enso_classes = pd.DataFrame(enso_classes)

    # Years of strong volcanic erruptions followed by an El Nino
    if drop_volcano_year:
        volcano_years_idx = enso_classes.loc[
            (enso_classes['start'] == '1955-12-01') |
            (enso_classes['start'] == '1956-12-01') |
            (enso_classes['start'] == '1957-12-01') |
            (enso_classes['start'] == '1963-12-01') |
            (enso_classes['start'] == '1980-12-01') |
            (enso_classes['start'] == '1982-12-01') |
            (enso_classes['start'] == '1991-12-01')
        ].index
        enso_classes = enso_classes.drop(index=volcano_years_idx)

    return enso_classes


def get_enso_flavor_EC(e_index, c_index, month_range=[12, 2],
                       offset=0.0, mean=True, nino_indices=None):
    """Classify winters into their ENSO flavors based-on the E- and C-index.

    The E- and C-index was introduced by Takahashi et al. (2011).
    The following criterias are used:


    Args:
        e_index (xr.DataArray): E-index.
        c_index (xr.DataArray): C-index.
        month_range (list, optional): Month range where to consider the criteria.
            Defaults to [12,2].
        offset (float, optional): Offset to identify only extremes of the flavors.
            Defaults to 0.0.
        mean (boolean, optional): If True the mean of the range must exceed the threshold.
            Otherwise all months within the range must exceed the threshold.
            Defaults to True.

    Returns:
        enso_classes (pd.DataFrame): Dataframe containing the classification.
    """
    e_threshold = e_index.std(dim='time', skipna=True) + offset
    c_threshold = c_index.std(dim='time', skipna=True) + offset

    years = np.arange(
        np.array(e_index.time.min(), dtype='datetime64[Y]'),
        np.array(e_index.time.max(), dtype='datetime64[Y]')
    )
    enso_classes = []
    for y in years:
        time_range = [np.datetime64(f"{y}-{month_range[0]:02d}-01", "D"),
                      np.datetime64(f"{y+1}-{month_range[1]+1:02d}-01", "D")-1]
        # Either mean or min of DJF must exceed threshold
        e_range = e_index.sel(time=slice(*time_range))
        c_range = c_index.sel(time=slice(*time_range))
        if mean:
            e_range = e_range.mean(dim='time', skipna=True)
            c_range = c_range.mean(dim='time', skipna=True)

        # TODO: Nino indices for pre-selection might be obsolete
        # Preselect EN and LN conditions based on Nino34
        if nino_indices is not None:
            nino34 = nino_indices['nino34'].sel(
                time=slice(time_range[0], time_range[1]))
            nino3 = nino_indices['nino3'].sel(
                time=slice(time_range[0], time_range[1]))
            nino4 = nino_indices['nino4'].sel(
                time=slice(time_range[0], time_range[1]))

            # Normal conditions
            if ((nino34.min() >= -0.5 and nino34.max() <= 0.5)
                or (nino3.min() >= -0.5 and nino3.max() <= 0.5)
                    or (nino4.min() >= -0.5 and nino4.max() <= 0.5)):
                buff_dic = {'start': time_range[0], 'end': time_range[1],
                            'type': 'Normal'}
                enso_classes.append(buff_dic)
                continue

        # EPEN
        if (e_range.min() > 0) & (c_range.min() > 0):
            if e_range.min() > c_range.min():
                buff_dic = {'start': time_range[0], 'end': time_range[1],
                            'type': 'Nino_EP'}
            else:
                buff_dic = {'start': time_range[0], 'end': time_range[1],
                            'type': 'Nino_CP'}

        elif (e_range.min() < 0) & (c_range.min() < 0):
            if e_range.min() > c_range.min():
                buff_dic = {'start': time_range[0], 'end': time_range[1],
                            'type': 'Nina_EP'}
            else:
                buff_dic = {'start': time_range[0], 'end': time_range[1],
                            'type': 'Nina_CP'}
        else:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Normal'}

        enso_classes.append(buff_dic)

    return pd.DataFrame(enso_classes)


def get_enso_flavors_obs(definition='N3N4', fname=None,
                         vname='sst', climatology='month',
                         detrend_from=1950, month_range=[12, 2],
                         time_range=None, threshold=0.5, offset=0.0):
    """Classifies given month range into ENSO flavors.

    Args:
        definition (str, optional): Definition used for classification.
            Defaults to 'N3N4'.
        fname (str, optional): Each definition might require information of other
            datasets, i.e.:
                'N3N4' requires the global SST dataset.
                'EC' requires the global SST dataset for the EOF analysis.
                'N3N4_NOAA' requires the nino-indices by NOAA
                'Cons' requires a table of classifications.
            Defaults to None which uses the preset paths.
        vname (str): Varname of SST only required for 'N3N4' and 'EC'. Defaults to 'sst'.
        land_area_mask (xr.Dataarray): Land area fraction mask, i.e. 0 over oceans.
            Defaults to None.
        climatology (str, optional): Climatology to compute anomalies.
            Only required for 'N3N4' and 'EC'. Defaults to 'month'.
        month_range (list, optional): Month range. Defaults to [11,1].
        time_range (list, optional): Time period of interest.
            Defaults to None.
        offset (float, optional): Offset for 'extreme' events.
            Defaults to 0.0.

    Raises:
        ValueError: If wrong definition is defined.

    Returns:
        (pd.Dataframe) Containing the classification including the time-period.
    """
    if definition in ['N3N4', 'EC']:
        if fname is None:
            raise ValueError(f"Attribute fname must be set if definition={definition}!")
        # Process global SST data
        da_sst = xr.open_dataset(fname)[vname]

        # Check dimensions
        # TODO: remove sorting here, this takes forever
        da_sst = preproc.check_dimensions(da_sst, sort=True)
        # Detrend data
        da_sst = preproc.detrend_dim(da_sst, dim='time', startyear=detrend_from)
        # Anomalies
        ssta = preproc.compute_anomalies(da_sst, group=climatology)

    if definition == 'N3N4':
        nino_indices = get_nino_indices(ssta, time_range=time_range)
        enso_classes = get_enso_flavors_N3N4(
            nino_indices, month_range=month_range, mean=True,
            threshold=threshold, offset=offset,
            min_diff=0.1, drop_volcano_year=False
        )
    elif definition == 'EC':
        # Cut area for EOFs
        lon_range = [120, -80]
        lat_range = [-10, 10]
        ssta = preproc.cut_map(
            ds=ssta, lon_range=lon_range, lat_range=lat_range, shortest=True
        )

        # EC-index based-on EOFs
        e_index, c_index = EC_indices(
            ssta, pc_sign=[1, -1], time_range=time_range
        )
        enso_classes = get_enso_flavor_EC(
            e_index, c_index, month_range=month_range, mean=True,
            offset=offset
        )
    elif definition == 'N3N4_NOAA':
        # N3N4 approach
        if fname is None:
            fname = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
        nino_indices = get_nino_indices_NOAA(
            fname, time_range=time_range, time_roll=0)
        enso_classes = get_enso_flavors_N3N4(
            nino_indices, month_range=month_range, mean=True, threshold=0.5,
            offset=offset, min_diff=0.1, drop_volcano_year=False
        )
    else:
        raise ValueError(f"Specified ENSO definition type {definition} is not defined! "
                         + "The following are defined: 'N3N4', 'Cons', 'EC'")

    return enso_classes


def get_enso_flavors_cmip(fname_sst, vname='ts', land_area_mask=None, climatology='month',
                          definition='N3N4', month_range=[12, 2],
                          time_range=None, offset=0.0, detrend_from=1950):
    """Classifies CMIP data into ENSO flavors.

    Args:
        fname_sst (str): Path to global SST dataset.
        vname (str): Varname of SST. Defaults to 'ts'.
        land_area_mask (xr.Dataarray): Land area fraction mask, i.e. 0 over oceans.
            Defaults to None.
        climatology (str, optional): Climatology to compute anomalies. Defaults to 'month'.
        definition (str, optional): Definition used for classification.
            Defaults to 'N3N4'.
        month_range (list, optional): Month range. Defaults to [11,1].
        time_range (list, optional): Time period of interest.
            Defaults to None.
        offset (float, optional): Offset for 'extreme' events.
            Defaults to 0.0.

    Raises:
        ValueError: If wrong definition is defined.

    Returns:
        (pd.Dataframe) Containing the classification including the time-period.
    """
    # Process global SST data
    da_sst = xr.open_dataset(fname_sst)[vname]
    # Mask only oceans
    if land_area_mask is not None or definition != 'N3N4':
        da_sst = da_sst.where(land_area_mask == 0.0)

    # Check dimensions
    da_sst = preproc.check_dimensions(da_sst, sort=True)
    # Detrend data
    if detrend_from is not None:
        da_sst = preproc.detrend_dim(da_sst, startyear=detrend_from)
    # Anomalies
    if climatology is not None:
        ssta = preproc.compute_anomalies(da_sst, group=climatology)
    else:
        ssta = da_sst

    # ENSO event classification 
    if definition == 'N3N4':
        nino_indices = get_nino_indices(ssta, time_range=time_range)
        enso_classes = get_enso_flavors_N3N4(
            nino_indices, month_range=month_range, mean=True, threshold=0.5,
            offset=offset, min_diff=0.1, drop_volcano_year=False
        )
    elif definition == 'EC':
        # Cut area for EOFs
        lon_range = [120, -80]
        lat_range = [-10, 10]
        ssta = preproc.cut_map(
            ds=ssta, lon_range=lon_range, lat_range=lat_range, shortest=True
        )

        # EC-index based-on EOFs
        e_index, c_index = EC_indices(
            ssta, pc_sign=[-1, -1], time_range=time_range
        )
        enso_classes = get_enso_flavor_EC(
            e_index, c_index, month_range=month_range, mean=True,
            offset=offset
        )
    else:
        raise ValueError(f"Specified ENSO definition type {definition} is not defined! "
                         + "The following are defined: 'N3N4', 'Cons', 'EC'")
    return enso_classes



def select_enso_events(ds: xr.Dataset, month_range=[12, 2], threshold=0.5,
                       include_normal=False):
    """Select enso events from dataset with multible members."""
    x_enso = []
    x_events = []
    for member in np.unique(ds['member']):
        idx_member = np.where(ds['member'] == member)[0]
        x_member = ds.isel(time=idx_member)

        nino_ids = get_nino_indices(x_member['ssta'], antimeridian=True)
        enso_classes = get_enso_flavors_N3N4(nino_ids, month_range=month_range,
                                             threshold=threshold)

        enso_classes = enso_classes.loc[~np.isnan(enso_classes['N3'])]
        if not include_normal:
            enso_classes = enso_classes.loc[enso_classes['type'] != 'Normal']

        x_member_enso = []
        x_member_events = []
        times = []
        for i, time_period in enso_classes.iterrows():
            buff = x_member.sel(time=slice(time_period['start'], time_period['end']))
            buff = buff.assign_coords(enso=('time', len(buff['time']) * [time_period['type']]))
            x_member_enso.append(buff)
            x_member_events.append(
                x_member.sel(time=slice(time_period['start'], time_period['end'])).mean(dim='time')
            )
            times.append(time_period['start'])

        x_member_events = xr.concat(x_member_events, dim=pd.Index(times, name='time'))
        x_member_enso = xr.concat(x_member_enso, dim='time')

        x_member_events = x_member_events.assign_coords(member=('time', len(x_member_events['time']) * [member]),
                                                        enso=('time', enso_classes['type'].values))
        x_member_enso = x_member_enso.assign_coords(member=('time', len(x_member_enso['time']) * [member]))

        x_events.append(x_member_events)
        x_enso.append(x_member_enso)

    x_events = xr.concat(x_events, dim='time')
    x_enso = xr.concat(x_enso, dim='time')

    if ds.attrs is not {}:
        x_enso.attrs = ds.attrs
        x_events.attrs = ds.attrs

    return x_enso, x_events


#########################################################################################
# Weighting of ENSO events
#########################################################################################
def mahalanobis_weight(nino_indices, gmm, normalize=False):
    """Compute mahalanobis distance to gaussians.

    Args:
        nino_indices (xr.Dataset): Dataset which must containing variables 'nino34' and 'tni'.
        gmm (dict): Dictionary including the parameters of the GMM,
                    i.e. means, covariances, weights and classes
        normalize (bool, optional): If True, the densitys are normalized between 0 and 1. 
                                    Defaults to False.

    Returns:
        (xr.Dataset) Containing the mahalanobis distance of each time point and class.
    """
    X = np.array([nino_indices['nino34'].data, nino_indices['tni'].data]).T
    class_mhlb = []
    for i, cl in enumerate(gmm['classes']):
        mhlb = []
        mean = gmm['means'][i]
        cov_inv = np.linalg.inv(gmm['covariances'][i])
        for n, x in enumerate(X):
            mhlb.append(dist.mahalanobis(x, mean, cov_inv))

        da = xr.DataArray(data=mhlb, dims=['time'],
                          coords=dict(time=nino_indices['time'].data), name=cl)

        if normalize:
            class_mhlb.append(1 - preproc.normalize(da, method='minmax'))
        else:
            class_mhlb.append(da)
        
    return xr.merge(class_mhlb)


def likelihood_weight(nino_indices, gmm, normalize=False):
    """ENSO weighting based on likelihood p(x|c).

    Args:
        nino_indices (xr.Dataset): Dataset which must containing variables 'nino34' and 'tni'.
        gmm (dict): Dictionary including the parameters of the GMM,
                    i.e. means, covariances, weights and classes
        normalize (bool, optional): If True, the densitys are normalized between 0 and 1. 
                                    Defaults to False.

    Returns:
        _type_: _description_
    """
    X = np.array([nino_indices['nino34'].data, nino_indices['tni'].data]).T
    da_ps = []
    for i, cl in enumerate(gmm['classes']):
        p_x = stats.multivariate_normal.pdf(X, mean=gmm['means'][i],
                                            cov=gmm['covariances'][i])

        da = xr.DataArray(data=p_x, dims=['time'],
                            coords=dict(time=nino_indices['time'].data), name=cl)

        if normalize:
            da_ps.append(
                preproc.normalize(da, method='minmax')
            )
        else:
            da_ps.append(da)

    return xr.merge(da_ps)


def log_gaussians(X, means, covariances):
    """Estimate the log Gaussian probability.

    Implementation taken from sklearn [1].

    Args:
        X (np.ndarray): array-like of shape (n_samples, n_features)
        means (np.ndarray): array-like of shape (n_components, n_features)
        covariances (np.ndarray) : array-like covariance matrices of 
                                shape (n_components, n_features, n_features)

    Returns:
        log_prob (np.ndarray): array of shape (n_samples, n_components)
    
    Ref:
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    # Precision matrices of covariance 
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError("Only works for positive semi-definite matrices.")
        precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                     np.eye(n_features),
                                                     lower=True).T

    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)
    
    # Log determinant
    # det(precision_chol) is half of det(precision)
    log_det_chol = (np.sum(np.log(
        precisions_chol.reshape(
            n_components, -1)[:, ::n_features + 1]), 1))

    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det_chol


def posterior_weight_n3_tni(nino_indices, gmm):
    """ENSO weighting based on posterior p(c|x).

    Args:
        nino_indices (xr.Dataset): Dataset which must containing variables 'nino34' and 'tni'.
        gmm (dict): Dictionary including the parameters of the GMM,
                    i.e. means, covariances, weights and classes
        normalize (bool, optional): If True, the densitys are normalized between 0 and 1. 
                                    Defaults to False.

    Returns:
        xr.Dataset: _description_
    """
    X = np.array([nino_indices['nino34'].data, nino_indices['tni'].data]).T
    n_samples, n_features = X.shape

    # log likelihood p(x|c) of shape (n_samples, n_classes)
    log_p_x_given_c = log_gaussians(X, gmm['means'], gmm['covariances'])    
    # prior p(c) of shape (n_classes)
    p_c = gmm['weights']
    # evidence p(x) of shape (n_samples)
    p_x = np.exp(log_p_x_given_c) @ p_c

    da_ps = []
    for i, cl in enumerate(gmm['classes']):
        # log posterior p(c|x)
        log_p_c_given_x = log_p_x_given_c[:, i] + np.repeat([np.log(p_c[i])], repeats=n_samples, axis=0) - np.log(p_x)
        p_c_given_x = np.exp(log_p_c_given_x)

        da_ps.append(
            xr.DataArray(data=p_c_given_x, dims=['time'],
                        coords=dict(time=nino_indices['time'].data), name=cl)
        )

    return xr.merge(da_ps)


def posterior_weights(X, means, covariances, prior_weights):
    """Compute posterior probabilities p(c|x) for Gaussian mixtures.

    Args:
        X (np.ndarray): Input samples of shape (n_samples, n_features)
        means (np.ndarray): Means of gaussians of shape (n_classes, n_features)
        covariances (np.ndarray): Covariances of gaussians of shape (n_classes, n_features, n_features)
        prior_weights (np.ndarray): Weights of gaussians of shape (n_classes)

    Returns:
        p_c_given_x (np.ndarray): Posterior of samples of shape (n_samples, n_classes) 
    """
    n_samples, n_features = X.shape
    n_classes = means.shape[0]

    # log likelihood p(x|c) of shape (n_samples, n_classes)
    log_p_x_given_c = log_gaussians(X, means, covariances)    
    # prior p(c) of shape (n_classes)
    p_c = prior_weights
    # evidence p(x) of shape (n_samples)
    p_x = np.exp(log_p_x_given_c) @ p_c

    p_c_given_x = []
    for i in range(n_classes):
        # log posterior p(c|x)
        log_p_c_given_x = log_p_x_given_c[:, i] + np.repeat([np.log(p_c[i])], repeats=n_samples, axis=0) - np.log(p_x)
        p_c_given_x.append(np.exp(log_p_c_given_x))
    
    return np.array(p_c_given_x).T


def evidence(X: np.ndarray, means: np.ndarray, covariances: np.ndarray,
             prior_weights: np.ndarray) -> np.ndarray:
    """Compute evidence p(x) = sum_k p(x|c) p(c) for Gaussian mixtures.

    Args:
        X (np.ndarray): Input samples of shape (n_samples, n_features)
        means (np.ndarray): Means of gaussians of shape (n_classes, n_features)
        covariances (np.ndarray): Covariances of gaussians of shape (n_classes, n_features, n_features)
        prior_weights (np.ndarray): Weights of gaussians of shape (n_classes)

    Returns:
        p_c_given_x (np.ndarray): Posterior of samples of shape (n_samples, n_classes) 
    """
    n_samples, n_features = X.shape
    n_classes = means.shape[0]

    # log likelihood p(x|c) of shape (n_samples, n_classes)
    log_p_x_given_c = log_gaussians(X, means, covariances)    
    # prior p(c) of shape (n_classes)
    p_c = prior_weights
    # evidence p(x) of shape (n_samples)
    p_x = np.exp(log_p_x_given_c) @ p_c
    

    return p_x

# Composites
########################################################################################
def get_unweighted_composites(ds: xr.Dataset, f_sst: str,
                              enso_types: list=['Nino_EP', 'Nino_CP','Nina_EP', 'Nina_CP'],
                              criteria: str='N3N4', enso_months: list=[12, 2],
                              month_offset: int=0,
                              stattest: str='pos', alpha: float=0.05,
                              null_hypothesis: str='neutral',
                              n_samples_mean: int=100, n_samples_time: int=12,
                              serial_data: bool=False, multiple_testing: str='dunn'):
    """Get unweighted ENSO composites for dataarray.

    Args:
        ds (xr.Dataset): Data to compute composites from
        f_sst (str): File to sea surface temperature data over the same time period.
        enso_types (list, optional): ENSO types. Defaults to ['Nino_EP', 'Nino_CP','Nina_EP', 'Nina_CP'].
        criteria (str, optional): Selection criteria for ENSO types. Defaults to 'N3N4'.
        enso_months (list, optional): Months for ENSO. Defaults to [12, 2].
        month_offset (int, optional): Offset of month. Defaults to 0.
        stattest (str, optional): Statistical test. Defaults to 'pos'.
        alpha (float, optional): Significance threshold. Defaults to 0.05.
        null_hypothesis (str, optional): Null hypothesis. Defaults to 'neutral'.
        n_samples_mean (int, optional): _description_. Defaults to 100.
        n_samples_time (int, optional): _description_. Defaults to 12.
        serial_data (bool, optional): _description_. Defaults to False.
        multiple_testing (str, optional): _description_. Defaults to 'dunn'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        composites, masks, composite_null, samples_null, pvalues
    """
    # Split data into enso-types
    enso_classes = get_enso_flavors_obs(
        definition=criteria, fname=f_sst, vname='sst', climatology='month',
        month_range=enso_months
    )

    # Null hypothesis
    if null_hypothesis == 'neutral':
        # Neutral years as null hypothesis
        time_snippets_null = np.array(
            [enso_classes.loc[enso_classes['type'] == 'Normal']['start'],
             enso_classes.loc[enso_classes['type'] == 'Normal']['end']]
        ).T
    elif null_hypothesis == 'all':
        # All winters as null hypothesis
        time_snippets_null = np.array(
            [enso_classes['start'],
             enso_classes['end']]
        ).T
    else:
        raise ValueError(f"Unknown null hypothesis: {null_hypothesis}")

    composite_vars = []
    mask_vars = []
    composite_null = []
    samples_null = []
    pvalues_arr = []
    for var in ds.data_vars:
        da = ds[var]

        # Neutral DJF
        da_null = preproc.select_time_snippets(da, time_snippets_null)
        composite_null.append(da_null.mean(dim='time', skipna=True))

        if stattest in ['ks', 'mw', 'pos']:
            # Sample means from null hypothesis
            samples_null_var = []
            for n in range(n_samples_mean):
                time_samples = np.random.choice(da_null['time'], size=n_samples_time,
                                                replace=True)
                samples_null_var.append(da_null.sel(
                    time=time_samples).mean(dim='time'))
            samples_null_var = xr.concat(samples_null_var, dim='samples')
            samples_null_var.name = var
            samples_null.append(samples_null_var)
        else:
            samples_null = None

        # Mean composites and sign mask
        samples_mean = []
        composite_flavor_arr = []
        mask_flavor_arr = []
        pvalues_var = []
        for i, enso_type in enumerate(enso_types):
            # Composites
            time_snippets = np.array(
                [enso_classes.loc[enso_classes['type'] == enso_type]['start'] + np.timedelta64(month_offset, 'M'),
                 enso_classes.loc[enso_classes['type'] == enso_type]['end'] + np.timedelta64(month_offset, 'M')]
            ).T
            da_flavor = preproc.select_time_snippets(da, time_snippets)
            print(f"Num of datapoints {enso_type}: {len(da_flavor['time'])}")

            # Mean
            mean = da_flavor.mean(dim='time', skipna=True)

            if stattest in ['ks', 'mw']:
                # Sample means
                samples_class = []
                for n in range(n_samples_mean):
                    time_samples = np.random.choice(da_flavor['time'], size=n_samples_time,
                                                    replace=True)
                    samples_class.append(da_flavor.sel(
                        time=time_samples).mean(dim='time'))
                samples_class = xr.concat(samples_class, dim='samples')

                # TODO: remove
                samples_mean.append(samples_class)

            if stattest == 'ks':
                print(f"KS-test for {enso_type}")
                _, pvals = utstats.kstest_field(samples_class, samples_null_var)

            elif stattest == 'mw':
                print(f"Mannwhitneyu test for {enso_type}")
                _, pvalues = stats.mannwhitneyu(
                    samples_class.data, samples_null_var.data,
                    use_continuity=True, alternative='two-sided',
                    axis=0, method='auto', nan_policy='propagate', keepdims=False
                )
                pvals = xr.DataArray(data=pvalues, coords=mean.coords)

            elif stattest == 'ttest':
                _, pvals = utstats.ttest_field(
                    da_flavor, da_null, serial_data=serial_data
                )
            elif stattest == 'pos':
                print(f"Percentile of score for {enso_type}")
                X = samples_null_var.stack(z=('lat', 'lon'))
                y = mean.stack(z=('lat', 'lon'))
                pvalues = xr.ones_like(y) * np.nan
                for i in tqdm(range(len(y['z']))):
                    p = stats.percentileofscore(
                        X.isel(z=i), y.isel(z=i), kind='weak')
                    if p > 50:
                        p = 100 - p
                    pvalues[i] = p/100
                pvals = pvalues.unstack('z')
            else:
                raise ValueError(
                    f"Specified stattest={stattest} does not exist!")

            mask = utstats.field_significance_mask(
                pvals, alpha=alpha, corr_type=multiple_testing)

            # Save to arr
            composite_flavor_arr.append(mean)
            mask_flavor_arr.append(mask)
            pvalues_var.append(pvals)

        # Concat for each flavor
        composite_flavor = xr.concat(composite_flavor_arr,
                                     dim=pd.Index(enso_types, name='classes'))
        composite_vars.append(composite_flavor)

        mask_flavor = xr.concat(mask_flavor_arr,
                                dim=pd.Index(enso_types, name='classes'))
        mask_flavor.name = var
        mask_vars.append(mask_flavor)

        pvalues_var = xr.concat(pvalues_var, dim=pd.Index(enso_types, name='classes'))
        pvalues_var.name = var  
        pvalues_arr.append(pvalues_var)

        # TODO: remove
        #samples_mean = xr.concat(samples_mean, dim=pd.Index(enso_types, name='classes'))

    composites = xr.merge(composite_vars)
    masks = xr.merge(mask_vars)
    composite_null = xr.merge(composite_null)
    pvalues = xr.merge(pvalues_arr)

    if stattest in ['ks', 'mw', 'pos']:
        samples_null = xr.merge(samples_null)

    return composites, masks, composite_null, samples_null, pvalues


def get_weighted_composites(ds: xr.Dataset, f_sst: str, weights: xr.DataArray,
                            null_hypothesis: str = 'all', stattest: str = 'ks',
                            n_samples_time: int = 100, n_samples_mean: int = 100,
                            alpha: float = 0.05,
                            multiple_testing: str='dunn', serial_data: bool = False):
    """Create weighted composites and significant mask using conditional probabilities.

    Args:
        ds (xr.Dataset): _description_
        f_sst (str): _description_
        weights (xr.DataArray): _description_
        null_hypothesis (str, optional): _description_. Defaults to 'all'.
        stattest (str, optional): _description_. Defaults to 'ks'.
        n_samples_time (int, optional): _description_. Defaults to 100.
        n_samples_mean (int, optional): _description_. Defaults to 100.
        alpha (float, optional): _description_. Defaults to 0.05.
        multiple_testing (str, optional): _description_. Defaults to 'dunn'.
        serial_data (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Make sure dataset for composites are on the same time points as weights
    tmax = ds['time'].max() if ds['time'].max(
    ) < weights['time'].max() else weights['time'].max()
    tmin = ds['time'].min() if ds['time'].min(
    ) > weights['time'].min() else weights['time'].min()
    weights = weights.sortby(weights['time'])
    weights = weights.sel(time=slice(tmin.data, tmax.data))

    weights['time'] = np.array(weights['time'].data, dtype='datetime64[M]')
    ds['time'] = np.array(ds['time'].data, dtype='datetime64[M]')

    ds = ds.sel(time=weights['time'])
    assert len(ds['time']) == len(weights['time'])

    # Null hypothesis
    enso_classes = get_enso_flavors_obs(
        definition='N3N4', fname=f_sst, vname='sst', climatology='month',
        month_range=[12, 2],
    )
    if null_hypothesis == 'neutral':
        # Neutral years as null hypothesis
        time_snippets = np.array(
            [enso_classes.loc[enso_classes['type'] == 'Normal']['start'],
             enso_classes.loc[enso_classes['type'] == 'Normal']['end']]
        ).T
    elif null_hypothesis == 'all':
        # All winters as null hypothesis
        time_snippets = np.array(
            [enso_classes['start'],
             enso_classes['end']]
        ).T
    else:
        raise ValueError(f"Unknown null hypothesis: {null_hypothesis}")
    print(f"Len of null-times: {len(time_snippets)}")

    # Get weighted composites and statistical significance
    composite_vars = []
    mask_vars = []
    samples_null = []
    pvals_vars = []
    for var in ds.data_vars:
        # Mean and std at each location for standardization
        print(f"Compute mean and stat-test for {var}")
        da = ds[var]

        # Null hypothesis
        da_null = preproc.select_time_snippets(da, time_snippets)

        if stattest in ['ks', 'mw', 'pos']:
            # Sample means from null hypothesis
            samples_null_var = []
            for n in range(n_samples_mean):
                time_samples = np.random.choice(da_null['time'], size=n_samples_time,
                                                replace=True)
                samples_null_var.append(da_null.sel(
                    time=time_samples).mean(dim='time'))

            samples_null_var = xr.concat(samples_null_var, dim='sample')
            samples_null_var.name = f"{var}"
            samples_null.append(samples_null_var)
        else:
            samples_null = None

        # Get weighted composites and statistical significance
        samples_mean = []
        composite_class_arr = []
        mask_class_arr = []
        classes = []
        pvals_class_arr = []
        for i, k in enumerate(weights['classes'].data):
            weight_class = weights.sel(classes=k)

            # Weighted mean
            da_weighted = da.weighted(weight_class)
            weighted_mean = da_weighted.mean(dim='time')

            if stattest in ['ks', 'mw']:
                # Sample means using weights
                # Convert weights to probabilities
                prob_time = weight_class / weight_class.sum(dim='time')

                samples_class = []
                for n in range(n_samples_mean):
                    time_samples = np.random.choice(da['time'], size=n_samples_time,
                                                    replace=True, p=prob_time.data)
                    samples_class.append(
                        da.sel(time=time_samples).mean(dim='time'))
                samples_class = xr.concat(samples_class, dim='samples')

                samples_mean.append(samples_class)

            if stattest == 'ks':
                print(f"KS-test for c={k}")
                _, pvals = utstats.kstest_field(samples_class, samples_null_var)
            elif stattest == 'mw':
                print(f"Mannwhitneyu test for c={k}")
                _, pvalues = stats.mannwhitneyu(
                    samples_class.data, samples_null_var.data,
                    use_continuity=True, alternative='two-sided',
                    axis=0, method='auto', nan_policy='propagate', keepdims=False
                )
                pvals = xr.DataArray(data=pvalues, coords=weighted_mean.coords)
            elif stattest == 'ttest':
                _, pvals = utstats.ttest_field(
                    da_weighted, da_null, weights=weight_class, serial_data=serial_data
                )
            elif stattest == 'pos':
                print(f"Percentile of score for c={k}")
                X = samples_null_var.stack(z=('lat', 'lon'))
                y = weighted_mean.stack(z=('lat', 'lon'))
                pvalues = xr.ones_like(y) * np.nan
                for i in tqdm(range(len(y['z']))):
                    p = stats.percentileofscore(
                        X.isel(z=i).data, y.isel(z=i).data, kind='mean',
                        nan_policy='propagate')
                    if p > 50:
                        p = 100 - p
                    pvalues[i] = p/100 
                pvals = pvalues.unstack('z')
            else:
                raise ValueError(
                    f"Specified stattest={stattest} does not exist!")

            mask = utstats.field_significance_mask(
                pvals, alpha=alpha, corr_type=multiple_testing)

            composite_class_arr.append(weighted_mean)
            mask_class_arr.append(mask)
            classes.append(k)
            pvals_class_arr.append(pvals)

        composite_classes = xr.concat(
            composite_class_arr, dim=pd.Index(classes, name='classes')
        )
        composite_classes.name = var
        mask_classes = xr.concat(
            mask_class_arr, dim=pd.Index(classes, name='classes')
        )
        mask_classes.name = var

        pvals_class_arr = xr.concat(pvals_class_arr, dim=pd.Index(classes, name='classes'))
        pvals_class_arr.name = var
        pvals_vars.append(pvals_class_arr)

        composite_vars.append(composite_classes)
        mask_vars.append(mask_classes)

    composites = xr.merge(composite_vars)
    masks = xr.merge(mask_vars)
    pvals_vars = xr.merge(pvals_vars)

    if stattest in ['ks', 'mw', 'pos']:
        samples_null = xr.merge(samples_null)

    return composites, masks, samples_null, pvals_vars




def get_weighted_composites_following_enso(ds: xr.Dataset, f_sst: str, weights: xr.DataArray, 
                                 threshold: float = None, stattest: str = 'ks',
                                 alpha: float=0.05, months: list=[6,8],
                                 serial_data: bool=False):
    """Create weighted composites and significant mask using conditional probabilities.

    Args:
        ds (xr.Dataset): Dataset to create composites from with dim=['lat', 'lon', 'time']
        f_sst (str): Path to SST dataset.
        weights (xr.DataArray): Conditional probabilities of events of dim=['classes', 'time]
        threshold (float, optional): Threshold on weights. Defaults to None.
        stattest (str, optional): Statistical significance test. Defaults to 'ks'.
        alpha (float, optional): Alpha value for stat. significance. Defaults to 0.05.
        enso_months (list, optional): Months to compute null hypothesis over. Defaults to [12,2].
        serial_data (bool, optional): Use serial data correction. Defaults to False.
    """
    # Assign event weights to months of interest
    weights_arr = []
    for month in range(months[0], months[1]+1):
        times = [np.datetime64(f'{y+1}-{month:02d}-01', 'D') for y in np.array(weights['time'], dtype='datetime64[Y]')]
        weights_arr.append(
            xr.DataArray(
                data= weights.data,
                coords={'time': times, 'classes': weights['classes']}
            )
        )
    weights = xr.concat(weights_arr, dim='time').sortby('time')

    # Make sure dataset for composites are on the same time points as weights
    tmax = ds['time'].max() if ds['time'].max() < weights['time'].max() else weights['time'].max()
    tmin = ds['time'].min() if ds['time'].min() > weights['time'].min() else weights['time'].min()
    weights = weights.sortby(weights['time']) 
    weights = weights.sel(time=slice(tmin.data, tmax.data))

    weights['time'] = np.array(weights['time'].data, dtype='datetime64[M]')
    ds['time'] = np.array(ds['time'].data, dtype='datetime64[M]')
    ds = ds.sel(time=weights['time'])


    # Null hypothesis
    enso_classes = get_enso_flavors_obs(
        definition='N3N4', fname=f_sst, vname='sst', climatology='month',
        month_range=[12,2]
    )

    time_snippets_null = []
    for y in np.array(enso_classes['end'], dtype='datetime64[Y]'):
        time_snippets_null.append([
           np.datetime64(f"{y}-{months[0]:02d}-01", 'D'), 
           np.datetime64(f"{y}-{months[1]+1:02d}-01", 'D') - 1, 
        ])
    

    composite_vars = []
    mask_vars = []
    for var in ds.data_vars:
        # Mean and std at each location for standardization 
        da = ds[var]

        # Null hypothesis
        da_null = preproc.select_time_snippets(da, time_snippets_null)

        # Get weighted composites and statistical significance
        composite_class_arr = []
        mask_class_arr = []
        classes = []
        for i, k in enumerate(weights['classes'].data):
            # Weighted mean
            # Use threshold on weights to crop X for a realistic sample size
            if threshold is not None:
                ids = np.where(weights.sel(classes=k).data >= threshold)[0]
                da_class = da.isel(time=ids)
                weight_class = weights.sel(classes=k).isel(time=ids)
                print(f"Num of datapoints c={k}: {len(da_class['time'])}")
            else:
                da_class = da
                weight_class = weights.sel(classes=k)

            # Weighted mean
            da_weighted = da_class.weighted(weight_class)
            mean = da_weighted.mean(dim='time')

            # Statistical test
            print(f"Test for statistical significance of c={k} using {stattest}!")
            if stattest == 'ks':
                _, pvals = utstats.kstest_field(da_class, da_null)
            elif stattest == 'ttest':
                _, pvals = utstats.ttest_field(
                    da_class, da_null, weights=weight_class, serial_data=serial_data
                )
            else:
                raise ValueError(f"Specified stattest={stattest} does not exist!")

            mask = utstats.field_significance_mask(pvals, alpha=alpha, corr_type='dunn')

            composite_class_arr.append(mean)
            mask_class_arr.append(mask)
            classes.append(k)
        
        composite_classes = xr.concat(
            composite_class_arr, dim=pd.Index(classes, name='classes')
        )
        composite_classes.name = var
        mask_classes = xr.concat(
            mask_class_arr, dim=pd.Index(classes, name='classes')
        )
        mask_classes.name = var
        
        composite_vars.append(composite_classes)
        mask_vars.append(mask_classes)

    composites = xr.merge(composite_vars)
    masks = xr.merge(mask_vars)

    return composites, masks


def get_mixed_composites(ds: xr.Dataset, f_sst: str, weights: xr.DataArray,
                         mix_classes: list=[4, 2], stattest: str='ks',
                        alpha: float=0.05, enso_months: list=[12,2],
                        serial_data: bool=False):
    """Create composites of mixed events, i.e. 0.3 < p(c=k) < 0.7, in both classes.

    # TODO: Remove because not needed
    Args:
        ds (xr.Dataset): Dataset to create composites from with dim=['lat', 'lon', 'time']
        f_sst (str): Path to SST dataset.
        weights (xr.DataArray): Conditional probabilities of dim=['classes', 'time]
        mix_classes (list, optional): Wich classes to use for mixed events. Defaults to [4, 2].
        threshold (float, optional): Threshold on weights. Defaults to None.
        stattest (str, optional): Statistical significance test. Defaults to 'ks'.
        alpha (float, optional): Alpha value for stat. significance. Defaults to 0.05.
        enso_months (list, optional): Months to compute null hypothesis over. Defaults to [12,2].
        serial_data (bool, optional): Use serial data correction. Defaults to False.
    """
    # Make sure dataset for composites are on the same time points as weights
    tmax = ds['time'].max() if ds['time'].max() < weights['time'].max() else weights['time'].max()
    tmin = ds['time'].min() if ds['time'].min() > weights['time'].min() else weights['time'].min()
    weights = weights.sortby(weights['time']) 
    weights = weights.sel(time=slice(tmin.data, tmax.data))
    ds = ds.sel(time=slice(tmin.data, tmax.data))

    # Null hypothesis
    enso_classes = get_enso_flavors_obs(
        definition='N3N4', fname=f_sst, vname='sst', climatology='month',
        month_range=enso_months
    )
#    ## Neutral ENSO winters
#    time_snippets = np.array(
#        [enso_classes.loc[enso_classes['type'] == 'Normal']['start'],
#         enso_classes.loc[enso_classes['type'] == 'Normal']['end']]
#    ).T
    ## All winters
    time_snippets = np.array(
        [enso_classes['start'],
         enso_classes['end']]
    ).T

    # Mix classes
    idx_mix = np.where(
        (weights.sel(classes=mix_classes[0]) > 0.3)
        & (weights.sel(classes=mix_classes[0]) < 0.7)
        & (weights.sel(classes=mix_classes[1]) > 0.3) 
        & (weights.sel(classes=mix_classes[1]) < 0.7)
    )[0]
    times_mix = weights['time'][idx_mix]
    mixed_event_timerange = []
    for year in np.array(times_mix, dtype='datetime64[Y]'):
        mixed_event_timerange.append([
           np.datetime64(f"{year}-{enso_months[0]:02d}-01", 'D'), 
           np.datetime64(f"{year+1}-{enso_months[1]+1:02d}-01", 'D') - 1, 
        ])
    print(f"Sample size mixed events: {len(times_mix)}")
    print(f"Sample size null events: {len(time_snippets)}")
    
    # Compute composites
    mixed_mean_arr = []
    mixed_mask_arr = []
    for var in ds.data_vars:
        # Mean and std at each location for standardization 
        da = ds[var]

        da_neutral = preproc.select_time_snippets(da, time_snippets)
        # Normalize arrays for statistical test
        # TODO: Remove std
        std = da.std(dim='time', skipna=True)
        da_neutral_norm = da_neutral #/ std

        # Get mixed composites and statistical significance
        da_mixed = preproc.select_time_snippets(da, mixed_event_timerange)

        # Stat. test
        print(f"Test for statistical significance of {mix_classes} using {stattest}!")
        # Normalize arrays for statistical test
        da_mixed_norm = da_mixed # /std
        if stattest == 'ks':
            _, pvals = utstats.kstest_field(da_mixed_norm, da_neutral_norm)
        elif stattest == 'ttest':
            _, pvals = utstats.ttest_field(
                da_mixed_norm, da_neutral_norm, serial_data=serial_data
            )
        else:
            raise ValueError(f"Specified stattest={stattest} does not exist!")

        # Mask
        mask = utstats.field_significance_mask(pvals, alpha=alpha, corr_type='dunn')
        mask.name = var
        # Mean
        mean = da_mixed.mean(dim='time', skipna=True)

        mixed_mean_arr.append(mean)
        mixed_mask_arr.append(mask)

    composites = xr.merge(mixed_mean_arr)
    masks = xr.merge(mixed_mask_arr)
    return composites, masks