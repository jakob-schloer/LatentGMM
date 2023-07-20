"""Plot functions for VAEs."""
import sys, os, string
import numpy as np
import xarray as xr
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import cartopy as ctp
import torch
from torch.utils.data import Dataset, DataLoader

from climvae.utils import preproc
from climvae.model import utils

PATH = os.path.dirname(os.path.abspath(__file__))
plt.style.use(PATH + "/../paper.mplstyle")


def loss(loss, n_batches=None, ax=None, yscale='linear', ylabel='mse', label='mse', **pargs):
    """Plot training and validation loss.

    Args:
    -----
    loss (np.array): (num_batches) Training loss of each batch.
    n_batches (int): Number of batches in the dataset.
    ax (plt.axes): Matplotlib axes, default: None
    y_scale (str): Y scale of plotting.

    Return:
    -------
    ax (plt.axes): Return axes.

    """
    # plot loss
    if ax is None:
        fig, ax = plt.subplots()

    xlabel = '# of batch'
    if n_batches is not None:
        loss = np.average(loss.reshape(-1, n_batches), axis=1)
        xlabel = 'epochs'

    ax.plot(loss, label=label+f" {loss[-1]:.2f}", **pargs)

    ax.set_xscale('linear')
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return ax


def plot_2dgaussian(mu, cov, p=0.5, ax=None, **kwargs):
    """Plot 2d gaussian at p-confidence level.

    Args:
        mu (np.array): (2) Mean.
        cov (np.array): (2,2) Covariance matrix.
        p (float, optional): Confidence. Defaults to 0.95.
        ax (plt.Axes, optional): Axes. Defaults to None.

    Returns:
        [type]: [description]
    """
    if ax is None:
        fig, ax = plt.subplots()

    s = -2 * np.log(1-p)
    v, w = np.linalg.eigh(cov*s)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(
        mu, v[0], v[1], 180 + angle, **kwargs
    )

#    ell.set_clip_box(ax.bbox)
    ax.add_patch(ell)
    ax.autoscale_view()

    return ax


def latent_encoding(model, dataset, loader,
                    idx=None, dim=None, ax=None, gmcolors=None,
                    **pltkwargs):
    """Plot points in latent space.

    Args:
        model (_type_): _description_
        dataset (_type_): _description_
        loader (_type_): _description_
        idx (_type_, optional): _description_. Defaults to None.
        dim (_type_, optional): _description_. Defaults to None.
        ax (_type_, optional): _description_. Defaults to None.
        gmcolors (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if idx is None:
        idx = np.arange(len(dataset))

    # Encoding
    encoding = utils.get_reconstructions(
        model, dataset, indices=idx)

    # PCA of encoding
    if model.z_dim > 2 and dim is None:
        pca_z = utils.latent_space_pca(model, loader)
        z_given_x = pca_z.transform(encoding['z'])
    else:
        z_given_x = encoding['z']
        pca_z = None

    if ax is None:
        fig, ax = plt.subplots()

    if dim is None:
        dim = [0, 1]

    ax.plot(z_given_x[:, dim[0]], z_given_x[:, dim[1]], '.', **pltkwargs)

    ax.set_xlabel(f'z_{dim[0]}')
    ax.set_ylabel(f'z_{dim[1]}')

    return ax, encoding, pca_z


def latent_GM(model, ax=None, dim=None, pca_z=None, gmcolors=None,
              labels=False, **kwargs):
    """Plot Gaussian mixture in latent space.

    Args:
        model (_type_): _description_
        ax (_type_, optional): _description_. Defaults to None.
        dim (_type_, optional): _description_. Defaults to None.
        pca_z (_type_, optional): _description_. Defaults to None.
        gmcolors (_type_, optional): _description_. Defaults to None.
        kwargs (dict): 

    Returns:
        _type_: _description_
    """

    # Plot GMM in latent space
    sigma = model.p_z_sigma.cpu().detach().numpy()
    mu = model.p_z_m.cpu().detach().numpy()

    if ax is None:
        fig, ax = plt.subplots()

    if dim is None:
        dim = [0, 1]

    for k in range(mu.shape[0]):
        cov = np.diag(np.exp(sigma[k, :]))
        m = mu[k, :]

        if pca_z is not None:
            # Linear projection of gaussian to pca components
            m = pca_z.components_ @ m
            cov = pca_z.components_ @ cov @ pca_z.components_.T

        m = m[dim]
        cov = np.array([[cov[dim[0], dim[0]], cov[dim[0], dim[1]]],
                        [cov[dim[1], dim[0]], cov[dim[1], dim[1]]]])

        if gmcolors is not None:
            kwargs['ec'] = gmcolors[k]
            kwargs['fc'] = gmcolors[k]
        for p in [0.9, 0.5, 0.1]:
            if (p == 0.1) and (labels is True):
                kwargs['label'] = f"c={k+1}"
            else:
                try:
                    del kwargs['label']
                except:
                    pass

            plot_2dgaussian(m, cov,
                            ax=ax, p=p, **kwargs)

    ax.set_xlabel(rf'$z_{dim[0]+1}$')
    ax.set_ylabel(rf'$z_{dim[1]+1}$')

    if 'label' in kwargs.keys():
        ax.legend()

    return ax


def latent_scatter(Z, y=None, ax=None, lookup=None,
                   colors=None, **kwargs):
    """Plot scatter in 2d.

    Args:
        Z (_type_): 2d points in latent space.
        y (_type_, optional): Label of points. Defaults to None.
        ax (_type_, optional): Defaults to None.
        lookup (_type_, optional): Lookup table for class. Defaults to None.

    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    if y is None:
        y = np.array([0] * Z.shape[0])

    # Plot pca encoding
    for i, n in enumerate(np.unique(y)):
        idx = np.where(y == n)[0]

        # Plot properties
        if lookup is None:
            kwargs['label'] = f'n={n}'
        else:
            kwargs['label'] = [k for k, v in lookup.items() if v == n][0]

        if colors is not None:
            kwargs['color'] = colors[n]

        ax.plot(Z[idx, 0], Z[idx, 1], '.', **kwargs)
    return ax


def plot_traverse_latent(vae_model, dataset, dirname, plot_func,
                         input_data=None, max_traverse=0.25,
                         num_steps=100, vmin=-1.0, vmax=1.0,
                         plot_traverse=True, **plot_args):
    """Plot traversing of the latent space.

    For each dimension a folder is created and for each step a figure is stored. The
    figures are enumerated accordinly and can be used to create a video using ffmpg.

    Args:
    -----
    vae_model: static.model
        Model of VAE
    dataset: preprocessing.utils.BaseDataset
        Dataset object.
    dirname: str
        Directory to store the created files in.
    plot_func: function
        Function plotting the map
    input_data: torch.dataset
        Input datapoint to traverse latent space around. If None, the latent space is
        traversed in each dimension while averaging all other dimensions. Default: None
    max_traverse: float
        Max traverse through.
    num_steps: int
        Number of steps for traverse.
    vmin: float
        Colorbar minimum
    vmax: float
        Colorbar maximum
    plot_traverse: bool
        If True the latent space traverse images are plotted and stored.
    **plot_args: dict
        Arguments passed to the plotting function.

    Return:
    -------
    std_traverse: list (features)
        Standard deviation of latent space traverse
    """
    # Create directory
    try:
        os.mkdir(dirname)
    except OSError:
        print("Creation of the directory %s failed" % dirname)
    else:
        print("Successfully created the directory %s " % dirname)

    latent_traverse_std = []  # store std of traverse of each dimension
    # Traverse prior ensemble
    feature_dim = np.arange(0, vae_model.latent_size, 1)
    for ld in feature_dim:
        print(f"Taverse direction {ld}")
        latent_samples = vae_model.traverse_latent_space(ld, max_traverse,
                                                         n_traversal=num_steps,
                                                         input_data=input_data)
        reconstructions = vae_model.decode_latents(latent_samples).numpy()

        # get std of traversion
        if len(reconstructions.shape) == 2:
            rec_std = np.std(reconstructions, axis=0)
            rec_std_im = dataset.get_map(rec_std, name=f"latent_dim_{ld}")
        else:
            shape = (reconstructions.shape[0],
                     reconstructions[0].flatten().shape[0])
            rec_std = np.std(reconstructions.reshape(shape), axis=0)
            rec_std_im = dataset.get_map_convolution(rec_std.reshape(
                reconstructions[0].shape), name=f"latent_dim_{ld}")  # convert array to map

        latent_traverse_std.append(rec_std_im)  # store map

        # save to file
        rec_std_im.to_dataset(name='latent_traverse').to_netcdf(
            dirname + f"/traverse_ld_{ld}_std.nc")

        # plot std
        fig = plt.figure()
        im = plot_func(rec_std_im, color='viridis', vmin=0, **plot_args)
        im['ax'].set_title(f'dim={ld}', size='xx-large')
        plt.savefig(dirname + f"/std_traverse_ld_{ld}")
        plt.close()

        if plot_traverse is True:
            # create folder
            dirname_ld = dirname + "/{:02d}_latent_dimension".format(ld)
            try:
                os.mkdir(dirname_ld)
            except OSError:
                print("Creation of the directory %s failed" % dirname_ld)
            else:
                print("Successfully created the directory %s " % dirname_ld)

            # plot single images
            for i in range(num_steps):
                fig = plt.figure()
                if len(reconstructions.shape) == 2:
                    rec = reconstructions[i, :]
                    # convert array to map
                    rec_im = dataset.get_map(rec, name='sst anomaly')
                else:
                    rec = reconstructions[i, :, :, :]
                    rec_im = dataset.get_map_convolution(
                        rec, name='sst anomaly')  # convert array to map

                im = plot_func(rec_im, vmin=vmin, vmax=vmax, **plot_args)
                im['ax'].set_title('{:.2f} latent'.format(latent_samples[i, ld]),
                                   size='xx-large')
                plt.savefig(dirname_ld + "/traverse_latent_{:03d}".format(i))
                plt.close("all")

    return latent_traverse_std


def get_2d_kde_grad(data, bw='scott'):
    """Get gaussian kde approximation and gradient of 2d input.

    Args:
        data (np.ndarray): Input data (2, n_samples)

    Returns:
        x (np.ndarray):
        y (np.ndarray):
        density (np.ndarray):
        gradient (list):
    """
    x = np.linspace(data[0, :].min(), data[0, :].max(), 100)
    y = np.linspace(data[1, :].min(), data[1, :].max(), 100)
    XX, YY = np.meshgrid(x, y)
    positions = np.vstack([XX.ravel(), YY.ravel()])

    kernel = stats.gaussian_kde(data, bw_method=bw)
    density = np.reshape(kernel(positions), XX.shape)
    grad = np.gradient(density)

    return x, y, density, grad


def latent_density(latent_x, latent_y, type='density', bw='scott',
                   ax=None, color='RdBu_r', cbar=False, **kwargs):
    """Plot latent density.

    Args:
        latent_x ([type]): [description]
        latent_y ([type]): [description]
        type (str): Plot type 'density, grad_x, grad_y, grad'
        ax ([type], optional): [description]. Defaults to None.
        bins (list, optional): [description]. Defaults to [50, 50].
        color (str, optional): [description]. Defaults to 'Blues'.

    Returns:
        [type]: [description]
    """
    if ax is None:
        fig, ax = plt.subplots()
    # plot density
    x, y, density, grad = get_2d_kde_grad(
        np.array([latent_x, latent_y]), bw=bw)

    if type == 'density':
        #        im = ax.contourf(x, y, density, cmap=color, **kwargs)
        im = sns.kdeplot(x=latent_x, y=latent_y, ax=ax, cmap=color, **kwargs)
    elif type == 'grad_x':
        im = ax.contourf(x, y, grad[0], cmap=color, **kwargs)
    elif type == 'grad_y':
        im = ax.contourf(x, y, grad[1], cmap=color, **kwargs)
    elif type == 'grad':
        grad_norm = np.linalg.norm(np.array(grad), axis=0)
        im = ax.contourf(x, y, grad_norm, cmap=color, **kwargs)
    else:
        raise ValueError(
            f"Type {type} is not defined. Choose either 'density, grad_x, grad_y, grad'")

    if cbar:
        plt.colorbar(im, ax=ax, orientation='vertical')

    return ax


def latent_distribution_matrix(latent_sample, plot_type='density', axs_dic=None,
                               **plot_args):
    """Plot matrix of latent density.

    Args:
        latent_sample ([type]): [description]
        plot_type (str, optional): [description]. Defaults to 'density'.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    n_latent_dim = latent_sample.shape[1]
    # Initialize subplots
    if axs_dic is None:
        fig = plt.figure(figsize=(n_latent_dim*3, n_latent_dim*2.5))
        axs_dic = []

        for i in range(n_latent_dim):
            for j in range(i+1, n_latent_dim):
                count = i*n_latent_dim + j
                ax = fig.add_subplot(n_latent_dim, n_latent_dim, count)
                axs_dic.append(
                    {'ax': ax, 'dim_x': j, 'dim_y': i, 'fig': fig}
                )

    for sbp in axs_dic:
        dim_x = sbp['dim_x']
        dim_y = sbp['dim_y']
        ax = sbp['ax']
        if plot_type == 'scatter':
            ax.scatter(latent_sample[:, dim_x], latent_sample[:, dim_y],
                       **plot_args)
        else:
            latent_density(latent_sample[:, dim_x], latent_sample[:, dim_y],
                           type=plot_type, ax=ax, **plot_args)

        ax.set_xlabel(f"latent {dim_x +1}")
        ax.set_ylabel(f"latent {dim_y +1}")

    return axs_dic


def matrix(mat_corr, pick_x=None, pick_y=None,
           label_x=None, label_y=None,
           ax=None, color='BrBG',
           vmin=-1, vmax=1,
           cbar_kw=dict(extend='both', orientation='horizontal',
                        label='correlation')
           ):
    """Plot correlation matrix.

    Args:
        mat_corr ([type]): [description]
        pick_x ([type], optional): [description]. Defaults to None.
        pick_y ([type], optional): [description]. Defaults to None.
        label_x ([type], optional): [description]. Defaults to None.
        label_y ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        vmin (int, optional): [description]. Defaults to -1.
        vmax (int, optional): [description]. Defaults to 1.
        color (str, optional): [description]. Defaults to 'BrBG'.
        cbar_kw (dict, optional): Arguments to plt.colorbar. 

    Returns:
        im (plt.imshow): [description]
    """
    if ax is None:
        fig, ax = plt.subplots()

    if pick_y is not None and pick_x is not None:
        corr = mat_corr[pick_x, :].copy()
        corr = corr[:, pick_y]
    elif pick_x is not None:
        corr = mat_corr[pick_x, :]
    elif pick_y is not None:
        corr = mat_corr[:, pick_y]
    else:
        corr = mat_corr

    cmap = plt.get_cmap(color)
    im = ax.imshow(corr, vmin=vmin, vmax=vmax, aspect='auto', cmap=cmap)

    cbar = plt.colorbar(im, ax=ax, **cbar_kw)

    if label_x is not None:
        ax.set_xticks(np.arange(0, len(label_x)))
        ax.set_xticklabels(label_x)
    if label_y is not None:
        ax.set_yticks(np.arange(0, len(label_y)))
        ax.set_yticklabels(label_y)

    return im


def gmm_sklearn(gmm, X, y=None, pca=None, ax=None, y_lookup=None,
                colors=None, **kwargs):
    """Plot data and gmm in 2d.

    Args:
        gmm (_type_): _description_
        X (_type_): _description_
        y (_type_, optional): _description_. Defaults to None.
        pca (_type_, optional): _description_. Defaults to None.
        ax (_type_, optional): _description_. Defaults to None.
        y_lookup (_type_, optional): Lookup table for class. Defaults to None.

    Returns:
        _type_: _description_
    """
    if ax is None:
        fig, ax = plt.subplots()

    if y is None:
        y = np.array([0] * X.shape[0])

    # Plot pca encoding
    for i, n in enumerate(np.unique(y)):
        idx = np.where(y == n)[0]

        # Plot properties
        pltargs = dict()
        if y_lookup is None:
            pltargs['label'] = f'n={n}'
        else:
            pltargs['label'] = [k for k, v in y_lookup.items() if v == n][0]

        if colors is not None:
            pltargs['color'] = colors[i]

        ax.plot(X[idx, 0], X[idx, 1], '.', **pltargs)

    # Plot GMM
    for k in range(gmm.means_.shape[0]):
        mean = gmm.means_[k, :]
        if gmm.covariance_type == "full":
            Sigma = gmm.covariances_[k, :]
        elif gmm.covariance_type == "diag":
            Sigma = np.diag(gmm.covariances_[k, :])

        if pca is not None:
            # Linear projection of gaussian to pca components
            mean = pca.components_ @ mean
            Sigma = pca.components_ @ Sigma @ pca.components_.T

        plot_2dgaussian(mean, Sigma,
                        ax=ax, p=0.5, label=f"k={k}", **kwargs)
    return ax

def enumerate_subplots(axs, pos_x=-0.07, pos_y=1.04, fontsize=None):
    """Adds letters to subplots of a figure.

    Args:
        axs (list): List of plt.axes.
        pos_x (float, optional): x position of label. Defaults to 0.02.
        pos_y (float, optional): y position of label. Defaults to 0.85.
        fontsize (int, optional): Defaults to 18.

    Returns:
        axs (list): List of plt.axes.
    """
    axs = np.array(axs)
    if type(pos_x) == float:
        pos_x = [pos_x] * len(axs.flatten())
    if type(pos_y) == float:
        pos_y = [pos_y] * len(axs.flatten())

    for n, ax in enumerate(axs.flatten()):
        ax.text(
            pos_x[n],
            pos_y[n],
            f"{string.ascii_uppercase[n]}" if n < 26 else f"{string.ascii_uppercase[n-26]}{string.ascii_uppercase[n-26]}.",
            transform=ax.transAxes,
            size=fontsize,
            weight="bold",
        )
    return axs







######################################################
# Functions for SSTA only
######################################################
def create_map_plot(ax=None, ctp_projection='PlateCarrree',
                    central_longitude=0,
                    gridlines_kw=dict(
                        draw_labels=True, dms=True, x_inline=False, y_inline=False
                    )):
    """Generate cartopy figure for plotting.

    Args:
        ax ([type], optional): [description]. Defaults to None.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarrree'.
        central_longitude (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns:
        ax (plt.axes): Matplotplib axes object.
    """
    if ax is None:
        # set projection
        if ctp_projection == 'Mollweide':
            proj = ctp.crs.Mollweide(central_longitude=central_longitude)
        elif ctp_projection == 'EqualEarth':
            proj = ctp.crs.EqualEarth(central_longitude=central_longitude)
        elif ctp_projection == 'Robinson':
            proj = ctp.crs.Robinson(central_longitude=central_longitude)
        elif ctp_projection == 'PlateCarree':
            proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
        else:
            raise ValueError(
                f'This projection {ctp_projection} is not available yet!')

        fig, ax = plt.subplots()
        ax = plt.axes(projection=proj)

    # axes properties
    ax.coastlines()
    if gridlines_kw is not None:
        gl = ax.gridlines(**gridlines_kw)
    else:
        gl = None
    ax.add_feature(ctp.feature.RIVERS)
    ax.add_feature(ctp.feature.BORDERS, linestyle=':')

    return ax, gl


def round_axes(ax=None):
    """Returns an matplotlib axes with round boundaries.

    Args:
        ax (mpl.axes, optional): Defaults to None.

    Returns:
        ax (mpl.axes, optional): Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpl.path.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    return ax

def plot_map(dmap, ax=None, vmin=None, vmax=None, step=0.1,   
             cmap='RdBu_r', centercolor=None, bar='discrete', add_bar=True, 
             ctp_projection='PlateCarree', transform=None, central_longitude=0,
             kwargs_pl=None,
             kwargs_cb=dict(orientation='horizontal', shrink=0.8, extend='both'),
             kwargs_gl=dict(auto_inline=False, draw_labels=True, dms=True)
             ):
    """Simple map plotting using xArray.

    Args:
        dmap ([type]): [description]
        central_longitude (int, optional): [description]. Defaults to 0.
        vmin ([type], optional): [description]. Defaults to None.
        vmax ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        fig ([type], optional): [description]. Defaults to None.
        color (str, optional): [description]. Defaults to 'RdBu_r'.
        bar (bool, optional): [description]. Defaults to True.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarree'.
        label ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if kwargs_pl is None:
        kwargs_pl = dict()

    # create figure
    ax, gl = create_map_plot(ax=ax, ctp_projection=ctp_projection,
                         central_longitude=central_longitude,
                         gridlines_kw=kwargs_gl)


    # choose symmetric vmin and vmax
    if vmin is None and vmax is None:
         vmin = dmap.min(skipna=True)
         vmax = dmap.max(skipna=True)
         vmax = vmax if vmax > (-1*vmin) else (-1*vmin)
         vmin = -1*vmax

    # Select colormap
    if bar == 'continuous':
        cmap = plt.get_cmap(cmap)
        kwargs_pl['vmin'] = vmin 
        kwargs_pl['vmax'] = vmax
    elif bar == 'discrete':
        if 'norm' not in kwargs_pl:
            step = (vmax-vmin)/10 if step is None else step
            bounds = np.arange(vmin, vmax+step-1e-5, step)
            # Create colormap
            n_colors = len(bounds)+1
            cmap = plt.get_cmap(cmap, n_colors)
            colors = np.array([mpl.colors.rgb2hex(cmap(i)) for i in range(n_colors)])
            # Set center of colormap to specific color
            if centercolor is not None:
                idx = [len(colors) // 2 - 1, len(colors) // 2]
                colors[idx] = centercolor 
            cmap = mpl.colors.ListedColormap(colors)
            kwargs_pl['norm'] = mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend='both')
        else:
            cmap = plt.get_cmap(cmap)
    else:
        raise ValueError(f"Specified bar={bar} is not defined!")

    if transform is None:
        transform = ctp.crs.PlateCarree(central_longitude=central_longitude)

    # plot map
    im = ax.pcolormesh(
        dmap.coords['lon'], dmap.coords['lat'], dmap.data,
        cmap=cmap, transform=transform,
        **kwargs_pl
    )

    # set colorbar
    if add_bar:
        if 'label' not in list(kwargs_cb.keys()):
            kwargs_cb['label'] = dmap.name
        cbar = plt.colorbar(im, ax=ax, **kwargs_cb)
    else:
        cbar = None

    return {'ax': ax, "im": im, 'gl': gl, 'cb': cbar}



def plot_vector_field(
    u,
    v,
    ax=None,
    lons=None,
    lats=None,
    key_loc=(0.95, -0.06),
    central_longitude=0,
    stream=False,
    ctp_projection='PlateCarree',
    gridlines_kw=dict(auto_inline=False, draw_labels=True, dms=True),
    **kwargs,
):

    ax = create_map_plot(ax=ax, ctp_projection=ctp_projection,
                         central_longitude=central_longitude,
                         gridlines_kw=gridlines_kw)
    if lons is None:
        lons = u.coords["lon"]
    if lats is None:
        lats = u.coords["lat"]
    steps = kwargs.pop('steps', 1)
    latsteps = lonsteps = steps

    u_dat = u.data[::latsteps, ::lonsteps]
    v_dat = v.data[::latsteps, ::lonsteps]

    lw = kwargs.pop("lw", 1)
    scale = kwargs.pop("scale", None)
    # headwidth = kwargs.pop('headwidth', 1)
    # width = kwargs.pop('width', 0.005)

    if stream:
        magnitude = (u ** 2 + v ** 2) ** 0.5
        im_stream = ax.streamplot(
            lons[::lonsteps],
            lats[::latsteps],
            u_dat,
            v_dat,
            transform=ctp.crs.PlateCarree(central_longitude=central_longitude),
            linewidths=lw,
            color=magnitude
        )
    else:
        pivot = kwargs.pop("pivot", "middle")
        tf = kwargs.pop("transform", True)

        if tf:
            Q = ax.quiver(
                lons[::lonsteps],
                lats[::latsteps],
                u_dat,
                v_dat,
                pivot=pivot,
                transform=ctp.crs.PlateCarree(central_longitude=central_longitude),
                linewidths=lw,
                scale=scale,
            )
        else:
            Q = ax.quiver(
                lons[::lonsteps],
                lats[::latsteps],
                u_dat,
                v_dat,
                pivot=pivot,
                linewidths=lw,
                scale=scale,
            )
        key_length = kwargs.pop('key_length', 1)
        wind_unit = kwargs.pop('wind_unit', r"$\frac{m}{s}$")
        if key_loc:
            ax.quiverkey(
                Q,
                key_loc[0],
                key_loc[1],
                key_length,
                f"{key_length} {wind_unit}",
                labelpos="W",
                coordinates="axes",
            )
    return {'ax': ax }


def significance_mask(mask, ax=None, ctp_projection='PlateCarree', hatch='..',
                      transform=None, central_longitude=0):
    """Plot significante areas using the mask map with True for significant
    and False for non-significant.

    Args:
        mask (xr.Dataarray): Significance mask. 
        ax (plt.Axes, optional): Axes object. Defaults to None.
        ctp_projection (str, optional): Cartopy projection. Defaults to 'PlateCarree'.
        transform (cartopy.Transform, optional): Transform object.
            Defaults to None meaning 'PlateCarree'.
        central_longitude (int, optional): Central longitude. Defaults to 0.

    Returns:
        {'ax': ax, "im": im} (dict): Dictionary with Axes and plot object.
    """
    if ax is None:
        ax = create_map_plot(ax=ax, ctp_projection=ctp_projection,
                             central_longitude=central_longitude)
    if transform is None:
        transform = ctp.crs.PlateCarree(central_longitude=central_longitude)

    # Convert True/False map into 1/NaN
    msk = xr.where(mask == True, 1, np.nan)
    im = ax.pcolor(
        msk['lon'],
        msk['lat'],
        msk.data,
        hatch=hatch,
        alpha=0.0,
        transform=transform,
    )

    return {'ax': ax, "im": im}


def highlight_box(ax, lon_range, lat_range, **kwargs):
    """Plots a rectangle on a cartopy map

    Args:
        ax (geoaxis): Axis of cartopy object
        lon_range (list): list of min and max longitude
        lat_range (list): list of min and max lat

    Returns:
        geoaxis: axis with rectangle plotted
    """
    from shapely.geometry.polygon import LinearRing

    shortest = kwargs.pop("shortest", True)
    if (max(lon_range) - min(lon_range) <= 180) or shortest is False:
        cl = 0
        lons = [max(lon_range), min(lon_range), min(lon_range), max(lon_range)]
    else:
        cl = 180
        lons = [
            max(lon_range) - 180,
            180 + min(lon_range),
            180 + min(lon_range),
            max(lon_range) - 180,
        ]
    lats = [min(lat_range), min(lat_range), max(lat_range), max(lat_range)]

    ring = LinearRing(list(zip(lons, lats)))
    lw = kwargs.pop("lw", 1)
    color = kwargs.pop("color", "k")
    fill = kwargs.pop("fill", False)
    facecolor = color if fill else "none"
    zorder = kwargs.pop('zorder', 11)
    ax.add_geometries(
        [ring],
        ctp.crs.PlateCarree(central_longitude=cl),
        facecolor=facecolor,
        edgecolor=color,
        linewidth=lw,
        zorder=zorder,
    )

    return ax


def plot_hovmoeller(da, x='lon', y='time', ax=None, vmin=None, vmax=None, step=None,
                    cmap='RdBu_r', centercolor=None, add_bar=True,
                    plkwargs=dict(), cbkwargs=dict()):
    if ax == None:
        fig, ax = plt.subplots(figsize=(7, 3))

    # choose symmetric vmin and vmax
    if vmin is None or vmax is None:
         vmin = da.min(skipna=True)
         vmax = da.max(skipna=True)
         vmax = vmax if vmax > (-1*vmin) else (-1*vmin)
         vmin = -1*vmax
         step = (vmax-vmin)/20 if step is None else step 
    
    # Levels for contours
    levels = np.arange(vmin, vmax+step-1e-5, step)

    # Create the new colormap
    n_colors = len(levels)+1
    cmap = plt.get_cmap(cmap, n_colors)
    # Discretize colormap
    cmap_list = [cmap(i) for i in range(cmap.N)]
    # Set center to centercolor
    if centercolor is not None:
        cmap_list[len(cmap_list) // 2 - 1] = centercolor 
        cmap_list[len(cmap_list) // 2] = centercolor 
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmap_list, cmap.N)

    im = ax.contourf(da[x], da[y], da, 
                     levels=levels, cmap=cmap, extend='both', **plkwargs) 

    # set colorbar
    if add_bar:
        if 'label' not in list(cbkwargs.keys()):
            cbkwargs['label'] = da.name
        cbar = plt.colorbar(im, **cbkwargs)
    else:
        cbar = None

    return {'ax': ax, "im": im, 'cb': cbar}


def ssta_reconstructions(dataset, x, x_hat, varname=None, label=None, axs=None,
                         normalizer=None, **kwargs):
    """Plot input and reconstruction of SST-pacific data.

    Args:
        dataset (_type_): SpatialData dataset object for get_map 
        x (_type_): _description_
        x_hat (_type_): _description_
        label (_type_, optional): _description_. Defaults to None.
        axs (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_plots = x.shape[0]
    if axs is None:
        fig, axs = plt.subplots(2, n_plots, figsize=(n_plots*4, 7),
                                subplot_kw={
            'projection': ctp.crs.Mollweide(central_longitude=180)
        })
    else:
        assert axs.shape[0] == 2
        assert axs.shape[1] == n_plots

    for i in range(n_plots):
        input_im = dataset.get_map(x[i])
        rec_im = dataset.get_map(x_hat[i])

        # For multidim input/output
        if varname is not None:
            input_im = input_im[varname]
            rec_im = rec_im[varname]

        # plot input
        if ('vmax' not in kwargs) & ('vmin' not in kwargs):
            kwargs['vmax'] = np.nanmax(input_im.data) if np.nanmax(input_im.data) > (
                1-np.nanmin(input_im.data)) else (1-np.nanmin(input_im.data))
            kwargs['vmin'] = -1*kwargs['vmax']

        plot_map(input_im, central_longitude=180,
                 color='RdBu_r', ax=axs[0, i], label=f'{input_im.name} obs.', **kwargs)

        # plot reconstruction
        plot_map(rec_im, central_longitude=180,
                 color='RdBu_r', ax=axs[1, i], label=f'{rec_im.name} rec.', **kwargs)

        if label is not None:
            axs[0, i].set_title(
                np.datetime_as_string(
                    preproc.timestamp2time(label[i]['time']),
                    unit='D')
            )
    return axs


def ssta_latent_encoding(z, time=None, z_dim=[0, 1], time_slices=None,
                         ax=None, axlabel='z', **pltkwargs):
    """Plot latent encodings of ssta.

    Args:
        z (np.ndarray): (samples, z_dim) Latent encodings z. 
        time (list, optional): (samples) Times corresponding to the encodings.
                                Defaults to None.
        z_dim (list, optional): Dimensions to plot of z. Defaults to [0, 1].
        time_slices (list, optional): If only part of the encoding should be plotted.
            E.g. only El Nino years. Defaults to None.
        ax (plt.Axes, optional): Matplotlib axes. Defaults to None.
        **pltkwargs (dict, optional): Kwargs for matplotlib.plot function.

    Returns:
        ax (plt.Axes): Matplotlib axes.
    """
    if time is None:
        time = np.arange(z.shape[0])
    da = xr.DataArray(data=z, dims=['time', 'dim'],
                      coords={
        'time': time,
        'dim': np.arange(0, z.shape[-1], 1)
    })

    if time_slices is not None:
        for i, ts in enumerate(time_slices):
            if i == 0:
                buff = da.sel(time=slice(ts[0], ts[1]))
            else:
                buff = xr.concat(
                    [buff, da.sel(time=slice(ts[0], ts[1]))], dim='time')
    else:
        buff = da

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(buff.sel(dim=z_dim[0]), buff.sel(dim=z_dim[1]), '.', **pltkwargs)

    ax.set_xlabel(rf'${axlabel}_{z_dim[0]+1}$')
    ax.set_ylabel(rf'${axlabel}_{z_dim[1]+1}$')

    if 'label' in pltkwargs.keys():
        ax.legend()

    return ax


def plot_traverse_marginal_posterior(vae_model, dataset, dirname,
                                     num_samples=1000, num_steps=100,
                                     plot_func=None, create_latents=False,
                                     **plot_args):
    """Traverse latent space in the range of the marginal posterior, i.e.

    \sum_x \sum_{j \neq i} p(z_j|x)

    For each dimension a folder is created and for each step a figure is stored. The
    figures are enumerated accordinly and can be used to create a video using ffmpg.

    Args:
    -----
    vae_model: static.model
        Model of VAE
    dataset: preprocessing.utils.BaseDataset
        Dataset object.
    dirname: str
        Directory to store the created files in.
    num_samples: int
        Number of samples to average over. Default: 1000
    num_steps: int
        Number of steps between sigma. Default: 100
    plot_func: function
        Function plotting the map frames of the traverse. If set to None, the maps will 
        not be plotted and stored. Default: None
    create_latents: bool
        Specifies wether latent values of the given dataset are taken or all latent values
        which have been used with the model, i.e. vae.latent_space array.
    **plot_args: dict
        Arguments passed to the plotting function.

    Return:
    -------
    std_traverse: list (features)
        Standard deviation of latent space traverse
    """
    # get limits of p(z|x) by encoding all datapoints
    if create_latents is True:
        dataset_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        z, logvar_z, rec_x = vae_model.get_ordered_latent_space(dataset_loader)
    else:
        z = vae_model.get_latent_space()

    min_z = np.min(z, axis=0)
    max_z = np.max(z, axis=0)

    # Create directory
    try:
        os.mkdir(dirname)
        print("Successfully created the directory %s " % dirname)
    except OSError:
        print("Creation of the directory %s failed" % dirname)

    latent_traverse_std = []  # store std of traverse of each dimension
    # Traverse latent space ensemble
    feature_dim = np.arange(0, vae_model.latent_size, 1)
    for ld in feature_dim:
        print(f"Taverse direction {ld}")
        print(f"From {min_z[ld]} to {max_z[ld]}")
        eps_traverse = np.linspace(min_z[ld], max_z[ld], num_steps)
        # create ensemble
        ensemble = []
        for i in range(num_samples):
            rec = vae_model.traverse_prior(ld, eps_traverse)
            rec = rec.to('cpu').detach().numpy()
            rec_shape = rec.shape  # if convolutional output
            rec_flat = rec.flatten()
            ensemble.append(rec_flat)

        # mean and std of ensemble
        ens_mean = np.mean(ensemble, axis=0)
        ens_std = np.std(ensemble, axis=0)

        reconstructions = ens_mean.reshape(rec_shape)

        # get std of mean traverse
        if len(reconstructions.shape) == 2:
            rec_std = np.std(reconstructions, axis=0)
            rec_std_im = dataset.get_map(rec_std, name=f"latent_dim_{ld}")
        else:
            shape = (reconstructions.shape[0],
                     reconstructions[0].flatten().shape[0])
            rec_std = np.std(reconstructions.reshape(shape), axis=0)
            rec_std_im = dataset.get_map_convolution(rec_std.reshape(
                reconstructions[0].shape), name=f"latent_dim_{ld}")  # convert array to map

        latent_traverse_std.append(rec_std_im)  # store map

        # save to file
        rec_std_im.to_dataset(name='latent_traverse').to_netcdf(
            dirname + f"/traverse_ld_{ld}_std.nc")

        # Plot frames of latent traverse
        if plot_func is not None:
            # create folder
            dirname_ld = dirname + "/{:02d}_latent_dimension".format(ld)
            try:
                os.mkdir(dirname_ld)
                print("Successfully created the directory %s " % dirname_ld)
            except OSError:
                print("Creation of the directory %s failed" % dirname_ld)

            # plot single images
            for i in range(num_steps):
                fig = plt.figure()
                if len(reconstructions.shape) == 2:
                    rec = reconstructions[i, :]
                    # convert array to map
                    rec_im = dataset.get_map(rec, name='sst anomaly')
                else:
                    rec = reconstructions[i, :, :, :]
                    rec_im = dataset.get_map_convolution(
                        rec, name='sst anomaly')  # convert array to map

                im = plot_func(rec_im, **plot_args)
                im['ax'].set_title('{:.2f} sigma'.format(
                    eps_traverse[i]), size='xx-large')
                plt.savefig(dirname_ld + "/traverse_latent_{:03d}".format(i))

                plt.close(fig)

    return latent_traverse_std


######################
# Sequential VAE
######################
def animate_reconstruction(input_map, rec_map,
                           vmin=-3, vmax=3,
                           fps=5, filename=None):
    """Animate input, reconstruction and difference of sequence of maps.

    Args:
        input_map (xr.DataArray): Input of maps.
        rec_map (xr.DataArray): Reconstruction of maps.
        fig (plt.figure, optional): Figure. Defaults to None.
        filename (str, optional): Filename to store animation. Defaults to None.
        fps (int, optional): Frames per second.. Defaults to 10.

    Returns:
        ani: Animation if filename is not given otherwise None 
    """

    # residuals
    res_map = input_map - rec_map

    # Create figures
    proj = ctp.crs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(1, 3,
                            subplot_kw=dict(projection=proj))

    def init():
        # Initialize animation
        print("Initialize animation")
        for ax in axs:
            ax.clear()
        im = plot_map(input_map[0], central_longitude=180,
                      vmin=vmin, vmax=vmax, ax=axs[0],
                      label="SpatialData observation",
                      bar=True)
        plot_map(rec_map[0], central_longitude=180,
                 vmin=vmin, vmax=vmax, ax=axs[1],
                 label="SpatialData reconstruction",
                 bar=True)
        plot_map(res_map[0], central_longitude=180,
                 vmin=vmin, vmax=vmax, ax=axs[2],
                 label="residuals",
                 bar=True)
        axs[0].set_title(str(np.array(input_map[0]['time'], 'datetime64[D]')))
        axs[1].set_title(str(np.array(rec_map[0]['time'], 'datetime64[D]')))
        axs[2].set_title(str(np.array(res_map[0]['time'], 'datetime64[D]')))

    def plot_frame(i):
        # Run animation
        print(f"Plot frame: {i}")
        plot_map(input_map[i], central_longitude=180,
                 vmin=vmin, vmax=vmax, ax=axs[0],
                 bar=False)
        plot_map(rec_map[i], central_longitude=180,
                 vmin=vmin, vmax=vmax, ax=axs[1],
                 bar=False)
        plot_map(res_map[i], central_longitude=180,
                 vmin=vmin, vmax=vmax, ax=axs[2],
                 bar=False)

        axs[0].set_title(str(np.array(input_map[i]['time'], 'datetime64[D]')))
        axs[1].set_title(str(np.array(rec_map[i]['time'], 'datetime64[D]')))
        axs[2].set_title(str(np.array(res_map[i]['time'], 'datetime64[D]')))

    iterations = np.arange(1, len(input_map['time']))
    ani = animation.FuncAnimation(fig, plot_frame, init_func=init,
                                  frames=iterations)

    if filename is None:
        return
    else:
        # Save animation to file
        ani.save(filename, writer='imagemagick', fps=fps)
#        ani.save(filename, writer='ffmpeg', fps=fps, bitrate=1000) # MP4
        plt.close()
        return


def map_sequence(maps, dirname=None, prefix=None, central_longitude=0,
                 show=False,
                 **kwargs):
    """Plot sequence of maps in single images.

    Args:
        input_map (xr.DataArray): Input of maps.
        dirname (str, optional): Foldername to store plots. Defaults to None.
        prefix (str, optional): Filename prefix. Defaults to None.
        vmin (float, optional): Defaults to -3. 
        vmax (float, optional): Defaults to 3. 
        central_longitude (int, optional): Defaults to 0.
        **kwargs (dict, optional): Kwargs of plot_map function.

    """
    # Create directory
    try:
        os.mkdir(dirname)
        print("Successfully created the directory %s " % dirname)
    except OSError:
        print("Creation of the directory %s failed" % dirname)

    for i, time in enumerate(maps['time']):
        # Create figures
        print(f"Plot fig. {i}: {str(np.array(time, 'datetime64[D]'))}")
        proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
        fig, ax = plt.subplots(1, 1,
                               subplot_kw=dict(projection=proj))
        # Input map
        plot_map(maps[i], central_longitude=central_longitude, ax=ax,
                 label=maps.name,
                 **kwargs)
        # Title
        ax.set_title(str(np.array(maps[i]['time'], 'datetime64[D]')))

        plt.tight_layout()
        plt.savefig(dirname + "/" + prefix + "_{0:03d}.png".format(i))

        if not show:
            plt.close()

    return


def sequence_reconstruction(input_map, rec_map,
                            dirname=None, prefix=None,
                            vmin=-3, vmax=3, central_longitude=0,
                            **kwargs):
    """Plot input, reconstruction and difference of sequence of maps in single images.

    Args:
        input_map (xr.DataArray): Input of maps.
        rec_map (xr.DataArray): Reconstruction of maps.
        dirname (str, optional): Foldername to store plots. Defaults to None.
        prefix (str, optional): Filename prefix. Defaults to None.
        vmin (float, optional): Defaults to -3. 
        vmax (float, optional): Defaults to 3. 
        central_longitude (int, optional): Defaults to 0.
        **kwargs (dict, optional): Kwargs of plot_map function.

    """
    # Create directory
    try:
        os.mkdir(dirname)
        print("Successfully created the directory %s " % dirname)
    except OSError:
        print("Creation of the directory %s failed" % dirname)

    # residuals
    res_map = input_map - rec_map

    for i, time in enumerate(input_map['time']):
        # Create figures
        print(f"Plot fig. {i}: {str(np.array(time, 'datetime64[D]'))}")
        proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
        fig, axs = plt.subplots(1, 3,
                                subplot_kw=dict(projection=proj))
        # Input map
        plot_map(input_map[i], central_longitude=central_longitude,
                 vmin=vmin, vmax=vmax, ax=axs[0],
                 label="SpatialData observation",
                 **kwargs)
        # Reconstruction map
        plot_map(rec_map[i], central_longitude=central_longitude,
                 vmin=vmin, vmax=vmax, ax=axs[1],
                 label="SpatialData reconstruction",
                 **kwargs)
        # Difference
        plot_map(res_map[i], central_longitude=180,
                 vmin=vmin, vmax=vmax, ax=axs[2],
                 label="residuals",
                 **kwargs)
        # Title
        axs[0].set_title(str(np.array(input_map[i]['time'], 'datetime64[D]')))
        axs[1].set_title(str(np.array(rec_map[i]['time'], 'datetime64[D]')))
        axs[2].set_title(str(np.array(res_map[i]['time'], 'datetime64[D]')))

        plt.tight_layout()
        plt.savefig(dirname + "/" + prefix + "_{0:03d}.png".format(i))
        plt.close()

    return
