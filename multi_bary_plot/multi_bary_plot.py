import numpy as np
import pandas as pd
from multiprocess import Pool
from tqdm import tqdm, tqdm_notebook
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

class multi_bary_plot:
    """

    This class can turn n-dimensional data into a
    2-d plot with barycentric coordinates.

    Parameters
    ----------
    data : pandas.DataFrame
        A column for the values and n columns for the n classes.
    value_column : string
        The name of the value coumn in `data`.
    res : int
        The number of pixel along one axis.
    notebook : bool
        If True the progress bar will be an IPython/Jupyter Notebook widget.
        
    Returns
    -------
    multi_bary_plot : instance
        An instance of the multi_bary_plot.
    
    Usage
    -----
    vec = list(range(100))
    pdat = pd.DataFrame({'class 1':vec,
                         'class 2':list(reversed(vec)),
                         'class 3':[50]*100,
                         'val':vec})
    bp = multi_bary_plot(pdat, 'val')
    fig, ax, im = bp.plot()
    """

    def __init__(self, data, value_column, res=500,
                 notebook=False):
        if value_column not in data.columns.values:
            raise ValueError('`value_column` musste be a coumn name of `data`.')
        if not isinstance(res, (int, float)):
            raise ValueError('`res` musst be numerical.')
        numerical = ['float64', 'float32', 'int64', 'int32']
        if not all([d in numerical for d in data.dtypes]):
            raise ValueError('The data needs to be numerical.')
        if not isinstance(notebook, bool):
            raise ValueError('`notebook` musst be boolean.')
        self.res = int(res)
        coords = data.drop([value_column], axis=1)
        norm = np.sum(coords.values, axis=1, keepdims=True)
        self.coords = coords.values / norm
        self.vertNames = list(coords.columns.values)
        self.values = data[value_column]
        self.nverts = self.coords.shape[1]
        self.colorbar_pad = .1
        if self.nverts < 3:
            raise ValueError('At least three dimensions are needed.')
        if self.nverts == 3:
            self.colorbar_pad = -.5
        elif (self.nverts & 1) == 0 or self.nverts > 5:
            self.colorbar_pad = .3
        if notebook:
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = tqdm
    
    @property
    def grid(self):
        """The grid of pixels to raster."""
        x = np.linspace(-1, 1, self.res)
        return np.array(np.meshgrid(x, -x))
    
    @property
    def fgrid(self):
        """Melted x and y coordinates of the pixel grid."""
        grid = self.grid
        return grid.reshape((grid.shape[0],
                             grid.shape[1]*grid.shape[2]))
    
    @property
    def vertices(self):
        """The vertices of the barycentric coordinate system."""
        n = self.nverts
        angles = np.array(range(n))*np.pi*2/n
        vertices = [[np.sin(a), np.cos(a)] for a in angles]
        vertices = pd.DataFrame(vertices, columns=['x', 'y'],
                                index=self.vertNames)
        return vertices
    
    @property
    def hull(self):
        """The edges of the confex hull for plotting."""
        return ConvexHull(self.vertices).simplices
    
    @property
    def df(self):
        """The 2-d coordinates of the given values."""
        parts = np.dot(self.coords, self.vertices)
        pdat = pd.DataFrame(parts, columns=['x', 'y'])
        pdat['val'] = self.values
        return pdat
        
    def _vals_on_grid(self):
        """Returns the unmasked pixel colors."""
        df = self.df
        fgrid = self.fgrid
        dist = cdist(fgrid.T, df[['x','y']].values)
        ind = np.argmin(dist, axis=1)
        vals = df['val'][ind]
        return vals.values.reshape(self.grid.shape[1:])

    def in_hull(self):
        """Determines element-wise whether the points are
        inside of the convex hull spanned by the vertices
        of the barycentric coordinate system."""
        points = self.fgrid.T
        inside = np.repeat(True, len(points))
        for simplex in self.hull:
            vec = self.vertices.values[simplex]
            vec = vec.mean(axis=0, keepdims=True)
            shifted = points - vec
            below = np.dot(shifted, vec.T) < 0
            inside = np.logical_and(inside, below.T)
        return inside.reshape(self.grid.shape[1:])
    
    @property
    def plot_values(self):
        """The Pixel colors masked to the inside of
        the barycentric coordinate system."""
        values = self._vals_on_grid()
        return np.ma.masked_where(~self.in_hull(), values)
    
    @property
    def textPos(self):
        """Vertex label positions."""
        half = int(np.floor(self.nverts/2))
        odd = (self.nverts & 1) == 1
        textPos = self.vertices.copy() * 1.05
        i = textPos.index
        textPos['textVPos'] = 'center'
        textPos.loc[i[0], 'textVPos'] = 'bottom'
        textPos.loc[i[half], 'textVPos'] = 'top'
        if odd:
            textPos.loc[i[half+1], 'textVPos'] = 'top'
        textPos['textHPos'] = 'center'
        textPos.loc[i[1:half], 'textHPos'] = 'left'
        textPos.loc[i[half+1+odd:], 'textHPos']= 'right'
        return textPos
    
    def plot(self, colorbar=True, **kwargs):
        """
        
        Plots the data in barycentric coordinates.
        
        Parameters
        ----------
        colorbar : bool
            If true a colorbar is plotted on the bottom of the image.
        **kwargs
            All keyword arguments are passed on to matplotlib.imshow.
            
        Returns
        -------
        fig, ax, im
            The Figure, AxesSubplot and AxesImage of the plot.

        """
        vertices = self.vertices    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')
        ax.axis('off')
        im = ax.imshow(self.plot_values, extent=[-1, 1, -1, 1], **kwargs)
        for simplex in self.hull:
            ax.plot(vertices.values[simplex, 0], vertices.values[simplex, 1], 'k-')
        for index, row in self.textPos.iterrows():
            ax.text(row['x'], row['y'], index, ha=row['textHPos'], va=row['textVPos'])
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=self.colorbar_pad)
            fig.colorbar(im, cax=cax, orientation="horizontal")
        return fig, ax, im