import warnings
import numpy as np
import pandas as pd
from multiprocess import Pool
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

    def __init__(self, data, value_column=None, res=500):
        if value_column is not None and value_column not in data.columns.values:
            raise ValueError('`value_column` musste be a coumn name of `data`.')
        if not isinstance(res, (int, float)):
            raise ValueError('`res` musst be numerical.')
        numerical = ['float64', 'float32', 'int64', 'int32']
        if not all([d in numerical for d in data.dtypes]):
            raise ValueError('The data needs to be numerical.')
        self.res = int(res)
        if value_column is None:
            coords = data
            self.values = None
        else:
            coords = data.drop([value_column], axis=1)
            self.values = data[value_column].values
        norm = np.sum(coords.values, axis=1, keepdims=True)
        ind = np.sum(np.isnan(coords), axis=1)==0
        ind = np.logical_and(ind, ~np.isnan(self.values))
        ind = np.logical_and(ind, (norm!=0).flatten())
        self.values = self.values[ind]
        norm = norm[ind]
        coords = coords[ind]
        self.coords = coords.values / norm
        self.vertNames = list(coords.columns.values)
        self.nverts = self.coords.shape[1]
        self.colorbar_pad = .1
        if self.nverts < 3:
            raise ValueError('At least three dimensions are needed.')
        if self.nverts == 3:
            self.colorbar_pad = -.5
        elif (self.nverts & 1) == 0 or self.nverts > 5:
            self.colorbar_pad = .3

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

    @property
    def in_hull(self):
        """A mask of the grid for the part outside
        the simplex."""
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
        return np.ma.masked_where(~self.in_hull, values)

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

    def draw_polygon(self, axis=None):
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        vertices = self.vertices
        for simplex in self.hull:
            axis.plot(vertices.values[simplex, 0], vertices.values[simplex, 1], 'k-')
        for index, row in self.textPos.iterrows():
            axis.text(row['x'], row['y'], index, ha=row['textHPos'], va=row['textVPos'])
        return axis

    def imshow(self, colorbar=True, figure=None, axis=None, **kwargs):
        """

        Plots the data in barycentric coordinates and colors pixel
        with closest given value.

        Parameters
        ----------
        colorbar : bool, optional
            If true a colorbar is plotted on the bottom of the image.
            Ignored if figure is None and axis is not None.
        figure : matplotlib.figure, optional
            The figure to plot in.
        axis : matplotlib.axis, optinal
            The axis to plot in.
        **kwargs
            All keyword arguments are passed on to matplotlib.imshow.

        Returns
        -------
        figure, axis, im
            The Figure, AxesSubplot and AxesImage of the plot.

        """
        if self.values is None:
            raise ValueError('No value column supplied.')
        if figure is None and axis is not None and colorbar:
            warnings.warn('Axis but no figure is supplied,'
                          + ' so a colorbar cannot be returned.')
            colorbar = False
        elif figure is None and axis is None:
            figure = plt.figure()
        if axis is None:
            axis = figure.add_subplot(111)
        axis.set_aspect('equal', 'datalim')
        axis.axis('off')
        im = axis.imshow(self.plot_values, extent=[-1, 1, -1, 1], **kwargs)
        axis = self.draw_polygon(axis)
        if colorbar:
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('bottom', size='5%', pad=self.colorbar_pad)
            ticks = np.linspace(np.min(self.plot_values), np.max(self.plot_values), 6)
            ticks = [float('{:.2g}'.format(i)) for i in ticks]
            figure.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
        return figure, axis, im

    def scatter(self, color=None, colorbar=None, figure=None, axis=None, **kwargs):
        """

        Plots the data in barycentric coordinates.

        Parameters
        ----------
        color : bool, optional
            Color points by given values. Ignored if no value column
            is given.
        colorbar : bool, optional
            If true a colorbar is plotted on the bottom of the image.
            Ignored if figure is None and axis is not None.
        figure : matplotlib.figure, optional
            The figure to plot in.
        axis : matplotlib.axis, optinal
            The axis to plot in.
        **kwargs
            All keyword arguments are passed on to matplotlib.imshow.

        Returns
        -------
        figure, axis, pc
            The Figure, AxesSubplot and PathCollection of the plot.

        """
        if color is None and self.values is not None:
            color = True
        elif color is None:
            color = False
        if color and self.values is None:
            raise ValueError('No value column for color supplied.')
        if color and colorbar is None:
            colorbar = True
        elif colorbar is None:
            colorbar = False
        if figure is None and axis is not None and colorbar:
            warnings.warn('Axis but no figure is supplied,'
                          + ' so a colorbar cannot be returned.')
            colorbar = False
        elif figure is None and axis is None:
            figure = plt.figure()
        if axis is None:
            axis = figure.add_subplot(111)
        axis.set_aspect('equal', 'datalim')
        axis.axis('off')
        df = self.df
        if color:
            pc = axis.scatter(df['x'], df['y'], c=df['val'], **kwargs)
        else:
            pc = axis.scatter(df['x'], df['y'], **kwargs)
        axis = self.draw_polygon(axis)
        if colorbar:
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('bottom', size='5%', pad=.2)
            ticks = np.linspace(np.min(self.plot_values), np.max(self.plot_values), 6)
            ticks = [float('{:.2g}'.format(i)) for i in ticks]
            figure.colorbar(pc, cax=cax, orientation='horizontal', ticks=ticks)
        return figure, axis, pc

    def plot(self, figure=None, axis=None, **kwargs):
        """

        Plots the data in barycentric coordinates.

        Parameters
        ----------
        figure : matplotlib.figure, optional
            The figure to plot in.
        axis : matplotlib.axis, optinal
            The axis to plot in.
        **kwargs
            All keyword arguments are passed on to matplotlib.imshow.

        Returns
        -------
        figure, axis, ll
            The Figure, AxesSubplot and list of Line2D of the plot.

        """
        if figure is None and axis is None:
            figure = plt.figure()
        if axis is None:
            axis = figure.add_subplot(111)
        axis.set_aspect('equal', 'datalim')
        axis.axis('off')
        df = self.df
        ll = axis.plot(df['x'], df['y'], **kwargs)
        axis = self.draw_polygon(axis)
        return figure, axis, ll
