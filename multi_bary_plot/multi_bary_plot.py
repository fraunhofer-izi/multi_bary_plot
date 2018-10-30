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
        The number of pixel along one axes.

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
        self.res = int(res)
        numerical = ['float64', 'float32', 'int64', 'int32']
        if not all([d in numerical for d in data.dtypes]):
            raise ValueError('The data needs to be numerical.')
        if value_column is None:
            coords = data
            self.values = None
        else:
            coords = data.drop([value_column], axis=1)
            self.values = data[value_column].values
        norm = np.sum(coords.values, axis=1, keepdims=True)
        ind = np.sum(np.isnan(coords), axis=1)==0
        ind = np.logical_and(ind, (norm!=0).flatten())
        if self.values is not None:
            ind = np.logical_and(ind, ~np.isnan(self.values))
            self.values = self.values[ind]
        norm = norm[ind]
        coords = coords[ind]
        self.coords = coords.values / norm
        self.vertNames = list(coords.columns.values)
        self.nverts = self.coords.shape[1]
        if self.nverts < 3:
            raise ValueError('At least three dimensions are needed.')

    @property
    def grid(self):
        """The grid of pixels to raster."""
        x = np.linspace(-1, 1, self.res)
        return np.array(np.meshgrid(x, -x))

    @property
    def mgrid(self):
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
    def points_2d(self):
        """The 2-d coordinates of the given values."""
        parts = np.dot(self.coords, self.vertices)
        pdat = pd.DataFrame(parts, columns=['x', 'y'])
        pdat['val'] = self.values
        return pdat

    def _vals_on_grid(self):
        """Returns the unmasked pixel colors."""
        p2 = self.points_2d
        dist = cdist(self.mgrid.T, p2[['x','y']].values)
        ind = np.argmin(dist, axis=1)
        vals = p2['val'][ind]
        return vals.values.reshape(self.grid.shape[1:])

    @property
    def in_hull(self):
        """A mask of the grid for the part outside
        the simplex."""
        pixel = self.mgrid.T
        inside = np.repeat(True, len(pixel))
        for simplex in self.hull:
            vec = self.vertices.values[simplex]
            vec = vec.mean(axis=0, keepdims=True)
            shifted = pixel - vec
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

    def draw_polygon(self, axes=None):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        vertices = self.vertices
        for simplex in self.hull:
            axes.plot(vertices.values[simplex, 0],
                    vertices.values[simplex, 1], 'k-')
        for index, row in self.textPos.iterrows():
            axes.text(row['x'], row['y'], index,
                    ha=row['textHPos'], va=row['textVPos'])
        return axes

    def imshow(self, colorbar=True, figure=None, axes=None, **kwargs):
        """

        Plots the data in barycentric coordinates and colors pixels
        according to the closest given value.

        Parameters
        ----------
        colorbar : bool, optional
            If true a colorbar is plotted on the bottom of the image.
            Ignored if figure is None and axes is not None.
        figure : matplotlib.figure, optional
            The figure to plot in.
        axes : matplotlib.axis, optinal
            The axes to plot in.
        **kwargs
            Other keyword arguments are passed on to
            matplotlib.pyplot.imshow.

        Returns
        -------
        figure, axes, im
            The matplotlib Figure, AxesSubplot,
            and AxesImage of the plot.

        """
        if self.values is None:
            raise ValueError('No value column supplied.')
        if figure is None and axes is not None and colorbar:
            warnings.warn('axes but no figure is supplied,'
                          + ' so a colorbar cannot be returned.')
            colorbar = False
        elif figure is None and axes is None:
            figure = plt.figure()
        if axes is None:
            axes = figure.add_subplot(111)
        axes.axis('off')
        im = axes.imshow(self.plot_values, extent=[-1, 1, -1, 1], **kwargs)
        axes = self.draw_polygon(axes)
        if colorbar:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('bottom', size='5%', pad=.2)
            ticks = np.linspace(np.min(self.plot_values),
                    np.max(self.plot_values), 6)
            ticks = [float('{:.2g}'.format(i)) for i in ticks]
            figure.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
        v = self.vertices
        xlen =  v['x'].max() - v['x'].min()
        axes.set_xlim([v['x'].min()-(xlen*.05), v['x'].max()+(xlen*.05)])
        ylen =  v['y'].max() - v['y'].min()
        axes.set_ylim([v['y'].min()-(ylen*.05), v['y'].max()+(ylen*.05)])
        axes.set_aspect('equal')
        return figure, axes, im

    def scatter(self, color=None, colorbar=None, figure=None,
            axes=None, **kwargs):
        """

        Scatterplot of the data in barycentric coordinates.

        Parameters
        ----------
        color : bool, optional
            Color points by given values. Ignored if no value column
            is given.
        colorbar : bool, optional
            If true a colorbar is plotted on the bottom of the image.
            Ignored if figure is None and axes is not None.
        figure : matplotlib.figure, optional
            The figure to plot in.
        axes : matplotlib.axis, optinal
            The axes to plot in.
        **kwargs
            Other keyword arguments are passed on to
            matplotlib.pyplot.scatter. The keyword argument c
            overwrites given values in the data.

        Returns
        -------
        figure, axes, pc
            The matplotib Figure, AxesSubplot,
            and PathCollection of the plot.

        """
        color_info = self.values is not None or 'c' in kwargs
        if color is None and color_info:
            color = True
        elif color is None:
            color = False
        if color and not color_info:
            raise ValueError('No value column for color supplied.')
        if color and colorbar is None:
            colorbar = True
        elif colorbar is None:
            colorbar = False
        if figure is None and axes is not None and colorbar:
            warnings.warn('axes but no figure is supplied,'
                          + ' so a colorbar cannot be returned.')
            colorbar = False
        elif figure is None and axes is None:
            figure = plt.figure()
        if axes is None:
            axes = figure.add_subplot(111)
        axes.set_aspect('equal', 'datalim')
        axes.axis('off')
        p2 = self.points_2d
        if color and 'c' not in kwargs:
            pc = axes.scatter(p2['x'], p2['y'], c=p2['val'], **kwargs)
        else:
            pc = axes.scatter(p2['x'], p2['y'], **kwargs)
        axes = self.draw_polygon(axes)
        if colorbar:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('bottom', size='5%', pad=.2)
            if 'c' in kwargs:
                vals = kwargs['c']
            else:
                vals = self.plot_values
            ticks = np.linspace(np.min(vals), np.max(vals), 6)
            ticks = [float('{:.2g}'.format(i)) for i in ticks]
            figure.colorbar(pc, cax=cax, orientation='horizontal', ticks=ticks)
        return figure, axes, pc

    def plot(self, figure=None, axes=None, **kwargs):
        """

        Plots the data in barycentric coordinates.

        Parameters
        ----------
        figure : matplotlib.figure, optional
            The figure to plot in.
        axes : matplotlib.axis, optinal
            The axes to plot in.
        **kwargs
            Other keyword arguments are passed on to
            matplotlib.pyplot.plot.

        Returns
        -------
        figure, axes, ll
            The matplotlib Figure, AxesSubplot,
            and list of Line2D of the plot.

        """
        if figure is None and axes is None:
            figure = plt.figure()
        if axes is None:
            axes = figure.add_subplot(111)
        axes.set_aspect('equal', 'datalim')
        axes.axis('off')
        p2 = self.points_2d
        ll = axes.plot(p2['x'], p2['y'], **kwargs)
        axes = self.draw_polygon(axes)
        return figure, axes, ll
