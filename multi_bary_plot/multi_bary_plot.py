import warnings
import numpy as np
import pandas as pd
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
    def text_position(self):
        """Vertex label positions."""
        half = int(np.floor(self.nverts/2))
        odd = (self.nverts & 1) == 1
        tp = self.vertices.copy() * 1.05
        i = tp.index
        tp['v_align'] = 'center'
        tp.loc[i[0], 'v_align'] = 'bottom'
        tp.loc[i[half], 'v_align'] = 'top'
        if odd:
            tp.loc[i[half+1], 'v_align'] = 'top'
        tp['h_align'] = 'center'
        tp.loc[i[1:half], 'h_align'] = 'left'
        tp.loc[i[half+1+odd:], 'h_align']= 'right'
        return tp

    def draw_polygon(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        vertices = self.vertices
        for simplex in self.hull:
            ax.plot(vertices.values[simplex, 0],
                    vertices.values[simplex, 1], 'k-')
        for index, row in self.text_position.iterrows():
            ax.text(row['x'], row['y'], index,
                    ha=row['h_align'], va=row['v_align'])
        return ax

    def imshow(self, colorbar=True, fig=None, ax=None, **kwargs):
        """

        Plots the data in barycentric coordinates and colors pixels
        according to the closest given value.

        Parameters
        ----------
        colorbar : bool, optional
            If true a colorbar is plotted on the bottom of the image.
            Ignored if figure is None and axes is not None.
        fig : matplotlib.figure, optional
            The figure to plot in.
        ax : matplotlib.axes, optional
            The axes to plot in.
        **kwargs
            Other keyword arguments are passed on to
            matplotlib.pyplot.imshow.

        Returns
        -------
        fig, ax, im
            The matplotlib Figure, AxesSubplot,
            and AxesImage of the plot.

        """
        if self.values is None:
            raise ValueError('No value column supplied.')
        if fig is None and ax is not None and colorbar:
            warnings.warn('axes but no figure is supplied,'
                          + ' so a colorbar cannot be plotted.')
            colorbar = False
        elif fig is None and ax is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        ax.axis('off')
        im = ax.imshow(self.plot_values, extent=[-1, 1, -1, 1], **kwargs)
        ax = self.draw_polygon(ax)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='5%', pad=.2)
            def round_ticks(ticks):
                return [float('{:.2g}'.format(t)) for t in ticks]
            ub = np.max(self.plot_values)
            lb = np.min(self.plot_values)
            pre_ticks = np.linspace(lb, ub, 6)
            ticks = round_ticks(pre_ticks)
            if ticks[0] < lb:
                pre_ticks[0] += 10**np.floor(np.log10(np.abs(lb))-1)
            if ticks[-1] > ub:
                pre_ticks[-1] -= 10**np.floor(np.log10(np.abs(ub))-1)
            ticks = round_ticks(pre_ticks)
            fig.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
        # manual limits because of masked data
        v = self.vertices
        xpad =  (v['x'].max()-v['x'].min()) * .05
        ax.set_xlim([v['x'].min()-xpad, v['x'].max()+xpad])
        ypad =  (v['y'].max()-v['y'].min()) * .05
        ax.set_ylim([v['y'].min()-ypad, v['y'].max()+ypad])
        ax.set_aspect('equal')
        return fig, ax, im

    def scatter(self, color=None, colorbar=None, fig=None,
            ax=None, **kwargs):
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
        fige : matplotlib.figure, optional
            The figure to plot in.
        ax : matplotlib.axes, optional
            The axes to plot in.
        **kwargs
            Other keyword arguments are passed on to
            matplotlib.pyplot.scatter. The keyword argument c
            overwrites given values in the data.

        Returns
        -------
        fig, ax, pc
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
        if fig is None and ax is not None and colorbar:
            warnings.warn('axes but no figure is supplied,'
                          + ' so a colorbar cannot be plotted.')
            colorbar = False
        elif fig is None and ax is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')
        ax.axis('off')
        p2 = self.points_2d
        if color and 'c' not in kwargs:
            pc = ax.scatter(p2['x'], p2['y'], c=p2['val'], **kwargs)
        else:
            pc = ax.scatter(p2['x'], p2['y'], **kwargs)
        ax = self.draw_polygon(ax)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='5%', pad=.2)
            if 'c' in kwargs:
                vals = kwargs['c']
            else:
                vals = self.plot_values
            ticks = np.linspace(np.min(vals), np.max(vals), 6)
            ticks = [float('{:.2g}'.format(i)) for i in ticks]
            fig.colorbar(pc, cax=cax, orientation='horizontal', ticks=ticks)
        return fig, ax, pc

    def plot(self, fig=None, ax=None, **kwargs):
        """

        Plots the data in barycentric coordinates.

        Parameters
        ----------
        fig : matplotlib.figure, optional
            The figure to plot in.
        ax : matplotlib.axes, optional
            The axes to plot in.
        **kwargs
            Other keyword arguments are passed on to
            matplotlib.pyplot.plot.

        Returns
        -------
        fig, ax, ll
            The matplotlib Figure, AxesSubplot,
            and list of Line2D of the plot.

        """
        if fig is None and ax is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')
        ax.axis('off')
        p2 = self.points_2d
        ll = ax.plot(p2['x'], p2['y'], **kwargs)
        ax = self.draw_polygon(ax)
        return fig, ax, ll
