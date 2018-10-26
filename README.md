
# Multi Bary Plot

A classy way to get a 2-d visualization of n-dimensional data using the generalized barycentric coordinate system.

We use the closest value in barycentric coordinates to color the pixels according to the given values.

## Install

```
pip install git+ssh://git@ribogit.izi.fraunhofer.de/Dominik/multi_bary_plot.git
```

## Example


```python
from multi_bary_plot import multi_bary_plot
import pandas as pd
```

### 3 Dimensions


```python
# generate data
vec = list(range(100))
pdat = pd.DataFrame({'class 1':vec,
                     'class 2':list(reversed(vec)),
                     'class 3':[50]*100,
                     'val':vec})

# plot
bp = multi_bary_plot(pdat, 'val')
fig, ax, im = bp.plot()
```


![png](README_files/README_3_0.png)


### 8 Dimensions


```python
# generate data
vec = list(range(100))
pdat = pd.DataFrame({'class 1':vec,
                     'class 2':[v**2/10 for v in vec],
                     'class 3':vec,
                     'class 4':vec,
                     'class 5':vec,
                     'class 6':vec,
                     'class 7':[50]*100,
                     'class 8':[50]*100,
                     'val':vec})

# plot
bp = multi_bary_plot(pdat, 'val')
fig, ax, im = bp.plot(cmap='plasma')
```


![png](README_files/README_5_0.png)

