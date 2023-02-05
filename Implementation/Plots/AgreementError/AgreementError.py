import csv
import pandas as pd
from matplotlib import pyplot as plt,ticker as mticker
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
from math import log

plt.rcParams.update({'text.usetex': True}) 
mpl.style.use('seaborn-colorblind')
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{mathbf} \usepackage{mathsf} \usepackage{bm}  \usepackage{boldmath} \usepackage{boldsymbol}'
  
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'

from scipy.signal import lfilter
 

ya = pd.read_csv('DD_AgreementError_0.csv',header = None)
yb = pd.read_csv('DD_AgreementError_1.csv',header = None)
yc = pd.read_csv('DD_AgreementError_2.csv',header = None)

ya = pd.DataFrame(ya)
yb = pd.DataFrame(yb)
yc = pd.DataFrame(yc)

# print(yb[1])
ya[1] = yb[0] 
ya[2] = yc[0] 
ya[3] = ya.mean(axis = 1)

y = ya.iloc[:,3]
print(ya)
print(yb)
v =  y.ewm(alpha=0.07).mean() 
  
import matplotlib as mpl
import matplotlib.font_manager as font_manager 

plt.rcParams["figure.figsize"] = (40,10)
fiii, ax = plt.subplots(1,1) 
ax.tick_params(axis='y') 
ax.set_axisbelow(False)

ax.grid(which = 'both',alpha = 0.5, zorder=3,linestyle='dashed')
ax.set_axisbelow(False)
plt.rc('axes', axisbelow=False) 
ax.plot(v, color = '#ED7117')
ax.plot(y, color = '#ED7117', alpha = 0.3, linewidth = 2)
 
ax.set_xlabel(r'$i$', fontsize=24)  
ax.set_ylabel(r'Agreement Error' , fontsize=24,position=(100, 0.5)) 

plt.yticks(fontsize=24)
plt.xticks(fontsize=24) 
ax.set_yscale('log')
plt.style.use(['science','ieee'])

fiii.subplots_adjust(bottom=0.2)
fiii.subplots_adjust(left=-0.1)
fiii.subplots_adjust(right=0.1) 
 
fiii.savefig('AgreementError.png') 
