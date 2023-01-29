import pandas as pd
from matplotlib import pyplot as plt,ticker as mticker
import matplotlib as mpl
import matplotlib.font_manager as font_manager 

plt.rcParams.update({'text.usetex': True}) 
mpl.style.use('seaborn-colorblind')
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{mathbf} \usepackage{mathsf} \usepackage{bm}  \usepackage{boldmath} \usepackage{boldsymbol}'
  
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'

from scipy.signal import lfilter

y = pd.read_csv('SBEHISTORY9-10__DD__Agents9_height_10_iterations_12001_rho_0.0001phi1_exp_3_alpha_0.1-1-0.csv')
y = pd.DataFrame(y)
y = y.iloc[:,0]

y3 = pd.read_csv('SBEHISTORY9-10__CC__Agents9_height_10_iterations_12001_rho_0.0001phi1_exp_3_alpha_0.1-0-0.csv')
y3 = pd.DataFrame(y3)
y3 = y3.iloc[:,0]

y2 = pd.read_csv('SBEHISTORY9-10__CD__Agents9_height_10_iterations_12001_rho_0.0001phi1_exp_3_alpha_0.1-2-0.csv')
y2 = pd.DataFrame(y2)
y2 = y2.iloc[:,0]
 
v = y.ewm(alpha=0.025).mean() 
v2 = y2.ewm(alpha=0.025).mean()
v3 = y3.ewm(alpha=0.025).mean()
 
plt.rcParams["figure.figsize"] = (40,10)
fiii, ax = plt.subplots(1,1)

ax.tick_params(axis='y') 
ax.set_axisbelow(False)

ax.grid(which = 'both',alpha = 0.5, zorder=3,linestyle='dashed')
ax.set_axisbelow(False)
plt.rc('axes', axisbelow=False) 

ax.plot(v, color = 'blue', label = r'Diffusion')
ax.plot(y, color = 'blue',alpha = 0.3, linewidth = 2)

ax.plot(v2, color = 'green', label = r' CD')
ax.plot(y2, color = 'green',alpha = 0.3, linewidth = 2)

ax.plot(v3, color = 'red', label = r'CC')
ax.plot(y3, color = 'red',alpha = 0.3, linewidth = 2)

 
ax.set_xlabel(r'$i$', fontsize=24) 
ax.set_ylabel(r'SBE' , fontsize=24,position=(100, 0.5)) 

plt.yticks(fontsize=24)
plt.xticks(fontsize=24)

ax.set_yscale('log')
plt.style.use(['science','ieee'])

fiii.subplots_adjust(bottom=0.2)
fiii.subplots_adjust(left=-0.1)
fiii.subplots_adjust(right=0.1) 

ax.legend(fontsize=24, handlelength=1.2,frameon=True)
 
fiii.savefig('SBE_Error.PNG') 
