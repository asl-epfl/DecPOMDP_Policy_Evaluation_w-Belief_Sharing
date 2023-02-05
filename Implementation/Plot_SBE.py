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

ya = pd.read_csv('DD_SBE_0.csv',header = None)
yb = pd.read_csv('DD_SBE_1.csv',header = None)
yc = pd.read_csv('DD_SBE_2.csv',header = None)
 
ya = pd.DataFrame(ya)
yb = pd.DataFrame(yb)
yc = pd.DataFrame(yc)

ya[1] = yb[0]
ya[2] = yc[0]
ya[3] = ya.mean(axis = 1)

y = ya.iloc[:,3]


ya = pd.read_csv('CC_SBE_0.csv',header = None)
yb = pd.read_csv('CC_SBE_1.csv',header = None)
yc = pd.read_csv('CC_SBE_2.csv',header = None)
 
ya = pd.DataFrame(ya)
yb = pd.DataFrame(yb)
yc = pd.DataFrame(yc)

ya[1] = yb[0]
ya[2] = yc[0]
ya[3] = ya.mean(axis = 1)

y3 = ya.iloc[:,3] 


ya = pd.read_csv('CD_SBE_0.csv',header = None)
yb = pd.read_csv('CD_SBE_1.csv',header = None) 
yc = pd.read_csv('CD_SBE_2.csv',header = None) 
 
ya = pd.DataFrame(ya)
yb = pd.DataFrame(yb) 
yc = pd.DataFrame(yb) 
ya[1] = yb[0] 
ya[2] = yc[0]
ya[3] = ya.mean(axis = 1)

y2 = ya.iloc[:,3] 
 
 
v = y.ewm(alpha=0.025).mean() 
y = y.rolling(20).mean()

v2 = y2.ewm(alpha=0.025).mean()
y2 = y2.rolling(20).mean()

v3 = y3.ewm(alpha=0.025).mean()
y3 = y3.rolling(20).mean()
 
plt.rcParams["figure.figsize"] = (40,10)
fiii, ax = plt.subplots(1,1)

ax.tick_params(axis='y') 
ax.set_axisbelow(False)

ax.grid(which = 'both',alpha = 0.5, zorder=3,linestyle='dashed')
ax.set_axisbelow(False)
plt.rc('axes', axisbelow=False) 

ax.plot(v, color = 'blue', label = r'Diffusion')
ax.plot(y, color = 'blue',alpha = 0.2, linewidth = 2)

ax.plot(v2, color = 'green', label = r' CD')
ax.plot(y2, color = 'green',alpha = 0.2, linewidth = 2)

ax.plot(v3, color = 'red', label = r'CC')
ax.plot(y3, color = 'red',alpha = 0.2, linewidth = 2)

 
ax.set_xlabel(r'$i$', fontsize=24) 
ax.set_ylabel(r'SBE' , fontsize=24,position=(100, 0.5)) 

plt.yticks(fontsize=24)
plt.xticks(fontsize=24)

ax.set_yscale('log')
plt.style.use(['science','ieee'])
plt.ylim(top=100)

fiii.subplots_adjust(bottom=0.2)
fiii.subplots_adjust(left=-0.1)
fiii.subplots_adjust(right=0.1) 

ax.legend(fontsize=24, handlelength=1.2,frameon=True)
 
fiii.savefig('SBE_Error.PNG') 
