#Figure 4 has our fiducial chain, non-tomographic constraints, 
#pseudo Cls, Cl bandpowers, and Planck
from cosmosis.postprocessing import plots
from cosmosis.postprocessing import lazy_pylab as pylab
from cosmosis.postprocessing import statistics
from cosmosis.plotting.kde import KDE

from cosmosis.postprocessing.elements import PostProcessorElement
from cosmosis.postprocessing.elements import MCMCPostProcessorElement, MultinestPostProcessorElement, WeightedMCMCPostProcessorElement
from cosmosis.postprocessing.elements import Loadable
from cosmosis.postprocessing.outputs import PostprocessPlot
from cosmosis.postprocessing.utils import std_weight, mean_weight

import numpy as np
import scipy

import matplotlib.colors
import matplotlib
#matplotlib.rcParams['font.family']='serif'
matplotlib.rcParams['font.size']=18
matplotlib.rcParams['legend.fontsize']=40
matplotlib.rcParams['xtick.major.size'] = 10.0
matplotlib.rcParams['ytick.major.size'] = 10.0

blind=False

class plot1D(plots.GridPlots1D):
    contours=[]
    proxies=[]
    def __init__(self, *args, **kwargs):
        super(plot1D, self).__init__(*args, **kwargs)
        self.colors=['darkmagenta', 'steelblue','plum','pink']
        self.linestyles=['-','--','-',':']*10
        pylab.style.use('y1a1')
        matplotlib.rcParams['xtick.major.size'] = 3.5
        matplotlib.rcParams['xtick.minor.size'] = 1.7
        matplotlib.rcParams['ytick.major.size'] = 3.5
        matplotlib.rcParams['ytick.minor.size'] = 1.7
        matplotlib.rcParams['xtick.direction']='in'
        matplotlib.rcParams['ytick.direction']='in'
        #matplotlib.rcParams['text.usetex']=False
        matplotlib.rcParams['font.family']='serif'
        self.labels= [ r'2pt $(r_\mathrm{p} > 6 h^{-1} \mathrm{Mpc})$',
                       r'1pt $(6.4 h^{-1} \mathrm{Mpc})$',
                       r'1pt $(3.2 h^{-1} \mathrm{Mpc})$',
                       r'1pt $(1.6 h^{-1} \mathrm{Mpc})$'] 
        self.axis=[-0.5,2.8]
        self.fill_list=[True,True,True,True,False,False,False,False,True,True,True]*10
        self.line_list=[True,True,True,True]*10
        self.opaque_list=[True,False,False,False,False,False,False,False,True,True]*10
        self.linewidths=[2]*len(self.colors)
        self.alphas=[0.2,0.2,0.2,0.2,0.4,0.4,0.4]*len(self.colors)
        self.imshow=[False]*10
        self.linecolors=[None]*10
        self.opaque_centre=[False]*10
        self.fill_colors=[None,None]*10
        self.plot_points=[]
        #self.proxies=[]

    def run(self):
        name="intrinsic_alignment_parameters--a1"
        filename = self.make_1d_plot(name)
        return filename

    def find_or_pad(self,name):
        try:
            x = self.reduced_col(name)
        except:
            if (name=='intrinsic_alignment_parameters--c1'):
                x = self.reduced_col('intrinsic_alignment_parameters--a')
            else:
                x0 = self.reduced_col('cosmological_parameters--sigma_8')
                x = np.random.rand(x0.size)*60 - 30
        return x 
    
    def make_1d_plot(self, name1):
          #Get the data
        #import pdb ; pdb.set_trace()

        filename = self.filename(name1)
        col1 = self.source.get_col(name1)
        like = self.source.get_col("post")

        vals1 = np.unique(col1)
        n1 = len(vals1)
        like_sum = np.zeros(n1)

        #normalizing like this is a big help 
        #numerically
        like = like-like.max()

        #marginalize
        for k,v1 in enumerate(vals1):
            w = np.where(col1==v1)
            like_sum[k] = np.log(np.exp(like[w]).sum())
        like = like_sum.flatten()
        like -= like.max()


        #linearly interpolate
        n1_interp = n1*10
        vals1_interp = np.linspace(vals1[0], vals1[-1], n1_interp)
        like_interp = np.interp(vals1_interp, vals1, like)
        if np.isfinite(like_interp).any():
            vals1 = vals1_interp
            like = like_interp
            n1 = n1_interp
        else:
            print()
            print("Parameter %s has a very wide range in likelihoods " % name1)
            print("So I couldn't do a smooth likelihood interpolation for plotting")
            print()


        #normalize
        like[np.isnan(like)] = -np.inf
        like -= like.max()


        #Determine the spacing in the different parameters
        dx = vals1[1]-vals1[0]

        #Set up the figure
        fig,filename = self.figure(name1)
        pylab.figure(fig.number)

        #Plot the likelihood
        print('----------------------- HITTA',self.source.index, self.colors[self.source.index])
        pylab.plot(vals1, np.exp(like), linewidth=self.linewidths[self.source.index], label=self.labels[self.source.index], color=self.colors[self.source.index], linestyle='-')
        self.proxies.append(pylab.Line2D([0,2.5],[0,0], color=self.colors[self.source.index], linewidth=self.linewidths[self.source.index]))

        #Find the levels of the 68% and 95% contours
        X, L = self.find_edges(np.exp(like), 0.68, 0.95, vals1)
        #Plot black dotted lines from the y-axis at these contour levels
        for (x, l) in zip(X,L):
            if np.isnan(x[0]):
                continue
            pylab.plot([x[0],x[0]], [0, l[0]], ':', color='black')
            pylab.plot([x[1],x[1]], [0, l[1]], ':', color='black')

        #import pdb ; pdb.set_trace()

        if self.labels is not None:
            leg = pylab.legend(self.proxies,self.labels,loc="upper left", fontsize=16)
            leg.get_frame().set_alpha(0.75) # this will make the box totally transparent
            leg.get_frame().set_edgecolor('white') # this will make the edges of the 

        #Set the x and y limits
        pylab.ylim(0,1.05)
        pylab.yticks(visible=False)
        pylab.xticks(visible=True, fontsize=16)
        pylab.xlim(self.axis[0],self.axis[1])
        #Add label
        pylab.xlabel('$A_1$', fontsize=18)
        return filename