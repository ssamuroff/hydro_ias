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

class plot2D(plots.MultinestPlots2D):
    contours=[]
    proxies=[]
    def __init__(self, *args, **kwargs):
        super(plot2D, self).__init__(*args, **kwargs)
        self.colors=[ 'darkmagenta', 'k', 'forestgreen','purple','royalblue', 'k','pink','steelblue',"purple", 'k', 'k', 'k', 'k']*10
        self.linestyles=['-','--','-',':']*10
        matplotlib.rcParams['xtick.major.size'] = 3.5
        matplotlib.rcParams['xtick.minor.size'] = 1.7
        matplotlib.rcParams['ytick.major.size'] = 3.5
        matplotlib.rcParams['ytick.minor.size'] = 1.7
        matplotlib.rcParams['xtick.direction']='in'
        matplotlib.rcParams['ytick.direction']='in'
        #matplotlib.rcParams['text.usetex']=False
        matplotlib.rcParams['font.family']='serif'
        self.labels= [ ] 

        self.axis=[0.5,1.4,1.3,3.8]
        self.fill_list=[True,False,False,False,False,False,False,False,True,True,True]*10
        self.line_list=[True,True,True,True]*10
        self.opaque_list=[True,False,False,False,False,False,False,False,True,True]*10
        self.linewidths=[2.]*len(self.colors)
        self.alphas=[0.2,0.6,0.6,0.6,0.4,0.4,0.4]*len(self.colors)
        self.imshow=[False]*10
        self.linecolors=[None]*10
        self.opaque_centre=[False]*10
        self.fill_colors=[None,None]*10
        self.plot_points=[(0.05,1.5)]
        #self.proxies=[]

    def run(self):
        name1="bias_parameters--b_0_300"
        name2="intrinsic_alignment_parameters--a_0_300"
        if "fisher" not in self.source.name:
            filename = self.make_2d_plot(name1, name2)
        else:
            filename=None  
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
    
    def make_2d_plot(self, name1, name2):
          #Get the data
        if 'planck' in self.source.name.lower():
            def smooth_likelihood(obj,x, y):
                n = obj.options.get("n_kde", 100)
                factor = obj.options.get("factor_kde", 2.0)
                kde = KDE([x,y], factor=factor)
                x_range = (x.min(), x.max())
                y_range = (y.min(), y.max())
                (x_axis, y_axis), like = kde.grid_evaluate(n, [x_range, y_range])
                return n, x_axis, y_axis, like
            x = self.reduced_col(name1) ; y =self.reduced_col(name2)

        else:
            if name1[-2:]=='s8':
                omm = self.reduced_col('cosmological_parameters--omega_m')
                sigma_8 = self.reduced_col('cosmological_parameters--sigma_8')
                x = sigma_8 * (omm/0.3)**0.5
            else:
                x = self.find_or_pad(name1)
            if name2[-2:]=='s8':
                omm= self.reduced_col('cosmological_parameters--omega_m')
                sigma_8 = self.reduced_col('cosmological_parameters--sigma_8')
                y =sigma_8 * (omm/0.3)**0.5
            else:
                y = self.find_or_pad(name2)


        if x.max()-x.min()==0 or y.max()-y.min()==0:
            return
        print "  (making %s vs %s)" % (name1, name2)

        #import pdb ; pdb.set_trace()


        #Interpolate using KDE
        try:
            n, x_axis, y_axis, like = self.smooth_likelihood(x, y)
        except:
            try:
                n, x_axis, y_axis, like = self.smooth_likelihood_noweight(x, y)
            except:
                def smooth_likelihood(obj,x, y):
                    n = 100
                    factor =  2.0
                    kde = KDE([x,y], factor=factor)
                    x_range = (x.min(), x.max())
                    y_range = (y.min(), y.max())
                    (x_axis, y_axis), like = kde.grid_evaluate(n, [x_range, y_range])
                    return n, x_axis, y_axis, like
                n, x_axis, y_axis, like = smooth_likelihood(self, x, y)

        figure,filename = self.figure("2D", name1, name2)

          #Choose levels at which to plot contours
        
        contour1=1-0.68
        contour2=1-0.95
        try: 
            level1, level2, total_mass = self._find_contours(like, x, y, n, x_axis[0], x_axis[-1], y_axis[0], y_axis[-1], contour1, contour2)
        except:
            def find_contours(like, x, y, n, xmin, xmax, ymin, ymax, contour1, contour2):
                N = len(x)
                x_axis = np.linspace(xmin, xmax, n+1)
                y_axis = np.linspace(ymin, ymax, n+1)
                histogram, _, _ = np.histogram2d(x, y, bins=[x_axis, y_axis])
                def objective(limit, target):
                    w = np.where(like>=limit)
                    count = histogram[w]
                    return count.sum() - target
                target1 = histogram.sum()*(1-contour1)
                target2 = histogram.sum()*(1-contour2)

                level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,))
                level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,))
                return level1, level2, like.sum()
            level1, level2, total_mass = find_contours(like, x, y, n, x_axis[0], x_axis[-1], y_axis[0], y_axis[-1], contour1, contour2)
            #import pdb ; pdb.set_trace()
        level0 = 1.1
        levels = [level2, level1, level0]
        #Create the figure
        fig,filename = self.figure("2D", name1, name2)
        pylab.figure(fig.number)

        #Make the plot
        #pylab.figure(figure.number, figsize=(8,7))
        keywords = self.keywords_2d()
        fill = self.options.get("fill", True)
        if self.fill_list!=None:
            fill = self.fill_list[self.source.index]
        imshow = self.imshow[self.source.index]
        #plot_points = self.options.get("plot_points", False)
        color = self.colors[self.source.index]
        linestyle = self.linestyles[self.source.index]
        if self.linecolors[self.source.index] is not None:
            linecolor=self.linecolors[self.source.index]
        else:
            linecolor=color
        print self.source.index,linecolor
        
        if self.opaque_list[self.source.index]:
            pylab.contourf(x_axis, y_axis, like.T, [level2,level0], colors=['white'])
        print "fill",self.source.index, fill

        if imshow:
            pylab.imshow(like.T, extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]), aspect='auto', origin='lower', cmap=cmap_white_to_color(color))
        elif fill:
            if self.fill_colors[self.source.index] is not None:
                c=self.fill_colors[self.source.index]
                pylab.contourf(x_axis, y_axis, like.T, [level2,level0], colors=[c[0]])
                pylab.contourf(x_axis, y_axis, like.T, [level1,level0], colors=[c[1]])
            else:
                pylab.contourf(x_axis, y_axis, like.T, [level2,level0], colors=[color], alpha=self.alphas[self.source.index])
                if self.opaque_centre[self.source.index]:
                    pylab.contourf(x_axis, y_axis, like.T, [level1,level0], colors=[color])
                else:
                    pylab.contourf(x_axis, y_axis, like.T, [level1,level0], colors=[color], alpha=self.alphas[self.source.index])            
        if (not fill) or self.line_list[self.source.index]: 
            print self.source.index,linecolor
            pylab.contour(x_axis, y_axis, like.T, [level2,level1], colors=[linecolor], linestyles=linestyle, linewidths=self.linewidths[self.source.index])
            #if self.labels is not None:
            #    self.proxies.append(pylab.plot([],[], color=color, linestyle=linestyle, linewidth=self.linewidths[self.source.index])[0])
        if self.labels is not None:
            if self.fill_colors[self.source.index] is not None:
                print self.fill_colors[self.source.index]
            elif fill:
                print self.source.index,self.line_list[self.source.index]
                if self.line_list[self.source.index] is not None:                    
                    conv=matplotlib.colors.ColorConverter()
                    edgecolor=linecolor
                    self.proxies.append(pylab.Line2D([0,2.5], [0,0], linestyle=linestyle, linewidth=3, color=linecolor))
                    #facecolor=conv.to_rgba(color,alpha=1-(1-self.alphas[self.source.index])**2)
                    #self.proxies.append(pylab.Rectangle((0,0),1,1,facecolor=facecolor,edgecolor=edgecolor))#,alpha=(1-(1-self.alphas[self.source.index])**2)))
                else:
                    self.proxies.append(pylab.Rectangle((0,0),1,1,fc=color,edgecolor=color,alpha=(1-(1-self.alphas[self.source.index])**2)))
            else:
                self.proxies.append(pylab.plot([],[], color=color, linestyle=linestyle, linewidth=self.linewidths[self.source.index])[0])

        if self.plot_points is not None:
            point_markers=['x','+','^']
            for p,m in zip(self.plot_points,point_markers):
                print p,m
                pylab.plot(p[0], p[1], m, color='k', markersize=10, markerfacecolor='none')

        #Do the labels
        print self.proxies,self.labels
        if self.labels is not None:
            leg=pylab.legend(self.proxies,self.labels,loc="upper left", fontsize=13)
            leg.get_frame().set_alpha(0.75) # this will make the box totally transparent
            leg.get_frame().set_edgecolor('white') # this will make the edges of the 
        pylab.ylabel(r"$A_\mathrm{IA}$",fontsize=22, fontname='serif')
        pylab.xlabel(r"$b_g$", fontsize=22, fontname='serif')
        pylab.tight_layout()
        for col,text in zip(self.colors,leg.get_texts()):
            text.set_color(col)
            if col=='pink':
                text.set_color('hotpink')
        if blind:
            pylab.xticks(visible=False)
        pylab.xticks(visible=True, fontsize=18)
        pylab.yticks(visible=True, fontsize=18)
        pylab.xlim(self.axis[0], self.axis[1])
        pylab.ylim(self.axis[2], self.axis[3])
        pylab.title('$z=0.3$', fontsize=25)
 #       pylab.subplots_adjust(top=0.95)
        x0 = [4.6]
        dx = [0.5]
        y0 = [1.77]
        pylab.errorbar(x0,y0,xerr=dx,yerr=[0.04], markersize=9.0, markerfacecolor='k',color='k', marker='*',markeredgecolor='k', linestyle='none')
      #  pylab.errorbar([4.35341365],[1.80416667], xerr=7.080459e-01, yerr=1.706998e-02,color='pink',marker='o')
        #pylab.title('$\gamma \gamma + \delta_g \gamma + \delta_g\delta_g$')
        #pylab.axhline(0,color="k",alpha=0.5,ls=':')
        #pylab.axhline(5.0,color="darkviolet",lw=1, ls="--")
        #print "Plotting bounds line..."
        return filename     
