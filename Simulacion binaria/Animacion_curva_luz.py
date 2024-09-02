import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import lens_plot
from matplotlib import cm
from lenstronomy.Util import util
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LightModel.light_model import LightModel
from matplotlib import animation

#plt.rcParams.update(plt.rcParamsDefault)
#plt.rcParams['figure.dpi'] = 100
#plt.rcParams['savefig.dpi'] =100
plt.rcParams.update({'font.size': 12})



numpix=700
xx,yy = util.make_grid(numpix, deltapix=0.1)
theta_E_1= 4
theta_E_2=6

d=5

lens_model_list=['POINT_MASS','POINT_MASS']
lens=LensModel(lens_model_list)
lensext=LensModelExtensions(lens)
pointmass1_params={'theta_E':theta_E_1,'center_x':d, 'center_y':0} #thetaE en arcosegundos 0.00010090910723578977#
pointmass2_params={'theta_E':theta_E_2,'center_x':0, 'center_y':0}  
kwargs=[pointmass1_params,pointmass2_params]


xs,ys=lens.ray_shooting(x=xx,y=yy,kwargs=kwargs)
t0=70
tE=10
theta=-np.pi/8
y0=7
def SourcePos(t):
    p=(t-t0)/tE
    xsource=np.cos(theta)*p-np.sin(theta)*y0
    ysource=np.sin(theta)*p+np.cos(theta)*y0
    return(xsource,ysource)
    
criticalra,criticalde,causticsra,causticsd=lensext.critical_curve_caustics(kwargs,compute_window=30)
print(criticalra)

#SIMULAMOS
#ploteamos la critica
fig,ax=plt.subplots(2,1,figsize=(8,16))
for i in range(len(criticalra)):
    ax[0].plot(criticalra[i], criticalde[i], label=f'Crítica {i+1}')
ax[0].set_xlabel(r'$\theta_x$')
ax[0].set_ylabel(r'$\theta_y$',rotation=0)
ax[0].set_title('Críticas')
ax[0].legend(loc='lower right')
ax[0].grid(True)
ax[0].set(xlim=[-7.5,15],ylim=[-8,10])
ax[0].scatter([0],[0],s=50,c='red')
ax[0].scatter([d],[0],s=30,c='red')
ax[0].text(-1, -1, '$M_1$', fontsize = 12)
ax[0].text(d+0.2, -1, '$M_2$', fontsize = 12)


ax[1].grid(True)
ax[1].set_xlabel('tiempo [dias]')
ax[1].set_ylabel('aumento')
ax[1].set(xlim=[-5,205],ylim=[0,10])
t=np.linspace(-5,205,200)
lightcurve=[]
xc=[]
yc=[]
for i in range(len(t)):
    xcord=SourcePos(t[i])[0]
    ycord=SourcePos(t[i])[1]
    xc.append(xcord)
    yc.append(ycord)
    magt=lens.magnification(xcord,ycord,kwargs)
    lightcurve.append(magt)
    
#ploteo puntos iniciales
plt1=ax[0].plot(xc[0],yc[0],)[0]
plt2=ax[1].plot(t[0],lightcurve[0])[0]
plt.show()

def update(frame):
    # for each frame, update the data stored on each artist.
    tax = t[:frame]
    xcax = xc[:frame]
    ycax = yc[:frame]
    # update the line plot:
    plt2.set_xdata(tax)
    plt2.set_ydata(lightcurve[:frame])
    plt1.set_xdata(xcax)
    plt1.set_ydata(ycax)
    print(frame)
    return [plt1, plt2]

writergif = animation.PillowWriter(fps=30)
ani2 = animation.FuncAnimation(fig=fig, func=update, frames=len(t))
plt.show()
ani2.save('filename2.gif',writer=writergif)