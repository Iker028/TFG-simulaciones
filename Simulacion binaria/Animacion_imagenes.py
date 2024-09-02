import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import lens_plot
from matplotlib import cm
from lenstronomy.Util import util
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LightModel.light_model import LightModel
from matplotlib import animation


plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.size': 12})



numpix=700
xx,yy = util.make_grid(numpix, deltapix=0.1)
theta_E_1= 4
theta_E_2=6

d=5

lens_model_list=['POINT_MASS','POINT_MASS']
lens=LensModel(lens_model_list)
lensext=LensModelExtensions(lens)
pointmass1_params={'theta_E':theta_E_1,'center_x':d, 'center_y':0}
pointmass2_params={'theta_E':theta_E_2,'center_x':0, 'center_y':0}  
kwargs=[pointmass1_params,pointmass2_params]


xs,ys=lens.ray_shooting(x=xx,y=yy,kwargs=kwargs)
t0=70
tE=10
theta=-np.pi*0.5/4
y0=0.5


def SourcePos(t):
    p=(t-t0)/tE
    xsource=np.cos(theta)*p-np.sin(theta)*y0
    ysource=np.sin(theta)*p+np.cos(theta)*y0
    return(xsource,ysource)
    
criticalra,criticalde,causticsra,causticsd=lensext.critical_curve_caustics(kwargs,compute_window=30)

t=np.linspace(-5,120,200)
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

#Posicion imagenes
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
solver=LensEquationSolver(lens)
ximl=[]
yiml=[]
def flatten(xss):
    return [x for xs in xss for x in xs]
for i in range(len(xc)):
    x,y=solver.image_position_lenstronomy(xc[i], yc[i], kwargs,search_window=50,num_iter_max=200)
    if len(x)%2!=0:
        ximl.append(x)
        yiml.append(y)
#ximl=flatten(ximl)
#yiml=flatten(yiml)

fig,ax=plt.subplots(1,1,figsize=(16,8))
for i in range(len(causticsra)):
    ax.plot(causticsra[i], causticsd[i], label=f'Caustica {i+1}')
ax.set_xlabel(r'$\beta_x$')
ax.set_ylabel(r'$\beta_y$',rotation=0)
ax.set_title('Caústicas')
ax.grid(True)
ax.set(xlim=[-11,15],ylim=[-10,10])
ax.scatter([0],[0],s=50,c='red')
ax.scatter([d],[0],s=30,c='red')
ax.text(-1, -1, '$M_1$', fontsize = 12)
ax.text(d+0.2, -1, '$M_2$', fontsize = 12)


#ploteo puntos iniciales
plt1=ax.plot(xc[0],yc[0],'*',markersize=12,label='fuente')[0]
plt2=ax.plot(ximl[0],yiml[0],'.',markersize=12,label='imágenes')[0]
ax.legend(loc='lower right')
def update2(frame):
    ax.clear()
    for i in range(len(causticsra)):
        ax.plot(causticsra[i], causticsd[i], label=f'Caustica {i+1}')
    ax.set_xlabel(r'$\beta_x$')
    ax.set_ylabel(r'$\beta_y$',rotation=0)
    ax.set_title('Caústicas')
    ax.grid(True)
    ax.set(xlim=[-11,15],ylim=[-10,10])
    ax.scatter([0],[0],s=50,c='red')
    ax.scatter([d],[0],s=30,c='red')
    ax.text(-1, -1, '$M_1$', fontsize = 12)
    ax.text(d+0.2, -1, '$M_2$', fontsize = 12)

    # for each frame, update the data stored on each artist.
    xcax = xc[frame]
    ycax = yc[frame]
    # update the line plot:s
    plt2=ax.plot(ximl[frame],yiml[frame],'.',markersize=12,label='imágenes',color='black')[0]
    plt1=ax.plot(xc[frame],yc[frame],'*',markersize=12,label='fuente',color='purple')[0]
    ax.legend(loc='lower right')
    print(frame)
    return [plt1, plt2]

writergif = animation.PillowWriter(fps=30)
ani3 = animation.FuncAnimation(fig=fig, func=update2, frames=len(t))
plt.show()
ani3.save('filename3.gif',writer=writergif)
