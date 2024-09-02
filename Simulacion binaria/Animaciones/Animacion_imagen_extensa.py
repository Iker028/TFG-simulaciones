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
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
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



from lenstronomy.LightModel.light_model import LightModel
ligth = LightModel(light_model_list=['ELLIPSOID'])
t=np.linspace(-50,300,100)
f, ax = plt.subplots()
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
xpos= SourcePos(t[0])[0]
ypos=SourcePos(t[0])[1]
kwargs_light = [{'amp': 1, 'radius': 3, 'e1': 0, 'e2': 0, 'center_x': xpos, 'center_y': ypos}]
surface_brightness = ligth.surface_brightness(xs, ys, kwargs_light)
image = util.array2image(surface_brightness)
cax=ax.imshow(image)
f.colorbar(cax,label='Luminosidad (lx)')
print(len(image))

def init():
    cax.set_data(np.zeros((len(image), len(image))))
    print(cax)
    return [cax]

def animate(i):   
    xpos= SourcePos(t[i])[0]
    ypos=SourcePos(t[i])[1]
    kwargs_light = [{'amp': 1, 'radius': 3, 'e1': 0, 'e2': 0, 'center_x': xpos, 'center_y': ypos}]
    surface_brightness = ligth.surface_brightness(xs, ys, kwargs_light)
    image = util.array2image(surface_brightness)
    cax.set_data(image)
    print(i)
    return [cax]

ani = animation.FuncAnimation(f, animate, frames=len(t), init_func=init, blit=True)

# Display the animation
#plt.show()

# To save the animation, you can use:
# ani.save('matrix_animation.mp4', writer='ffmpeg')
writergif = animation.PillowWriter(fps=30)
ani.save('EXTENSA.gif',writer=writergif)

#writermp4 = animation.FFMpegWriter(fps=30)
#ani.save('filename.mp4', writer=writermp4)