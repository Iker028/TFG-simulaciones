import numpy as np
import matplotlib.pyplot as plt
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import param_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util import util
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Data.psf import PSF
#rsadsfasdfadg
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
# observation parameters:
background_rms = 0.5 # background noise per pixel
exp_time = 100 # exposure time (arbitrary units)
numPix = 200 # number of pixels
deltaPix = 0.05 # pixel size in arcsec

# PSF specification
fwhm = 0.1 # PSF FWHM
kwargs_data = sim_util.data_configure_simple(numPix, deltaPix,
exp_time,
background_rms)
data_class = ImageData(**kwargs_data)
kwargs_psf = {"psf_type": "GAUSSIAN",
"fwhm": fwhm,
"pixel_size": deltaPix,
"truncation": 5}
psf_class = PSF(**kwargs_psf)


# lens parameters
zl=0.3 # lens redshift
zs=1.5 # source redshift

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
lens_cosmo = LensCosmo(z_lens=zl, z_source=zs, cosmo=cosmo)

R_sersic = 0.5
n_sersic = 4
m_star=10**(12) #Masa en masas solares
k_eff = lens_cosmo.sersic_m_star2k_eff(m_star=m_star, R_sersic=R_sersic, n_sersic=n_sersic)
m = lens_cosmo.sersic_k_eff2m_star(k_eff=3.1790633, R_sersic=0.52854867, n_sersic=4.14999)
print(k_eff)
print(m)
# lens Einstein radius
from astropy.cosmology import FlatLambdaCDM
co = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.constants import c, G
dl=co.angular_diameter_distance(zl)
ds=co.angular_diameter_distance(zs)
dls=co.angular_diameter_distance_z1z2(zl,zs)
# compute the Einstein radius
lens_model_list = ["SERSIC_ELLIPSE_POTENTIAL"]
lens_model_class = LensModel(lens_model_list=lens_model_list)
kwargs_sie = {"k_eff": k_eff,'R_sersic':R_sersic,'n_sersic':n_sersic,
"center_x": 0,
"center_y":0,'e1':0.1,'e2':0.1}
kwargs_lens = [kwargs_sie]


# create the light model for the lens (SERSIC_ELLIPSE)
lens_light_model_list = ["SERSIC_ELLIPSE"]
kwargs_sersic = {"amp":400, # flux of the lens (arbitrary units)
"R_sersic": 2 ,# effective radius
"n_sersic": 4, # sersic index
"center_x": 0., # x-coordinate
"center_y": 0., # y-coordinate
"e1": 0.1,
"e2": 0.1}
kwargs_lens_light = [kwargs_sersic]
lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
# create the light model for the source (SERSIC_ELLIPSE)
source_model_list = ["SERSIC_ELLIPSE"]



# set the position of the source
ra_source, dec_source = 0.7,0
kwargs_sersic_ellipse = {"amp": 400.,
"R_sersic": 2.5,
"n_sersic": 3,
"center_x": ra_source,
"center_y": dec_source,
"e1": 0.1,
"e2": 0.3}
kwargs_source = [kwargs_sersic_ellipse]
source_model_class = LightModel(light_model_list=source_model_list)


# solve the lens equation and find the image positions
# using the LensEquationSolver class of Lenstronomy.
lensEquationSolver = LensEquationSolver(lens_model_class)
x_image, y_image = lensEquationSolver.image_position_from_source(ra_source,dec_source,kwargs_lens, min_distance=deltaPix,
search_window=numPix * deltaPix,
precision_limit=1e-10, num_iter_max=100,
arrival_time_sort=True,
initial_guess_cut=True,
verbose=False,
x_center=0,
y_center=0,
num_random=0,
non_linear=False,
magnification_limit=None)
# compute lensing magnification at image positions
mag = lens_model_class.magnification(x_image, y_image,
kwargs=kwargs_lens)
mag = np.abs(mag) # ignore the sign of the magnification
# perturb observed magnification due to e.g. micro-lensing
# the noise is generated from a normal distribution
# with mean "mag" and standard deviation 0.5
mag_pert = np.random.normal(mag, 0.5, len(mag))


# quasar position in the lens plane
kwargs_ps = [{"ra_image": x_image,
"dec_image": y_image,
"point_amp": 100000}]
point_source_list = ["LENSED_POSITION"]
point_source_class =PointSource(point_source_type_list=point_source_list,
fixed_magnification_list=[False])
# create the simulated observation of lens and (lensed)
# source
kwargs_numerics = {"supersampling_factor": 1,
"supersampling_convolution": False}
# imageModel includes the details of the instrument, psf, lens,
# and source models
imageModel = ImageModel(data_class, psf_class, lens_model_class,source_model_class,lens_light_model_class,point_source_class,
                        kwargs_numerics=kwargs_numerics)
# now, the simulated image is saved in image_sim
image_sim = imageModel.image(kwargs_lens, kwargs_source,kwargs_lens_light, kwargs_ps)
# add noise and background
poisson = image_util.add_poisson(image_sim, exp_time=exp_time)
bkg = image_util.add_background(image_sim, sigma_bkd=background_rms)
image_sim = image_sim + bkg + poisson

data=np.log10(np.abs(image_sim))
img=plt.imshow(data,cmap='viridis',vmin=data.min(),vmax=data.max())
plt.xlabel(r'$\theta_x$')
plt.ylabel(r'$\theta_y$ ',rotation=0)
plt.xticks([0,50,100,150,200],[-5,-2.5,0,2.5,5])
plt.yticks([0,50,100,150,200],[5,2.5,0,-2.5,-5])
#cbar=plt.colorbar(img)
#cbar.ax.yaxis.set_ticks([])
#cbar.set_label('Luminosidad',rotation=90)
plt.show()


# quasar position in the source plane
kwargs_ps = [{"ra_image": [ra_source],
"dec_image": [dec_source],
"point_amp": [100000]}]
point_source_list2 = ["UNLENSED"]
point_source_class2 =PointSource(point_source_type_list=point_source_list2,
fixed_magnification_list=[False])


imageModel2 = ImageModel(data_class, psf_class,lens_light_model_class=source_model_class ,point_source_class=point_source_class2,
                         kwargs_numerics=kwargs_numerics)

image2 = imageModel2.image(kwargs_lens_light=kwargs_source, kwargs_ps=kwargs_ps)
poisson = image_util.add_poisson(image2, exp_time=exp_time)
bkg = image_util.add_background(image2, sigma_bkd=background_rms)
image2 = image2 + bkg + poisson

data=np.log10(np.abs(image2))
img=plt.imshow(data,cmap='viridis',vmin=data.min(),vmax=data.max())
plt.xlabel(r'$\beta_x$')
plt.ylabel(r'$\beta_y$ ',rotation=0)
plt.xticks([0,50,100,150,200],[-5,-2.5,0,2.5,5])
plt.yticks([0,50,100,150,200],[-5,-2.5,0,2.5,5])
#cbar=plt.colorbar(img)
#cbar.ax.yaxis.set_ticks([])
#cbar.set_label('Luminosidad',rotation=90)
plt.show()


#Fitear la lente de la imagen lensada
mu, sigma = 0, 0.015 # mean and standard deviation
s1 = np.random.normal(mu, sigma, len(x_image))
s2 = np.random.normal(mu, sigma, len(y_image))
x1_ima=x_image #+s1 #Tomamos las imágenes del quasar
x2_ima=y_image #+s2 

plt.plot(x1_ima,x2_ima,'.')
plt.show()

#hacemos un guess 
def guess(kwargs,x1im,x2im):
    #defino la lente
    lens_model_listguess = ["SERSIC_ELLIPSE_POTENTIAL"]
    lens_model_classguess = LensModel(lens_model_list=lens_model_listguess)
    #calculo posición de la fuente
    xs,ys=lens_model_classguess.ray_shooting(x=x1im,y=x2im,kwargs=kwargs)
    return xs,ys
#en general encontraremos n:=len(xs) posiciones de la fuente; asumimos que la correcta es la media de estas n posiciones
#calculo media
def media(x,y):
    x_media=0
    y_media=0
    for i in range(len(x)):
        x_media+=x[i]
        y_media+=y[i]
    x_media=x_media/len(x)
    y_media=y_media/(len(x))
    return x_media,y_media


#hacemos el ajuste
import lmfit
kwargs_sie = {"k_eff": k_eff,'R_sersic':R_sersic,'n_sersic':n_sersic,
"center_x": 0,
"center_y":0,'e1':0.1,'e2':0.1}

p = lmfit.Parameters()
p.add_many(('m', 1.28, True, 0.1, 100),
('R_sersic', 0.6, True, 0.1, 10.), #hay una relación entre n y R_sersic 
('n_sersic', 3.7, True, 1, 8.),
('e1',0.15,True,0,1),
('e2',0.15,True,0,1),
('center_x',0.,False,-2,2),
('center_y',0.,False,-2,2))
m=lens_cosmo.sersic_k_eff2m_star(3.3, 0.6, 3.7)
print(m)

#definimos función de coste
def coste(p,x1_ima,x2_ima,sigma_ima):
    #kwargs= [{"k_eff": p['k_eff'].value,'R_sersic':p['R_sersic'].value,'n_sersic':p['n_sersic'].value, "center_x": p['center_x'].value,
              #"center_y":p['center_y'].value,
              #'e1':p['e1'].value,'e2':p['e2'].value}]
    k_eff=lens_cosmo.sersic_m_star2k_eff(m_star=p['m'].value*10**(12), R_sersic=p['R_sersic'].value, n_sersic=p['n_sersic'].value)
    kwargs= [{"k_eff": k_eff,'R_sersic':p['R_sersic'].value,'n_sersic':p['n_sersic'].value, "center_x": p['center_x'].value,
              "center_y":p['center_y'].value,
              'e1':p['e1'].value,'e2':p['e2'].value}]
    xs,ys=guess(kwargs,x1_ima,x2_ima)
    x_media,y_media=media(xs,ys)
    x_image, y_image = lensEquationSolver.image_position_from_source(x_media,y_media,kwargs,
    search_window=numPix*deltaPix*2,
    precision_limit=1e-10, num_iter_max=100,
    arrival_time_sort=True,
    initial_guess_cut=True,
    verbose=False,
    x_center=0,
    y_center=0,
    num_random=0,
    non_linear=False,
    magnification_limit=None)
    imod=[]
    for i in range(len(x1_ima)):
        d=(x1_ima[i]-x_image)**2+(x2_ima[i]-y_image)**2
        imod.append(np.argmin(d))
    res1=(x_image[imod]-x1_ima)/sigma_ima
    res2=(y_image[imod]-x2_ima)/sigma_ima
    return res1, res2
#defino funcion xi^2
def chi2(p,x1_ima,x2_ima):
    d1,d2=coste(p,x1_ima,x2_ima,0.015)
    return np.sqrt(d1**2+d2**2)
# minimize the cost function (here using the ’powell’ method)
sigma_ima=0.015
mi = lmfit.minimize(coste, p, method='lq', args=(x1_ima,x2_ima,sigma_ima))
lmfit.report_fit(mi.params)


res = lmfit.minimize(chi2, method='emcee',
    nan_policy='omit',
    params=mi.params,
    float_behavior='chi2',
    progress=True,args=(x1_ima,x2_ima))


lmfit.report_fit(res.params)

'''
import corner
figure = corner.corner(res.flatchain,
labels=[r"$\sigma_0$", r"$R_a$",
r"$R_s$",r"$e_1$",r"e_2",r"$center_x$",r"$center_y$"],
truths=list(res.params.valuesdict().values()),
quantiles=[0.16,0.84],
show_titles=True,
title_kwargs={"fontsize": 14},
label_kwargs={"fontsize": 14})
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(axis='both', labelsize=12)
'''


#####################################################
#
k_eff=3.3
R_sersic=0.6
n_sersic=3.7
kwargs = [{"k_eff": k_eff,'R_sersic':R_sersic,'n_sersic':n_sersic,
"center_x": 0,
"center_y":0,'e1':0.15,'e2':0.15}]
lensext=LensModelExtensions(lens_model_class)
lensEquationSolver = LensEquationSolver(lens_model_class)
criticalra,criticalde,causticsra,causticsd=lensext.critical_curve_caustics(kwargs,compute_window=30)

fig,ax=plt.subplots(1,1,figsize=(12,12))
plt.legend(loc='lower right')
plt.xlabel(r'$\theta_x$')
plt.ylabel(r'$\theta_y$',rotation=0)
x_image, y_image = lensEquationSolver.image_position_from_source(ra_source,dec_source,kwargs, min_distance=deltaPix,
search_window=numPix * deltaPix,
precision_limit=1e-10, num_iter_max=100,
arrival_time_sort=True,
initial_guess_cut=True,
verbose=False,
x_center=0,
y_center=0,
num_random=0,
non_linear=False,
magnification_limit=None)
for i in range(len(x2_ima)):
    x2_ima[i]=-x2_ima[i]
    y_image[i]=-y_image[i]
    plt.plot([x1_ima[i],x_image[i]],[x2_ima[i],y_image[i]],color='black')
for i in range(len(causticsra)):
    #causticsd[i]=-causticsd[i]
    plt.plot(causticsra[i], causticsd[i], label=f'Caústica {i+1}')
plt.plot(x_image,y_image,'.',markersize=12,label='imágenes del modelo')
plt.plot(x1_ima,x2_ima,'.',markersize=12,label='imágenes observadas')
plt.plot(ra_source,dec_source,'*',color='pink',markersize=10,label='quasar fuente')
plt.legend(loc='lower right')
#plt.text(0.715, -3, r'$\boldmath{r_i}$', fontsize = 10)
plt.grid(True)
plt.show()

#####################################################
k_eff=3.17906338
R_sersic=0.52854867
n_sersic=4.14999480 
kwargs = [{"k_eff": k_eff,'R_sersic':R_sersic,'n_sersic':n_sersic,
"center_x": 0,
"center_y":0,'e1':0.09914618,'e2':0.09904297}]
lensext=LensModelExtensions(lens_model_class)
lensEquationSolver = LensEquationSolver(lens_model_class)
criticalra,criticalde,causticsra,causticsd=lensext.critical_curve_caustics(kwargs,compute_window=30)

fig,ax=plt.subplots(1,1,figsize=(12,12))
plt.xlabel(r'$\theta_x$')
plt.ylabel(r'$\theta_y$',rotation=0)
x_image, y_image = lensEquationSolver.image_position_from_source(ra_source,dec_source,kwargs, min_distance=deltaPix,
search_window=numPix * deltaPix,
precision_limit=1e-10, num_iter_max=100,
arrival_time_sort=True,
initial_guess_cut=True,
verbose=False,
x_center=0,
y_center=0,
num_random=0,
non_linear=False,
magnification_limit=None)
for i in range(len(x2_ima)):
   y_image[i]=-y_image[i]
for i in range(len(causticsra)):
    plt.plot(causticsra[i], causticsd[i], label=f'Caústica {i+1}')
plt.plot(x_image,y_image,'.',markersize=12,label='imágenes del modelo')
plt.plot(x1_ima,x2_ima,'.',markersize=12,label='imágenes observadas')
plt.plot(ra_source,dec_source,'*',color='pink',markersize=10,label='quasar fuente')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()