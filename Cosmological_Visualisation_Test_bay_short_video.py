#!/usr/bin/env python
# coding: utf-8

# In[1]:

#import matplotlib
#matplotlib.use('Agg')
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sphviewer.tools import QuickView
from matplotlib.colors import LogNorm
#from virgodb import VirgoDB
import pyread_eagle as read_eagle
#import eagleSqlTools as sql
import pymssql
from logon_code import connect, query
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
#plt.rcParams["text.usetex"] = True
np.warnings.filterwarnings('ignore')
import sys

# In[2]:


# Setup the connection
user = 'nrr565'
password = 'fr53DS79'
url = 'http://galaxy-catalogue.dur.ac.uk:8080/Eagle/'
server = 'virgodb'
#vdb = VirgoDB(user, password, url)
#vdb = sql.connect(user, password=password)
engine = connect()


# In[3]:


sqlquery = ''' SELECT        
            fof.Group_M_Crit200 as halo_mass,
            ap.Mass_Star as stellar_mass,
            sub.CentreOfMass_x as COM_x, 
            sub.CentreOfMass_y as COM_y, 
            sub.CentreOfMass_z as COM_z, 
            sub.GroupNumber as GroupNumber,
            sub.SubGroupNumber as SubGroupNumber
            FROM 
            RefL0012N0188_FOF as fof,
            RefL0012N0188_SubHalo as sub,
            RefL0012N0188_Aperture as ap
            WHERE  
            sub.SnapNum = 28
            and sub.GalaxyID = ap.GalaxyID
            and sub.GroupID = fof.GroupID      
            and ap.ApertureSize = 30   
            and ap.Mass_Star > 1e7 '''


# In[4]:


#result = vdb.execute_query(sqlquery)
#result = sql.execute_query(vdb, sqlquery)
#df = pd.DataFrame(result)
df = query(sqlquery, engine)
print(df)
#sys.exit()
# In[5]:


boxsize = 12
sub_sample = df.loc[(df.SubGroupNumber == 0) & (df.halo_mass >= 1e12)].sort_values(by='halo_mass')


# In[6]:


h = 0.6777
size = 10 * h
x = sub_sample.loc[sub_sample.GroupNumber == 3].COM_x.values * h
y = sub_sample.loc[sub_sample.GroupNumber == 3].COM_y.values * h
z = sub_sample.loc[sub_sample.GroupNumber == 3].COM_z.values * h
extent = [-size / h, size / h, -size / h, size / h]


# In[7]:


snap23 = read_eagle.EagleSnapshot("/cosma7/data/Eagle/DataRelease/L0012N0188/PE/REF_COSMA5/data/snapshot_023_z000p503/snap_023_z000p503.0.hdf5")
snap24 = read_eagle.EagleSnapshot("/cosma7/data/Eagle/DataRelease/L0012N0188/PE/REF_COSMA5/data/snapshot_024_z000p366/snap_024_z000p366.0.hdf5")
snap25 = read_eagle.EagleSnapshot("/cosma7/data/Eagle/DataRelease/L0012N0188/PE/REF_COSMA5/data/snapshot_025_z000p271/snap_025_z000p271.0.hdf5")
snap26 = read_eagle.EagleSnapshot("/cosma7/data/Eagle/DataRelease/L0012N0188/PE/REF_COSMA5/data/snapshot_026_z000p183/snap_026_z000p183.0.hdf5")
snap27 = read_eagle.EagleSnapshot("/cosma7/data/Eagle/DataRelease/L0012N0188/PE/REF_COSMA5/data/snapshot_027_z000p101/snap_027_z000p101.0.hdf5")
snap28 = read_eagle.EagleSnapshot("/cosma7/data/Eagle/DataRelease/L0012N0188/PE/REF_COSMA5/data/snapshot_028_z000p000/snap_028_z000p000.0.hdf5")
snap23.select_region(x - size, x + size, y - size, y + size, z - size, z + size)
snap24.select_region(x - size, x + size, y - size, y + size, z - size, z + size)
snap25.select_region(x - size, x + size, y - size, y + size, z - size, z + size)
snap26.select_region(x - size, x + size, y - size, y + size, z - size, z + size)
snap27.select_region(x - size, x + size, y - size, y + size, z - size, z + size)
snap28.select_region(x - size, x + size, y - size, y + size, z - size, z + size)


# In[27]:


pos_dm_23 = snap23.read_dataset(1, "Coordinates") / h
dm = pd.DataFrame(pos_dm_23, columns=['x','y','z'])
print(dm)
pos_gas_23 = snap23.read_dataset(0, "Coordinates") / h
gas_23 = pd.DataFrame(pos_gas_23, columns=['x','y','z'])
gas_23['mass'] = snap23.read_dataset(0, "Mass") / h * 1e10
gas_23['hsml'] = snap23.read_dataset(0, "SmoothingLength") / h
pos_gas_24 = snap24.read_dataset(0, "Coordinates") / h
gas_24 = pd.DataFrame(pos_gas_24, columns=['x','y','z'])
gas_24['mass'] = snap24.read_dataset(0, "Mass") / h * 1e10
gas_24['hsml'] = snap24.read_dataset(0, "SmoothingLength") / h
pos_gas_25 = snap25.read_dataset(0, "Coordinates") / h
gas_25 = pd.DataFrame(pos_gas_25, columns=['x','y','z'])
gas_25['mass'] = snap25.read_dataset(0, "Mass") / h * 1e10
gas_25['hsml'] = snap25.read_dataset(0, "SmoothingLength") / h
pos_gas_26 = snap26.read_dataset(0, "Coordinates") / h
gas_26 = pd.DataFrame(pos_gas_26, columns=['x','y','z'])
gas_26['mass'] = snap26.read_dataset(0, "Mass") / h * 1e10
gas_26['hsml'] = snap26.read_dataset(0, "SmoothingLength") / h
pos_gas_27 = snap27.read_dataset(0, "Coordinates") / h
gas_27 = pd.DataFrame(pos_gas_27, columns=['x','y','z'])
gas_27['mass'] = snap27.read_dataset(0, "Mass") / h * 1e10
gas_27['hsml'] = snap27.read_dataset(0, "SmoothingLength") / h
pos_gas_28 = snap28.read_dataset(0, "Coordinates") / h
gas_28 = pd.DataFrame(pos_gas_28, columns=['x','y','z'])
gas_28['mass'] = snap28.read_dataset(0, "Mass") / h * 1e10
gas_28['hsml'] = snap28.read_dataset(0, "SmoothingLength") / h
pos_stars_28 = snap28.read_dataset(4, "Coordinates") / h
stars_28 = pd.DataFrame(pos_stars_28, columns=['x','y','z'])
stars_28['mass'] = snap28.read_dataset(4, "Mass") / h * 1e10
stars_28['hsml'] = (snap28.read_dataset(4, "SmoothingLength") / h)/2


# In[29]:


point_one = pd.Series(.01, index=range(len(stars_28)))
stars_28['hsml'] = stars_28.mask(stars_28 > .01, point_one, axis=0)


# In[ ]:


def img_norm(img, boxsize):
    img_norm = img / np.power(boxsize / np.shape(img)[0], 2)
    img_norm[np.where(img_norm < 1e-8)] = 1e-8
    img_log = np.log10(img_norm)
    return img_log


# In[ ]:


def rotate(phi_min, phi_max, field, boxsize, frame, colour, extent):
    
    for p in range(phi_min, phi_max+1): 
    
        if field == 'dm':
            qv = QuickView(dm[['x', 'y', 'z']].values, r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=p, t=0, extent = extent)
        else:
            qv = QuickView(field[['x', 'y', 'z']].values, mass=field.mass.to_numpy(), hsml=field.hsml.to_numpy(), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=p, t=0, extent = extent)

        img = qv.get_image()
        img_log = img_norm(img, boxsize)
        plt.imsave('./Video_test1/vid_test1_%d.png'%frame, img_log, vmin=14, vmax=20, cmap=colour, origin='lower')
        frame += 1
        
        if p != phi_max:
            pass
        elif p == phi_max:
            return phi_max, frame


# In[ ]:


def zoom(zoom_start, zoom_stop, zoom_step, field, boxsize, frame, colour, phi=0):
    
    zoom_mag = np.arange(zoom_stop, zoom_start+zoom_step, zoom_step)
    zoom_mag = zoom_mag[::-1]
    
    for m in zoom_mag: 
    
        extent = [-m, m, -m, m]
        if field == 'dm':
            qv = QuickView(dm[['x', 'y', 'z']].values, r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, extent = extent, p=phi)
        else:
            qv = QuickView(field[['x', 'y', 'z']].values, mass=field.mass.to_numpy(), hsml=field.hsml.to_numpy(), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent = extent)

        img = qv.get_image()
        img_log = img_norm(img, boxsize)
        plt.imsave('./Video_test1/vid_test1_%d.png'%frame, img_log, vmin=14, vmax=20, cmap=colour, origin='lower')
        frame += 1
        
        if m != zoom_stop:
            pass
        elif m == zoom_stop:
            return extent, frame


# In[ ]:


def change(field1, field2, steps, boxsize, frame, colour1, colour2, extent, phi=0):
    u = np.linspace(1, 0, steps+1)

    if field1 == 'dm':
        qv1 = QuickView(dm[['x', 'y', 'z']].values, r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
    else:
        qv1 = QuickView(field1[['x', 'y', 'z']].values, mass=field1.mass.to_numpy(), hsml=field1.hsml.to_numpy(), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
    
    if field2 == 'dm':
        qv2 = QuickView(dm[['x', 'y', 'z']].values, r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
    else:
        qv2 = QuickView(field2[['x', 'y', 'z']].values, mass=field2.mass.to_numpy(), hsml=field2.hsml.to_numpy(), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
    
    img1 = qv1.get_image()
    ext1 = qv1.get_extent()
    img_log1 = img_norm(img1, boxsize)
    img2 = qv2.get_image()
    ext2 = qv2.get_extent()
    img_log2 = img_norm(img2, boxsize)
    
    for i in range(0, steps+1):
        fig = plt.figure(i, figsize=(10,10), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        alpha1 = u[i]
        alpha2 = 1-u[i]
    
        a=plt.imshow(img_log1, extent=ext1, cmap=colour1, origin='lower', alpha=alpha1, aspect='auto')
        b=plt.imshow(img_log2, extent=ext2, cmap=colour2, origin='lower', alpha=alpha2, aspect='auto')
        plt.savefig('./Video_test1/vid_test1_%d.png'%frame)
        plt.clf()
        frame += 1
        
        if i != steps:
            pass
        elif i == steps:
            return frame


# In[ ]:


def fade(field, steps, boxsize, frame, colour, fade_in = False, fade_out = False):
    if fade_in == True:
        u = np.linspace(0, 1, steps+1)
    elif fade_out == True:
        u = np.linspace(1, 0, steps+1)

    if field == 'dm':
        qv = QuickView(dm[['x', 'y', 'z']].values, r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000)
    else:
        qv = QuickView(field[['x', 'y', 'z']].values, mass=field.mass.to_numpy(), hsml=field.hsml.to_numpy(), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000)
    img = qv.get_image()
    ext = qv.get_extent()
    img_log = img_norm(img, boxsize)
    
    for i in range(0, steps+1):
        fig = plt.figure(i, figsize=(10,10), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        alpha = u[i]
    
        a=plt.imshow(img_log, extent=ext, cmap=colour, origin='lower', alpha=alpha, aspect='auto')
        plt.savefig('./Video_test1/vid_test1_%d.png'%frame)
        #plt.clf()
        frame += 1
        
        if i != steps:
            pass
        elif i == steps:
            return frame


# In[ ]:


def evolve(fields, boxsize, frame, colour, extent, phi=0):
    
    for f in fields:
        if field == 'dm':
            qv = QuickView(dm[['x', 'y', 'z']].values, r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
        else:
            qv = QuickView(f[['x', 'y', 'z']].values, mass=f.mass.to_numpy(), hsml=f.hsml.to_numpy(), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
        img = qv.get_image()
        img_log = img_norm(img, boxsize)
        plt.imsave('./Video_test1/vid_test1_%d.png'%frame, img_log, vmin=14, vmax=20, cmap=colour, origin='lower')
        frame += 1
        
    return frame


# In[ ]:


#Start with DM fade in
frame = fade('dm', 30, boxsize, 0, 'viridis', fade_in = True)
print('Done with fade in')
#DM zoom x2.5
extent, frame = zoom(1, 2.5, 0.05, 'dm', boxsize, frame, 'viridis')
print('Done with zoom in x2.5')
#Rotate DM by pi
phi, frame = rotate(0, 180, 'dm', boxsize, frame, 'viridis', extent)
print('Done with rotate by pi')
#Change from DM to gas
frame = change('dm', gas_23, 30, boxsize, frame, 'viridis', 'plasma', extent, phi)
print('Done with change to gas')
#Evolve gas through z
frame = evolve((gas_23, gas_24, gas_25, gas_26, gas_27, gas_28), boxsize, frame, 'plasma', extent, phi)
print('Done with evolve')
#Rotate gas by pi/2
phi, frame = rotate(180, 270, gas_28, boxsize, frame, 'plasma', extent)
print('Done with rotate by pi/2')
#Gas zoom x2
extent, frame = zoom(2.5, 5, 0.05, gas_28, boxsize, frame, 'plasma', phi)
print('Done with zoom in x2')
#Change from gas to stars
frame = change(gas_28, stars_28, 30, boxsize, frame, 'plasma', 'bone', extent, phi)
print('Done with change to stars')
#Rotate stars by pi/2
phi, frame = rotate(270, 360, stars_28, boxsize, frame, 'bone', extent)
print('Done with rotate by pi/2')
#Stars zoom x(-4)
extent, frame = zoom(5, 1.25, 0.05, stars_28, boxsize, frame, 'bone', phi)
print('Done with zoom out x4')
#Fade out on stars
frame = fade(stars_28, 30, boxsize, frame, 'bone', fade_out = True)
print('Done with fade out')
print(frame)


# In[ ]:




