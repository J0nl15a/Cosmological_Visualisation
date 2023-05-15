#!/usr/bin/env python
# coding: utf-8

# In[14]:


#import numpy as np
import pandas as pd
#from virgodb import VirgoDB
import eagleSqlTools as sql


# In[15]:


# Setup the connection
user = 'nrr565'
password = 'fr53DS79'
url = 'http://galaxy-catalogue.dur.ac.uk:8080/Eagle/'
#vdb = VirgoDB(user, password, url)
vdb = sql.connect(user, password=password)


# In[16]:


query = ''' SELECT        
            fof.Group_M_Crit200 as halo_mass,
            ap.Mass_Star as stellar_mass,
            sub.CentreOfPotential_x as COP_x, 
            sub.CentreOfPotential_y as COP_y, 
            sub.CentreOfPotential_z as COP_z, 
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


# In[17]:


#result = vdb.execute_query(query)
result = sql.execute_query(vdb, query)
df = pd.DataFrame(result)
print(df)


# In[18]:


sub_sample = df.loc[(df.SubGroupNumber == 0) & (df.halo_mass >= 1e12)].sort_values(by='halo_mass')
print(sub_sample.head())


# In[19]:


#h = 0.6777
positions = pd.DataFrame()
positions['M_stellar'] = sub_sample.loc[sub_sample.GroupNumber == 3].stellar_mass.values
positions['x'] = sub_sample.loc[sub_sample.GroupNumber == 3].COP_x.values
positions['y'] = sub_sample.loc[sub_sample.GroupNumber == 3].COP_y.values
positions['z'] = sub_sample.loc[sub_sample.GroupNumber == 3].COP_z.values
#positions


# In[20]:


positions.to_csv('EAGLE_halo_positions.csv')


# In[ ]:




