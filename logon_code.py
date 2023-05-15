import pymssql
import pandas as pd

user = 'nrr565'
password = 'fr53DS79'
server= 'virgodb'

def connect():
    """Connect to Millennium database and return a connection object"""
    connection = pymssql.connect(server=server, user=user, password=password, database=user+'_db')
    return connection

def query(query, connection):
    """
    Execute an SQL query and stores the result as a data frame.
    """
    #store = pd.HDFStore('sql_store.h5')
    df =  pd.read_sql_query(query, connection)
    return df

if __name__ == "__main__":

    query1 = 'select top 10 GalaxyID, SnapNum, CentreOfMass_x, CentreOfMass_y, CentreOfMass_z from Eagle..RefL0025N0376_Subhalo'
    query2 = ''' SELECT        
            fof.Group_M_Crit200 as halo_mass,
            ap.Mass_Star as stellar_mass,
            sub.CentreOfMass_x as COM_x, 
            sub.CentreOfMass_y as COM_y, 
            sub.CentreOfMass_z as COM_z, 
            sub.GroupNumber as GroupNumber,
            sub.SubGroupNumber as SubGroupNumber
            FROM 
            Eagle..RefL0012N0188_FOF as fof,
            Eagle..RefL0012N0188_SubHalo as sub,
            Eagle..RefL0012N0188_Aperture as ap
            WHERE  
            sub.SnapNum = 28
            and sub.GalaxyID = ap.GalaxyID
            and sub.GroupID = fof.GroupID      
            and ap.ApertureSize = 30   
            and ap.Mass_Star > 1e7 '''

    engine = connect()
    df = query(query2, engine)
    print(df)
