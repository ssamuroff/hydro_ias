import pymysql as mdb
import numpy as np
from numpy.core.records import fromarrays
import fitsio as fi

snapshot=68

sqlserver='localhost'
user='flanusse'
password='mysqlpass'
dbname='mb2_hydro'
unix_socket='/home/rmandelb.proj/flanusse/mysql/mysql.sock'
db = mdb.connect(sqlserver, user, password, dbname, unix_socket=unix_socket)


# query for halo masses
sql = "SELECT mass,groupId FROM subfind_groups WHERE snapnum=%d;"%(snapshot)

print('Submitting query for halo masses')
cursor = db.cursor()
cursor.execute(sql)
results = fromarrays(np.array(cursor.fetchall()).squeeze().T,names="mass,groupId")


# query for galaxies
sql2 = "SELECT groupId,haloId FROM subfind_halos WHERE snapnum=%d;"%(snapshot)

print('Submitting query for galaxy IDs')
cursor = db.cursor()
cursor.execute(sql2)
results2 = fromarrays(np.array(cursor.fetchall()).squeeze().T,names="groupId,haloId")

ind = np.argsort(results['groupId'])

M = results['mass'][ind][results2['groupId']]

print('Saving output')
out = {'mass': M, 'gal_id': results2['haloId'], 'group_id': results2['groupId']}
outfits = fi.FITS('MBII_%d_halo_masses.fits'%snapshot, 'rw')
outfits.write(out)
outfits.close()

print('Done.')
