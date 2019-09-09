import numpy as np
import fitsio as fi
import sys
import yaml

ingenting = lambda a : a
mult = lambda a : a*1e10
operations = {'sqrt' : np.sqrt, 'None' : ingenting, 'delete' : None, '*1e10': mult}

class Catalogue:
	def __init__(self, filename):
		self.old_file = fi.FITS(filename)
		self.new_file = fi.FITS(filename.replace('.fits', '-processed.fits'), 'rw')

	def initialise_dict(self):
		self.data = {}
		return None

	def write_column(self):
		self.new_file.write(self.data)
		return None

	def process(self, i, old, new, operation):

		# decide what we need to do with the column before we save it
		if operation in operations.keys():
			fn = operations[operation]
		else:
			print('Unrecognised operation',operation)
			print("Will assume you meant 'None'")
			fn = operations['None']

		# special case : don't keep this column
		if fn is None:
			return None

		# read in the column from the existing FITS file
		# and apply the operation
		old_array = self.old_file[i].read()
		data = old_array[old]
		data = fn(data)

		self.data[new] = data

		return None


config = yaml.load(open(sys.argv[-1],'rb'))
cat = Catalogue(sys.argv[-2])


for hdr in config.keys():
	columns = config[hdr]
	cat.initialise_dict()

	for oldname in columns.keys():
		newname, operation = columns[oldname].split()
		cat.process(hdr, oldname, newname, operation)

	cat.write_column()

	print(hdr)

cat.new_file.close()
print('Done all.')