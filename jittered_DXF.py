'''
Generates a jittered 20*70 mm DXFs

##### Libraries required #####
To run this script you would require to install: ezdxf, numpy
To install these libraries do a "pip install" on each of them

##############################

'''

import ezdxf
import numpy as np

# Class that works on modelspace of a DXF: DXF_modelspace
class DXF_modelspace(object):
	def __init__(self, units= 'mm'): #res: resolution of the dxf, #apart: points drawn are how far apart from each other in "mm"?
		self.dwg= ezdxf.new('R2010')
		self.msp= self.dwg.modelspace()
		self.units= units

	def __repr__(self):
		return repr(self.msp)

	def write_points(self, res= (20, 17), apart= 0.5):
		self.res= (res[0], res[1])
		self.apart= apart

		arr= {"X": np.arange(0, -res[0]-apart, step= -apart),
			"Y": np.arange(0, res[1]+apart, step= apart)}

		point_placements= np.random.randint(0, 2, size= len(arr['X'])*len(arr['Y']))

		iterator= 0
		for x in arr['X']:
			for y in arr['Y']:
				if point_placements[iterator]== 1:
					if self.units == 'mm':
						self.msp.add_point((x/25.4,y/25.4))
					elif self.units == 'inch':
						self.msp.add_point((x, y))

				iterator += 1

	def save_dxf(self, save_as= "point.dxf"):
		self.dwg.saveas("point.dxf")
	



dxf= DXF_modelspace(units = 'mm') # for now only supports 'mm' and 'inch'
dxf.write_points(res= (20, 17), apart= 0.5) #res: resolution of the dxf, #apart: how far apart from each other are the points drawn(in "mm" or "inch")?
dxf.save_dxf(save_as= "point.dxf")