import numpy as np
from pyhyp import pyHyp
import os
import math

# read airfoils names
namelist =[]
f = open('not_done_yet.txt','r')
while True:
  filename=f.readline()
  if filename:
    namelist.append(filename.rstrip())
  else:
    break
f.close()

#os.system('mkdir test_cgns')
#os.system('mkdir test_xyz')

for iair in range(len(namelist)):
	data = np.loadtxt('/home/novermor/SARA/data_set_OFFICIAL/sampling/picked_uiuc/' + namelist[iair])
	x = data[:, 0].copy()
	y = data[:, 1].copy()
	ndim = x.shape[0]

	airfoil3d = np.zeros((ndim, 2, 3))
	for j in range(2):
	    airfoil3d[:, j, 0] = x[:]
	    airfoil3d[:, j, 1] = y[:]
	# set the z value on two sides to 0 and 1
	airfoil3d[:, 0, 2] = 0.0
	airfoil3d[:, 1, 2] = 1.0
	# write out plot3d
	P3D_fname = '/home/novermor/SARA/data_set_OFFICIAL/mesh/test_xyz/' + namelist[iair] + '.xyz'

	with open(P3D_fname, "w") as p3d:
		p3d.write(str(1) + "\n")
		p3d.write(str(ndim) + " " + str(2) + " " + str(1) + "\n")
		for ell in range(3):
			for j in range(2):
		    		for i in range(ndim):
		    			p3d.write("%.15f\n" % (airfoil3d[i, j, ell]))

	options = {
	    # ---------------------------
	    #        Input Parameters
	    # ---------------------------
	    "inputFile": P3D_fname,
	    "unattachedEdgesAreSymmetry": False,
	    "outerFaceBC": "farfield",
	    "autoConnect": True,
	    "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
	    "families": "wall",
	    
	    # ---------------------------
	    #        Grid Parameters
	    # ---------------------------
	    "N": 129,						#N is equal to the off-wall levels to march 			
	    "s0": 2e-6,						#Initial off-wall spacing of grid (thickness of viscous layer it must be imposed considering y+= 1)
	    "marchDist": 100.0,					#cord-distance of the farfield
	    # ---------------------------
	    #   Pseudo Grid Parameters
	    # ---------------------------
	    "ps0": -1.0,					#Initial pseudo off-wall spacing. This spacing **must** be less than or equal to ``s0``. This is the actual spacing the hyperbolic scheme uses
	    "pGridRatio": -1.0,					#The ratio between successive levels in the pseudo grid. This will be typically somewhere between ~1.05 for large grids to 1.2 for small grids.
	    "cMax": 3.0,					#The maximum permissible ratio of the step in the marching direction to the length of any in-plane edge
	    # ---------------------------
	    #   Smoothing parameters
	    # ---------------------------
	    "epsE": 1.0,
	    "epsI": 2.0,
	    "theta": 3.0,
	    "volCoef": 0.25,
	    "volBlend": 0.0001,
	    "volSmoothIter": 100,
	}
	namelist[iair]= namelist[iair].replace('.dat','.cgns')
	hyp = pyHyp(options=options)
	hyp.run()
	hyp.writeCGNS('/home/novermor/SARA/data_set_OFFICIAL/mesh/test_cgns/' + namelist[iair])
