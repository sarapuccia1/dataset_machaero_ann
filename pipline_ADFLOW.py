import numpy as np
import argparse
import os
import time

from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI


# read airfoils names
namelist = []
f = open('test_name.txt', 'r')
while True:
	filename = f.readline()
	if filename:
		namelist.append(filename.rstrip())
	else:
		break
f.close()

discarded = 0

for iair in range(len(namelist)):

	namelist[iair] = namelist[iair].replace('.dat', '.npy')
	samples = np.load('/home/mdolabuser/mount/test_dataset/sampling/uiuc_sampled/' + namelist[iair])
	y_coordinates = samples[1:21, 1].copy()
	y_coordinates = y_coordinates.T

	namelist[iair] = namelist[iair].replace('.npy', '.cgns')
	parser = argparse.ArgumentParser()
	
	output_directory = 'output/' + namelist[iair].replace('.cgns', '')
	parser.add_argument("--output", type=str, default=output_directory)
	parser.add_argument("--gridFile", type=str, default=("/home/mdolabuser/mount/test_dataset/mesh/test_cgns/" + namelist[iair]))
	parser.add_argument("--task", type=str, default="analysis")
	args = parser.parse_args()

	comm = MPI.COMM_WORLD
	if not os.path.exists(args.output):
		if comm.rank == 0:
			os.mkdir(args.output)
	
	T_array = np.array([288.15, 255.65, 223.15])
	Re_array = np.array([1.4e7, 8.7e6, 5.1e6])
	
	#ADFLOW ANALYSIS
	for T, Re in zip(T_array, Re_array):
		for mach in np.array([0.35, 0.45, 0.55, 0.65, 0.75, 0.85]):
		
			cl = np.empty((0,))
			cd = np.empty((0,))
			cm = np.empty((0,))
			AOA = np.empty((0,))

			for alpha in np.array([0, 2, 4, 6]):

				start_time = time.time()
				aeroOptions = {
				    # Common Parameters
				    "gridFile": args.gridFile,
				    "outputDirectory": args.output,
				    "writeTecplotSurfaceSolution": True,
				    # Physics Parameters
				    "equationType": "RANS",
				    "smoother": "DADI",
				    "MGCycle": "sg",
				    "nCycles": 20000,
				    "monitorvariables": ["resrho","resrhoe", "resturb", "cl", "cd"],
				    "useNKSolver": True,
				    "useanksolver": True,
				    "nsubiterturb": 10,
				    "liftIndex": 2,
				    "infchangecorrection": True,
				    # Convergence Parameters
				    "L2Convergence": 1e-10,
				    "L2ConvergenceCoarse": 1e-4,
				}

				# Create solver
				CFDSolver = ADFLOW(options=aeroOptions)
				namelist[iair]= namelist[iair].replace('.cgns','')
				name_cmp = namelist[iair] + '_' + str(Re) + '_' + str(mach) + '_' + str(alpha)

				ap = AeroProblem(name= name_cmp, alpha=alpha, mach=mach, T=T, reynolds=Re, reynoldsLength=1, areaRef=1.0, chordRef=1.0, evalFuncs=["cl", "cd", "cmy"])

				funcs = {}
				CFDSolver(ap)
				
				if args.task == "analysis":
					# Solve
					CFDSolver(ap)
					# rst Evaluate and print
					funcs = {}
					CFDSolver.evalFunctions(ap, funcs)
					hist = CFDSolver.getConvergenceHistory()

					# Print the evaluated functions
					if comm.rank == 0:

						#print (hist)
						selected_keys = ['RSDEnergyStagnationDensityRMS','RSDMassRMS','RSDTurbulentSANuTildeRMS']
						selected_items=[]
						for key in selected_keys:
							if key in hist:
								value = hist[key]
								if len(value) >=2:
									selected_items.append(value[0])


						all_lower = True
						end_time = time.time()
						elapsed_time = end_time - start_time
						total_time = total_time + elapsed_time

						for param in selected_items:
							if param >= 1e-04:
								all_lower = False
								break

						if all_lower and comm.rank == 0:
							converged += 1
							cd = np.append(cd, list(funcs.values())[0])
							cl = np.append(cl, list(funcs.values())[1])
							cm = np.append(cm, list(funcs.values())[2])
							AOA = np.append(AOA, alpha)
							with open('test_high_fidelity.csv', 'a') as csvfile:
									csvfile.write(','.join([str(i) for i in y_coordinates]) + ',' + str(Re) + ',' + str(mach) + ',' + str(alpha) + ',' + ','.join([str(i) for i in funcs.values()]) + ',' + str(elapsed_time)+ '\n')
						else:
							discarded += 1
							with open('discarded.csv', 'a') as csvfile:
									csvfile.write(','.join([str(i) for i in y_coordinates]) + ',' + str(Re) + ',' + str(mach) + ',' + str(alpha) + ',' + ','.join([str(i) for i in funcs.values()]) + ',' + str(elapsed_time)+ '\n')

		#inserire qui la regressione lineare
		#per inserire la regressione lineare è necessario salvare i coefficienti in un numpay array.

			if comm.rank == 0 and len(AOA) >= 2:
				AOA_predict = np.array([1,3,5])

				coeff_cd = np.polyfit(AOA, cd, 2)
				poly_drag = np.poly1d(coeff_cd)
				cd_pred = poly_drag(AOA_predict)

				coeff_cm = np.polyfit(AOA, cm, 2)
				poly_momentum = np.poly1d(coeff_cm)
				cm_pred = poly_momentum(AOA_predict)

				coeff_cl = np.polyfit(AOA, cl, 1)
				poly_momentum = np.poly1d(coeff_cl)
				cl_pred = poly_momentum(AOA_predict)


				for i in range(3):
					with open('test_infilling.csv', 'a') as csvfile:
						csvfile.write(','.join([str(i) for i in y_coordinates]) + ',' + str(Re) + ',' + str(mach) + ',' + str(
							AOA_predict[i]) + ',' + str(cd_pred[i]) + ',' + str(cl_pred[i]) + ',' + str(cm_pred[i]) + '\n')




