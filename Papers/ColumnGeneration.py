####IMPORTAMOS MÓDULOS
from collections import defaultdict
from gurobipy import *
from values3 import *
import json


def resolver_problema(iteraciones_max,iteraciones_max2):
	
	################################
	PI_Ui1 = defaultdict(dict)
	PI_Uin = defaultdict(dict)
	PI_UiN = defaultdict(dict)
	PI_Vij1 = defaultdict(dict)
	PI_Vijl = defaultdict(dict)
	PI_VijL = defaultdict(dict)
	################################

	############################
	## MASTER PROBLEM PHASE 1 ##
	############################
	master = Model("Master problem")

	###Generamos las variables del problema
	sigma = master.addVar(vtype = GRB.CONTINUOUS, lb=0, name="sigma_beta")
	sigmaui1 = master.addVars(I, vtype = GRB.CONTINUOUS, lb=0, name="sigmaU_i1")
	sigmauin = master.addVars(I, N, vtype = GRB.CONTINUOUS, lb=0, name="sigmaU_in")
	sigmauiN = master.addVars(I, vtype = GRB.CONTINUOUS, lb=0, name="sigmaU_iN")
	sigmavij1 = master.addVars(I, J, vtype = GRB.CONTINUOUS, lb=0, name="sigmaV_ij1")
	sigmavijl = master.addVars(I, J, L, vtype = GRB.CONTINUOUS, lb=0, name="sigmaV_ijl")
	sigmavijL = master.addVars(I, J, vtype = GRB.CONTINUOUS, lb=0, name="sigmaV_ijL")
	master.update()

	#### RESTRICCIONES - REVISADO 
	igualdad = master.addConstr((sigma == 1), name="igualdad")
	ui1 = master.addConstrs((sigmaui1[i] >= E_u[i][N[0]] for i in I), name="EsperanzaU_i1")
	uin = master.addConstrs((sigmauin[i,n] >= E_u[i][n] for i in I for n in N[1:-1]), name="EsperanzaU_in")
	uiN = master.addConstrs((sigmauiN[i] >= E_u[i][N[-1]] for i in I), name="EsperanzaU_iN")
	vij1 = master.addConstrs((sigmavij1[i,j] >= E_v[i][j][L[0]] for i in I for j in J_i[i]), name="EsperanzaV_ij1")
	vijl = master.addConstrs((sigmavijl[i,j,l] >= E_v[i][j][l] for i in I for j in J_i[i] for l in L[1:-1]), name="EsperanzaV_ijl")
	vijL = master.addConstrs((sigmavijL[i,j] >= E_v[i][j][L[-1]] for i in I for j in J_i[i]), name="EsperanzaV_ijL")
	master.update()

	#### FUNCIÓN OBJETIVO
	PUi1 = quicksum(sigmaui1[i] for i in I)
	PUin = quicksum(sigmauin[i,n] for i in I for n in N[1:-1])
	PUiN = quicksum(sigmauiN[i] for i in I)
	PVij1 = quicksum(sigmavij1[i,j] for i in I for j in J_i[i])
	PVijl = quicksum(sigmavijl[i,j,l] for i in I for j in J_i[i] for l in L[1:-1])
	PVijL = quicksum(sigmavijL[i,j] for i in I for j in J_i[i])
	master.setObjective(sigma + PUi1+ PUin+ PUiN + PVij1+ PVijl+ PVijL, GRB.MINIMIZE)
	master.update()

	############################
	## MASTER PROBLEM PHASE 2 ##
	############################
	master2 = Model("Master problem F2")

	###Generamos las variables del problema
	sigma2 = master2.addVar(vtype = GRB.CONTINUOUS, lb=0, name="sigma_beta")
	sigmaui12 = master2.addVars(I, vtype = GRB.CONTINUOUS, lb=0, name="sigmaU_i1")
	sigmauin2 = master2.addVars(I, N, vtype = GRB.CONTINUOUS, lb=0, name="sigmaU_in")
	sigmauiN2 = master2.addVars(I, vtype = GRB.CONTINUOUS, lb=0, name="sigmaU_iN")
	sigmavij12 = master2.addVars(I, J, vtype = GRB.CONTINUOUS, lb=0, name="sigmaV_ij1")
	sigmavijl2 = master2.addVars(I, J, L, vtype = GRB.CONTINUOUS, lb=0, name="sigmaV_ijl")
	sigmavijL2 = master2.addVars(I, J, vtype = GRB.CONTINUOUS, lb=0, name="sigmaV_ijL")
	master2.update()

	#### RESTRICCIONES - REVISADO 
	igualdad2 = master2.addConstr((sigma2 == 1), name="igualdad")
	ui12 = master2.addConstrs((sigmaui12[i] >= E_u[i][N[0]] for i in I), name="EsperanzaU_i12")
	uin2 = master2.addConstrs((sigmauin2[i,n] >= E_u[i][n] for i in I for n in N[1:-1]), name="EsperanzaU_in2")
	uiN2 = master2.addConstrs((sigmauiN2[i] >= E_u[i][N[-1]] for i in I), name="EsperanzaU_iN2")
	vij12 = master2.addConstrs((sigmavij12[i,j] >= E_v[i][j][L[0]] for i in I for j in J_i[i]), name="EsperanzaV_ij12")
	vijl2 = master2.addConstrs((sigmavijl2[i,j,l] >= E_v[i][j][l] for i in I for j in J_i[i] for l in L[1:-1]), name="EsperanzaV_ijl2")
	vijL2 = master2.addConstrs((sigmavijL2[i,j] >= E_v[i][j][L[-1]] for i in I for j in J_i[i]), name="EsperanzaV_ijL2")
	master2.update()

	#### FUNCIÓN OBJETIVO
	master2.setObjective(0, GRB.MINIMIZE)

	############################
	######## SUB PROBLEM #######
	############################
	sat = Model("Pricing problem")

	###Generamos las variables del problema
	x = sat.addVars(I, N, J, vtype=GRB.INTEGER, lb=0, name="x") #x_inj - oka
	y = sat.addVars(I, J, L, J, vtype=GRB.INTEGER, lb=0, name="y") #y_ijlk - OJO: los j y k son solo en J(i)
	u = sat.addVars(I, N, vtype=GRB.INTEGER, lb=0, name="u") #u_in - oka
	v = sat.addVars(I, J, L, vtype=GRB.INTEGER, lb=0, name="v") #v_ijl - oka
	delta = sat.addVars(I, N, vtype=GRB.BINARY, lb=0, name="d1") #delta_in - oka
	delta2 = sat.addVars(I, J, L, vtype=GRB.BINARY, lb=0, name="d2") #delta_ijl - OJO - solo en J(i)
	z = sat.addVars(I, vtype=GRB.INTEGER, lb=0, name="z") #z_i - oka
	jf = sat.addVars(F, vtype=GRB.CONTINUOUS, lb=0, name="JF") #J_f - oka
	sat.update()

	###Generamos las restricciones del problema pricing - REVISADO 
	sat.addConstrs((quicksum(x[i,n,j] for i in I_j[j] for n in N) + quicksum(y[i,k,l,j] for i in I_j[j] for k in J_i[i] for l in L)-quicksum(y[i,j,l,k] for i in I_j[j] for k in J_i[i] for l in L)<=C_j[j]-quicksum(v[i,j,l] for i in I_j[j] for l in L) for j in J), name = "Capacidad")
	sat.addConstrs((x[i,n,j] == 0 for i in I for n in N for j in J if j not in J_i[i]), name = "preferencias")
	sat.addConstrs((y[i,j,l,k] == 0 for i in I for j in J for l in L for k in J if j not in J_i[i] or k not in J_i[i] or j == k), name = "preferencias2")
	sat.addConstrs((quicksum(x[i,n,j] for j in J_i[i]) <= u[i,n] for i in I for n in N), name = "waiting_d1")
	sat.addConstrs((quicksum(y[i,j,l,k] for k in J_i[i]) <= v[i,j,l] for i in I for j in J_i[i] for l in L), name = "waiting_d2")
	sat.addConstrs((quicksum(x[i,n,j] for j in J_i[i])<=M*delta[i,n] for i in I for n in N[:-1]), name="FIFO")
	sat.addConstrs((quicksum(u[i,n+1]-quicksum(x[i,n+1,j] for j in J_i[i]) for n in N[:-1])<=M*(1-delta[i,n]) for i in I for n in N[:-1]), name = "FIFO2")
	sat.addConstrs((quicksum(y[i,j,l,k] for k in J_i[i])<=M*delta2[i,j,l] for i in I for j in J_i[i] for l in L[:-1]), name="FIFO3")
	sat.addConstrs((quicksum(v[i,j,l+1]-quicksum(y[i,j,l+1,k] for k in J_i[i]) for l in L[:-1])<=M*(1-delta2[i,j,l]) for i in I for j in J_i[i] for l in L[:-1]), name = "FIFO4")
	sat.addConstrs((z[i]>=u[i,N[-1]]-quicksum(x[i,N[-1],j] for j in J_i[i])- W_i[i] for i in I), name="derivacion")
	sat.addConstrs((z[i]<=u[i,N[-1]]-quicksum(x[i,N[-1],j] for j in J_i[i]) for i in I), name="derivacion_2")
	sat.addConstrs((u[i,n] <= 3*cajon_superior(E_u[i][n]) for i in I for n in N), name = "cota_u")
	sat.addConstrs((v[i,j,l] <= 10*cajon_superior(E_v[i][j][l]) for i in I for j in J for l in L), name = "cota_v")
	sat.addConstrs((jf[f] >= quicksum(u[i,n] for i in I_f[f] for n in N)-T_f[f] for f in F), name = "apoyo_costo")
	sat.update()

	############################
	############ G.C ###########
	############################	
	iteracion = 0
	print("------------------------")
	print("START PHASE 1")
	print("------------------------")

	while True:
		master.Params.OutputFlag = 0
		master.optimize()
		print(f"Iteración: {iteracion}, Valor F.O master: {master.objVal}")
		print(f"Dual igualdad: {igualdad.Pi}")
		
		##Condición de termino
		if master.objVal == 0 or iteracion > iteraciones_max:
			print("------------------------")
			print("END PHASE 1")
			print("------------------------")
			break

		###funcion objetivo pricing y optimización del pricing - REVISADO
		P1 = quicksum((u[i,N[0]]-gamma*tasa[i])*ui1[i].Pi for i in I)
		P2 = quicksum((u[i,n] - gamma*(1-rho_wait[i])*(u[i,n-1]-quicksum(x[i,n-1,j] for j in J_i[i])))*uin[i,n].Pi for i in I for n in N[1:-1])
		P3 = quicksum((u[i,N[-1]] - quicksum(gamma*(1-rho_wait[i])*(u[i,n]-quicksum(x[i,n,j] for j in J_i[i])) for n in N[-2:]) + gamma*(1-rho_wait[i])*z[i])*uiN[i].Pi for i in I)
		P4 = quicksum((v[i,j,L[0]]-gamma*(1-rho_been[i])*quicksum(x[i,n,j] for n in N))*vij1[i,j].Pi for i in I for j in J_i[i])
		P5 = quicksum((v[i,j,l] - gamma*(1-rho_been[i])*(v[i,j,l-1]+quicksum(y[i,k,l-1,j] for k in J_i[i])-quicksum(y[i,j,l-1,k] for k in J_i[i])))*vijl[i,j,l].Pi for i in I for j in J_i[i] for l in L[1:-1])
		P6 = quicksum((v[i,j,L[-1]]-quicksum(gamma*(1-rho_been[i])*(v[i,j,l]+quicksum(y[i,k,l,j] for k in J_i[i])-quicksum(y[i,j,l,k] for k in J_i[i])) for l in L[-2:]))*vijL[i,j].Pi for i in I for j in J_i[i])
		sat.setObjective((1-(gamma))*igualdad.Pi + P1 + P2+ P3+ P4+ P5+ P6, GRB.MAXIMIZE)
		sat.update()
		sat.Params.OutputFlag = 0
		sat.Params.MIPGap = 0.02
		sat.optimize()
		print(f"Valor F.O satelite: {sat.objVal}")

		##########################################################
		#### Agregamos columnas nuevas a maestro fase 1 - REVISADO
		col = Column()
		col.addTerms((1-gamma), igualdad) # se agrega columna primera restriccion
		for i in I:
			sumax1 = quicksum(gamma*(1-rho_wait[i])*(u[i,n]-quicksum(x[i,n,j] for j in J_i[i])) for n in N[-2:])
			col.addTerms((u[i,N[0]].X - gamma*tasa[i]), ui1[i]) # columna a restricción 2
			col.addTerms((u[i,N[-1]].X-sumax1.getValue() + gamma*(1-rho_wait[i])*z[i].X), uiN[i]) # columna a restricción 4
		
		for i in I:
			for n in N[1:-1]:
				sumax2 = quicksum(x[i,n-1,j] for j in J_i[i])
				col.addTerms((u[i,n].X - gamma*(1-rho_wait[i])*(u[i,n-1].X-sumax2.getValue())), uin[i,n]) # columna a restricción 3

		for i in I:
			for j in J_i[i]:
				sumax3 = quicksum(x[i,n,j] for n in N)
				sumay1 = quicksum(gamma*(1-rho_been[i])*(v[i,j,l]+quicksum(y[i,k,l,j] for k in J_i[i])-quicksum(y[i,j,l,k] for k in J_i[i])) for l in L[-2:])
				col.addTerms((v[i,j,L[0]].X - gamma*(1-rho_been[i])*sumax3.getValue()), vij1[i,j]) # columna a restricción 5
				col.addTerms((v[i,j,L[-1]].X-sumay1.getValue()), vijL[i,j]) # columna a restricción 7

		for i in I:
			for j in J_i[i]:
				for l in L[1:-1]:
					sumay2 = quicksum(y[i,k,l-1,j] for k in J_i[i])
					sumay3 = quicksum(y[i,j,l-1,k] for k in J_i[i])
					col.addTerms((v[i,j,l].X - gamma*(1-rho_been[i])*(v[i,j,l-1].X + sumay2.getValue() - sumay3.getValue())), vijl[i,j,l]) # columna a restricción 6



		### Agregamos variables a maestro de fase 1
		master.addVar(vtype = GRB.CONTINUOUS, name= f"pi1[{iteracion}]", column = col)
		master.update()	

		#########################################################
		#### Agregamos columnas nuevas a maestro fase 2 - REVISADO
		col = Column()
		col.addTerms((1-gamma), igualdad2) # se agrega columna primera restriccion
		for i in I:
			sumax1 = quicksum(gamma*(1-rho_wait[i])*(u[i,n]-quicksum(x[i,n,j] for j in J_i[i])) for n in N[-2:])
			col.addTerms((u[i,N[0]].X - gamma*tasa[i]), ui12[i]) # columna a restricción 2
			col.addTerms((u[i,N[-1]].X-sumax1.getValue() + gamma*(1-rho_wait[i])*z[i].X), uiN2[i]) # columna a restricción 4
		
		for i in I:
			for n in N[1:-1]:
				sumax2 = quicksum(x[i,n-1,j] for j in J_i[i])
				col.addTerms((u[i,n].X - gamma*(1-rho_wait[i])*(u[i,n-1].X-sumax2.getValue())), uin2[i,n]) # columna a restricción 3

		for i in I:
			for j in J_i[i]:
				sumax3 = quicksum(x[i,n,j] for n in N)
				sumay1 = quicksum(gamma*(1-rho_been[i])*(v[i,j,l]+quicksum(y[i,k,l,j] for k in J_i[i])-quicksum(y[i,j,l,k] for k in J_i[i])) for l in L[-2:])
				col.addTerms((v[i,j,L[0]].X - gamma*(1-rho_been[i])*sumax3.getValue()), vij12[i,j]) # columna a restricción 5
				col.addTerms((v[i,j,L[-1]].X-sumay1.getValue()), vijL2[i,j]) # columna a restricción 7

		for i in I:
			for j in J_i[i]:
				for l in L[1:-1]:
					sumay2 = quicksum(y[i,k,l-1,j] for k in J_i[i])
					sumay3 = quicksum(y[i,j,l-1,k] for k in J_i[i])
					col.addTerms((v[i,j,l].X - gamma*(1-rho_been[i])*(v[i,j,l-1].X + sumay2.getValue() - sumay3.getValue())), vijl2[i,j,l]) # columna a restricción 6

		
		costo1 = quicksum(cw_in*u[i,n] for i in I for n in N) # costo CW_IN
		costo2 = quicksum(cp_ijl*v[i,j,l] for i in I for j in J_i[i] for l in L) #costo CP_IJL
		costo3 = quicksum(b_ijlk[i][l]*y[i,j,l,k] for i in I for j in J_i[i] for l in L for k in J_i[i]) #b_IJLK
		costo4 = quicksum(a_inj[i][n]*x[i,n,j] for i in I for n in N for j in J_i[i]) #costo a_INJ
		costo5 = quicksum(cd_i*z[i] for i in I) #costo Z_I
		costo6 = quicksum(cw_f*jf[f] for f in F)
		costo = costo1.getValue() + costo2.getValue()+ costo3.getValue()+ costo4.getValue()+ costo5.getValue()+ costo6.getValue()

		### Agregamos variables a maestro de fase 2
		master2.addVar(obj=costo, vtype = GRB.CONTINUOUS, name= f"pi1[{iteracion}]", column = col)
		master2.update()	


		iteracion +=1

	#### Imprimir modelos FASE 1 
	master.write("masterF1.lp")
	sat.write("satF1.lp")

	####master fase2
	master2.remove(sigma2)
	master2.remove(sigmaui12)
	master2.remove(sigmauin2)
	master2.remove(sigmauiN2)
	master2.remove(sigmavij12)
	master2.remove(sigmavijl2)
	master2.remove(sigmavijL2)
	master2.update()

	#### Imprimir modelos FASE 2 
	master2.write("masterF2.lp")

	############################
	######### G.C F2 ###########
	############################
	iteracion2 = 0
	print("""
------------------------
START PHASE 2
------------------------""")
	while True:
		print(f"""
------------------
Iteración {iteracion2}
------------------""")

		master2.Params.OutputFlag = 0
		master2.optimize()
		print(f"Valor F.O master: {master2.objVal}")
		print(f"Dual igualdad: {igualdad2.Pi}")

		###funcion objetivo pricing y optimización del pricing - REVISADO
		P1 = quicksum((u[i,N[0]]-gamma*tasa[i])*ui12[i].Pi for i in I)
		P2 = quicksum((u[i,n] - gamma*(1-rho_wait[i])*(u[i,n-1]-quicksum(x[i,n-1,j] for j in J_i[i])))*uin2[i,n].Pi for i in I for n in N[1:-1])
		P3 = quicksum((u[i,N[-1]] - quicksum(gamma*(1-rho_wait[i])*(u[i,n]-quicksum(x[i,n,j] for j in J_i[i])) for n in N[-2:]) + gamma*(1-rho_wait[i])*z[i])*uiN2[i].Pi for i in I)
		P4 = quicksum((v[i,j,L[0]]-gamma*(1-rho_been[i])*quicksum(x[i,n,j] for n in N))*vij12[i,j].Pi for i in I for j in J_i[i])
		P5 = quicksum((v[i,j,l] - gamma*(1-rho_been[i])*(v[i,j,l-1]+quicksum(y[i,k,l-1,j] for k in J_i[i])-quicksum(y[i,j,l-1,k] for k in J_i[i])))*vijl2[i,j,l].Pi for i in I for j in J_i[i] for l in L[1:-1])
		P6 = quicksum((v[i,j,L[-1]]-quicksum(gamma*(1-rho_been[i])*(v[i,j,l]+quicksum(y[i,k,l,j] for k in J_i[i])-quicksum(y[i,j,l,k] for k in J_i[i])) for l in L[-2:]))*vijL2[i,j].Pi for i in I for j in J_i[i])
		costo1 = quicksum(cw_in*u[i,n] for i in I for n in N) # costo CW_IN
		costo2 = quicksum(cp_ijl*v[i,j,l] for i in I for j in J_i[i] for l in L) #costo CP_IJL
		costo3 = quicksum(b_ijlk[i][l]*y[i,j,l,k] for i in I for j in J_i[i] for l in L for k in J_i[i]) #b_IJLK
		costo4 = quicksum(a_inj[i][n]*x[i,n,j] for i in I for n in N for j in J_i[i]) #costo a_INJ
		costo5 = quicksum(cd_i*z[i] for i in I) #costo Z_I
		costo6 = quicksum(cw_f*jf[f] for f in F)
		sat.setObjective((1-(gamma))*igualdad2.Pi + P1 + P2 + P3 + P4 + P5 + P6 + (-1*costo1) + (-1*costo2) + (-1*costo3) + (-1*costo4) + (-1*costo5) + (-1*costo6), GRB.MAXIMIZE)
		sat.update()
		sat.Params.OutputFlag = 0
		sat.Params.MIPGap = 0.02
		sat.optimize()
		print(f"Valor F.O satelite: {sat.objVal}")

		### Condicón de termino para fase 2
		if iteracion2 >= iteraciones_max2 or sat.objVal <= 0.001:
			print("Fin Fase II")
			#Imprimir duales
			print(f"Dual igualdad: {igualdad2.Pi}")
			for i in I:
				PI_Ui1[i] = ui12[i].Pi
				print(f"Dual ui1 grupo {i}: {ui12[i].Pi}")
			
			for i in I:
				for n in N[1:-1]:
					PI_Uin[i][n] = uin2[i,n].Pi
					print(f"Dual uin grupo {i} espera {n}: {uin2[i,n].Pi}")	

			for i in I:
				PI_UiN[i] = uiN2[i].Pi
				print(f"Dual uiN grupo {i}: {uiN2[i].Pi}")
			

			for i in I:
				for j in J_i[i]:
					PI_Vij1[i][j] = vij12[i,j].Pi
					print(f"Dual vij1 grupo {i} destino {j}: {vij12[i,j].Pi}")

			for i in I:
				for j in J_i[i]:
					PI_Vijl[i][j] = defaultdict(dict)
					for l in L[1:-1]:
						PI_Vijl[i][j][l] = vijl2[i,j,l].Pi
						print(f"Dual vijl grupo {i} destino {j} espera {l}: {vijl2[i,j,l].Pi}")
					
			for i in I:
				for j in J_i[i]:
					PI_VijL[i][j] = vijL2[i,j].Pi
					print(f"Dual vijN grupo {i} destino {j}: {vijL2[i,j].Pi}")

			break

		#### Agregamos columnas nuevas a maestro fase 2 - REVISADO
		col = Column()
		col.addTerms((1-gamma), igualdad2) # se agrega columna primera restriccion
		for i in I:
			sumax1 = quicksum(gamma*(1-rho_wait[i])*(u[i,n]-quicksum(x[i,n,j] for j in J_i[i])) for n in N[-2:])
			col.addTerms((u[i,N[0]].X - gamma*tasa[i]), ui12[i]) # columna a restricción 2
			col.addTerms((u[i,N[-1]].X-sumax1.getValue() + gamma*(1-rho_wait[i])*z[i].X), uiN2[i]) # columna a restricción 4
		
		for i in I:
			for n in N[1:-1]:
				sumax2 = quicksum(x[i,n-1,j] for j in J_i[i])
				col.addTerms((u[i,n].X - gamma*(1-rho_wait[i])*(u[i,n-1].X-sumax2.getValue())), uin2[i,n]) # columna a restricción 3

		for i in I:
			for j in J_i[i]:
				sumax3 = quicksum(x[i,n,j] for n in N)
				sumay1 = quicksum(gamma*(1-rho_been[i])*(v[i,j,l]+quicksum(y[i,k,l,j] for k in J_i[i])-quicksum(y[i,j,l,k] for k in J_i[i])) for l in L[-2:])
				col.addTerms((v[i,j,L[0]].X - gamma*(1-rho_been[i])*sumax3.getValue()), vij12[i,j]) # columna a restricción 5
				col.addTerms((v[i,j,L[-1]].X-sumay1.getValue()), vijL2[i,j]) # columna a restricción 7

		for i in I:
			for j in J_i[i]:
				for l in L[1:-1]:
					sumay2 = quicksum(y[i,k,l-1,j] for k in J_i[i])
					sumay3 = quicksum(y[i,j,l-1,k] for k in J_i[i])
					col.addTerms((v[i,j,l].X - gamma*(1-rho_been[i])*(v[i,j,l-1].X + sumay2.getValue() - sumay3.getValue())), vijl2[i,j,l]) # columna a restricción 6

		costo = costo1.getValue() + costo2.getValue() + costo3.getValue() + costo4.getValue() + costo5.getValue() + costo6.getValue()

		### Agregamos variables a maestro de fase 2
		master2.addVar(obj=costo, vtype = GRB.CONTINUOUS, name= f"pi2[{iteracion2}]", column = col)
		master2.update()	


		### Avanzamos a otra iteración
		iteracion2 +=1
	

	master2.write("masterF2-2.lp")
	sat.write("satF2-2.lp")

	salida = []
	salida.append(PI_Ui1)
	salida.append(PI_Uin)
	salida.append(PI_UiN)
	salida.append(PI_Vij1)
	salida.append(PI_Vijl)
	salida.append(PI_VijL)
	return salida
	#return entrada_simu

output = resolver_problema(5000,100000)

#######################################
###SE GENERA JSON CON DUALES OPTIMAS
with open('D-OPTIMAS.json', 'w') as fp:
    json.dump(output, fp)