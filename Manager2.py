import RoboPLOT
import numpy as np

def add_telescopes(Plot, all_lightcurves):

	individuals = np.unique(all_lightcurves[:,-1])

	for individu in individuals :

		good = np.where(all_lightcurves[:,-1] == individu)[0]
		individu = 'Xg'
		Plot.add_telescope_to_plot(individu, telescope_lightcurve=all_lightcurves[good,:3].astype(float), 
					telescope_lightcurve_dictionnary={'time' : 0, 'mag' : 1, 'err_mag': 2 }, telescope_color='black')

		
		Plot.update_telescope_with_model_residuals(individu, 'PSPL', all_lightcurves[:,[0,3,2]][good].astype(float), 
							telescope_residuals_dictionnary = {'time' : 0, 'delta_mag' : 1, 'err_mag': 2 })
	for individu in individuals :

		good = np.where(all_lightcurves[:,-1] == individu)[0]
		individu = 'Xr'
		lala = all_lightcurves[good,:3].astype(float)
		lala[:,0] += 0.5		
		Plot.add_telescope_to_plot(individu, telescope_lightcurve=lala, 
					telescope_lightcurve_dictionnary={'time' : 0, 'mag' : 1, 'err_mag': 2 }, telescope_color='black')

		
		Plot.update_telescope_with_model_residuals(individu, 'PSPL', all_lightcurves[:,[0,3,2]][good].astype(float), 
							telescope_residuals_dictionnary = {'time' : 0, 'delta_mag' : 1, 'err_mag': 2 })
		

	for individu in individuals :

		good = np.where(all_lightcurves[:,-1] == individu)[0]
		individu = 'Yi'
		lala = all_lightcurves[good,:3].astype(float)
		lala[:,0] += 1.0		
		Plot.add_telescope_to_plot(individu, telescope_lightcurve=lala, 
					telescope_lightcurve_dictionnary={'time' : 0, 'mag' : 1, 'err_mag': 2 }, telescope_color='black')

		
		Plot.update_telescope_with_model_residuals(individu, 'PSPL', all_lightcurves[:,[0,3,2]][good].astype(float), 
							telescope_residuals_dictionnary = {'time' : 0, 'delta_mag' : 1, 'err_mag': 2 })

	
	for individu in individuals :

		good = np.where(all_lightcurves[:,-1] == individu)[0]
		individu = 'Ei'
		lala = all_lightcurves[good,:3].astype(float)
		lala[:,0] += 1.5		
		Plot.add_telescope_to_plot(individu, telescope_lightcurve=lala, 
					telescope_lightcurve_dictionnary={'time' : 0, 'mag' : 1, 'err_mag': 2 }, telescope_color='black')

		
		Plot.update_telescope_with_model_residuals(individu, 'PSPL', all_lightcurves[:,[0,3,2]][good].astype(float), 
							telescope_residuals_dictionnary = {'time' : 0, 'delta_mag' : 1, 'err_mag': 2 })
	for individu in individuals :

		good = np.where(all_lightcurves[:,-1] == individu)[0]
		individu = 'Ri'
		lala = all_lightcurves[good,:3].astype(float)
		lala[:,0] += 2.5		
		Plot.add_telescope_to_plot(individu, telescope_lightcurve=lala, 
					telescope_lightcurve_dictionnary={'time' : 0, 'mag' : 1, 'err_mag': 2 }, telescope_color='black')

		
		Plot.update_telescope_with_model_residuals(individu, 'PSPL', all_lightcurves[:,[0,3,2]][good].astype(float), 
							telescope_residuals_dictionnary = {'time' : 0, 'delta_mag' : 1, 'err_mag': 2 })

	for individu in individuals :

		good = np.where(all_lightcurves[:,-1] == individu)[0]
		individu = 'Fi'
		lala = all_lightcurves[good,:3].astype(float)
		lala[:,0] += 3.5		
		Plot.add_telescope_to_plot(individu, telescope_lightcurve=lala, 
					telescope_lightcurve_dictionnary={'time' : 0, 'mag' : 1, 'err_mag': 2 }, telescope_color='black')

		
		Plot.update_telescope_with_model_residuals(individu, 'PSPL', all_lightcurves[:,[0,3,2]][good].astype(float), 
							telescope_residuals_dictionnary = {'time' : 0, 'delta_mag' : 1, 'err_mag': 2 })
def main():
	
	Plot=RoboPLOT.EventPlot('OB170001')
	the_model = np.loadtxt('./plot_lightcurves/OB170001.pyLIMA_plot_model',dtype=str).astype(float)
	Plot.define_plot_limits(time_limit = [min(the_model[:,0]),max(the_model[:,0])], 
				mag_limit = [max(the_model[:,1])+0.1,min(the_model[:,1])-0.1], errmag_limit = [0, np.inf])
	Plot.define_plot_windows()		
	
	all_lightcurves = np.loadtxt('./plot_lightcurves/OB170001.pyLIMA_plot_lightcurves',dtype=str)

	add_telescopes(Plot, all_lightcurves)
	
	
	Plot.add_model_to_plot( 'PSPL', model_lightcurve=the_model, 
				    model_lightcurve_dictionnary={'time' : 0, 'mag' : 1}, model_color='red')

	Plot.generate_the_plot('PSPL')



 

if __name__ == "__main__":
    main()
