import RoboPLOT
import numpy as np

def add_telescopes(Plot, all_lightcurves):

	individuals = np.unique(all_lightcurves[:,-1])

	for individu in individuals :

		good = np.where(all_lightcurves[:,-1] == individu)[0]
		light = all_lightcurves[good,:3].astype(float)
		light[:,0] += 2450000
		Plot.add_telescope_to_plot(individu, telescope_lightcurve=light, 
					telescope_lightcurve_dictionnary={'time' : 0, 'mag' : 1, 'err_mag': 2 }, telescope_color='black')

		light = all_lightcurves[:,[0,3,2]][good].astype(float)
		light[:,0] += 2450000
		Plot.update_telescope_with_model_residuals(individu, 'PSPL', light, 
							telescope_residuals_dictionnary = {'time' : 0, 'delta_mag' : 1, 'err_mag': 2 })

def main():
	event_name = 'OB170001'
	Plot=RoboPLOT.EventPlot(event_name)
	the_model = np.loadtxt('/data/romerea/data/plotfiles/'+event_name+'.pyLIMA_PSPL_plot_model',dtype=str).astype(float)
	Plot.define_plot_limits(time_limit = [min(the_model[:,0]),max(the_model[:,0])], 
				mag_limit = [max(the_model[:,1])+0.1,min(the_model[:,1])-0.1], errmag_limit = [0, np.inf])
	Plot.define_plot_windows()		
	
	all_lightcurves = np.loadtxt('/data/romerea/data/plotfiles/'+event_name+'.pyLIMA_PSPL_plot_lightcurves',dtype=str)

	add_telescopes(Plot, all_lightcurves)
	
	
	Plot.add_model_to_plot( 'PSPL', model_lightcurve=the_model, 
				    model_lightcurve_dictionnary={'time' : 0, 'mag' : 1}, model_color='red')

	script, div = Plot.generate_the_plot('PSPL')
	import pdb; pdb.set_trace()


 

if __name__ == "__main__":
    main()
