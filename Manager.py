import MLPlots


def main():
	
	Plot=MLPlots.MLplots('OB151737','BINARY')

 	Plot.path_lightcurves('/home/bachelet/Robonet/Database/Plotter/Event_test/')
	Plot.path_models('/home/bachelet/Robonet/Database/Plotter/Event_test/')
	Plot.load_models()
	Plot.set_data_limits()
	Plot.load_data()
	Plot.find_survey()
	Plot.set_plot_limits()
	
	Plot.align_data()
	Plot.get_colors()	
	Plot.plot_data()
	#

if __name__ == "__main__":
    main()
