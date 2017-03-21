import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import collections

from bokeh.plotting import figure, show, output_file,gridplot,vplot
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, Legend,CheckboxGroup
from bokeh.models import FixedTicker,PrintfTickFormatter

n = 100
color = plt.cm.jet(np.linspace(0.1, 0.99, n))  # This returns RGBA; convert:
hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
                   tuple(color[:, 0:-1]))
hexcolor[0] = '#000000'


TELESCOPES_COLOR = {'O' : '#190707',
	            'K' : '#61210B',
	            'D' : '#FE2EF7',
		    'E' : '#F781F3',
                    'F' : '#DF01D7',
		    'R' : '#F5A9F2',
		    'S' : '#FF8000', 		
		    'T' : '#DF3A01',	
	            'X' : '#4B088A',
		    'Y' : '#8000FF',
		    }  		 
 
TELESCOPES_SHAPE = {'O' : 'circle',
	            'K' : 'circle',
	            'D' : 'circle',
		    'E' : 'square',
                    'F' : 'triangle',
		    'R' : 'circle',
		    'S' : 'square', 		
		    'T' : 'triangle',	
	            'X' : 'circle',
		    'Y' : 'triangle',
		    } 

TELESCOPES_FILTER_COLOR = {'I' : 'circle',
	            'R' : 'circle',
	            'i' : 'circle',
		    'g' : 'diamond',
		    'r' : 'square',
		    } 
def plot_bokeh_segments_for_errorbars(figure, x_data, y_data, y_error, data_color, data_alpha = 0.3):

	figure.segment(x_data, y_data+np.abs(y_error),x_data, y_data-np.abs(y_error), color= data_color, line_alpha= data_alpha)

def plot_bokeh_points_for_errorbars(figure, x_data, y_data, data_color, data_shape, data_alpha):

	r0 = figure.scatter(x_data, y_data, color= data_color, marker = data_alpha, size = 10)
	return r0
def plot_bokeh_points_with_metadata(figure, the_meta_data, hover_name=None):

	metapoints = figure.scatter(x='time', y='mag', fill_color='color', line_color=None, size=4, source=the_meta_data, name=hover_name)
		
	return metapoints


class PlotTelescope():

	def __init__(self, name, lightcurve=[], lightcurve_dico={'time' : 0, 'mag' : 1, 'err_mag': 2 }, couleur='black'):

		self.name = name
		self.lightcurve = lightcurve
		self.lightcurve_dictionary = lightcurve_dico 
		self.couleur = couleur
		
		self.residuals = {}

		self.define_time_mag_err_mag()
		try :

			self.couleur = TELESCOPES_COLOR[name[0]]
		except :

			pass

		try:

			self.shape = TELESCOPES_SHAPE[name[0]]

		except :

			self.shape = 'circle'


		try:

			self.alpha = TELESCOPES_FILTER_COLOR[name[1]]

		except :

			self.alpha = 'black'


	def define_time_mag_err_mag(self):
		
		self.time = self.lightcurve[:, self.lightcurve_dictionary['time']]
		self.mag = self.lightcurve[:, self.lightcurve_dictionary['mag']]
		self.err_mag = self.lightcurve[:, self.lightcurve_dictionary['err_mag']]
			
	def define_residuals(self, model_name, residuals, residuals_dictionnary):

		keys = ['time','delta_mag','err_mag']
		self.residuals[model_name] = collections.namedtuple('RESIDUALS', keys)

		values = [residuals[:,residuals_dictionnary['time']],
			  residuals[:,residuals_dictionnary['delta_mag']],
			  residuals[:,residuals_dictionnary['err_mag']]
			 ]
		
		count = 0
    		for key in keys:
        		setattr(self.residuals[model_name], key, values[count])
			
        		count += 1

	def define_the_bokeh_metadata(self):

		
		
		observatory = [self.name]*len(self.time)
		color = [self.couleur]*len(self.time)
		### add whatever you want....	


		metadata_elements = [self.time, self.mag, self.err_mag, observatory,color]
		metadata_elements_names = ['time','mag','err_mag','observatory','color']		
		metadata_dictionnary = {}

		#count = 0
		#for key in metadata_elements_names:
			
			#metadata_dictionnary[key] = metadata_elements[count]
			
			#count += 1
		
		#self.metadata = ColumnDataSource(data=metadata_dictionnary)


class PlotModel():

	def __init__(self, name, lightcurve, lightcurve_dico={'time' : 0, 'mag' : 1}, couleur='red') :

		self.name = name
		self.lightcurve = lightcurve
		self.lightcurve_dictionary = lightcurve_dico 
		self.couleur = couleur

		self.define_time_mag()

	def define_time_mag(self):

		self.time = self.lightcurve[:, self.lightcurve_dictionary['time']]
		self.mag = self.lightcurve[:, self.lightcurve_dictionary['mag']]
		


class EventPlot():

	def __init__(self, event_name) :

		self.event_name = event_name

		self.telescopes = []

		self.models = []

		self.plot_limits = []

		self.plot_characteristic = {}
	

	def define_plot_windows(self, plot_width = 800, min_border_left = 50, lightcurve_height = 380, residuals_height = 120, residuals_border_top = 10, 
					   plot_title = None, plot_title_alignement = 'center', 
					   lightcurve_y_label = 'Mag', residuals_x_label = 'HJD-2450000', residuals_y_label = '',
					   x_range_limits = None, lightcurve_y_range_limits = None, residuals_y_range_limits = (0.2,-0.2)):

		hover = HoverTool(names=['points'])
		bokeh_plot_tools = ['box_zoom',hover,'reset','save']
		if not plot_title :

			plot_title = self.event_name

		if not  x_range_limits :

			x_range_limits = self.time_plot_limits	

		if not  lightcurve_y_range_limits :

			lightcurve_y_range_limits = self.magnitude_plot_limits			

		self.figure_lightcurve = figure(width=plot_width, plot_height=lightcurve_height, title=plot_title, title_text_align=plot_title_alignement, 
						y_axis_label=lightcurve_y_label,x_range= x_range_limits,y_range=lightcurve_y_range_limits,
						min_border_left=min_border_left, tools=bokeh_plot_tools)


		self.figure_lightcurve.xaxis.minor_tick_line_color=None
		self.figure_lightcurve.xaxis.major_tick_line_color=None
		self.figure_lightcurve.xaxis.major_label_text_font_size='0pt'
		self.figure_lightcurve.xaxis[0].ticker.desired_num_ticks = 3


		self.figure_residuals = figure(width=plot_width, plot_height=residuals_height, x_range=self.figure_lightcurve.x_range, y_range=residuals_y_range_limits, 
					       x_axis_label=residuals_x_label, y_axis_label=residuals_y_label, min_border_left=min_border_left+20, 
					       min_border_top=residuals_border_top)
	
		
			
		self.figure_residuals.xaxis[0].formatter = PrintfTickFormatter(format="%.2f")
		self.figure_residuals.xaxis[0].ticker.desired_num_ticks = 3
		self.figure_residuals.yaxis[0].ticker.desired_num_ticks = 3
		self.figure_residuals.xaxis.minor_tick_line_color = None
		self.figure_residuals.yaxis.minor_tick_line_color = None

        def define_plot_limits(self, time_limit = [-np.inf,np.inf], mag_limit = [-np.inf, np.inf], errmag_limit = [0, np.inf]):
		
		self.time_plot_limits = time_limit
		self.magnitude_plot_limits = mag_limit
		self.error_magnitude_plot_limits = errmag_limit			

	def add_telescope_to_plot(self, telescope_name, telescope_lightcurve=[], 
					telescope_lightcurve_dictionnary={'time' : 0, 'mag' : 1, 'err_mag': 2 }, telescope_color='black'):
		
		the_telescope = PlotTelescope(telescope_name, telescope_lightcurve, telescope_lightcurve_dictionnary, telescope_color)
		
		self.telescopes.append(the_telescope)

	def add_model_to_plot(self, model_name, model_lightcurve=[], 
				    model_lightcurve_dictionnary={'time' : 0, 'mag' : 1}, model_color='red'):
		
		the_model = PlotModel(model_name, model_lightcurve, model_lightcurve_dictionnary, model_color)
		
		self.models.append(the_model)
	
	
	
	def update_telescope_with_model_residuals(self, telescope_name, model_name, telescope_residuals, 
							telescope_residuals_dictionnary = {'time' : 0, 'delta_mag' : 1, 'err_mag': 2 }):

		good_telescope = [ telescope for telescope in self.telescopes if telescope.name == telescope_name][0]
	
		good_telescope.define_residuals(model_name, telescope_residuals, telescope_residuals_dictionnary)

	
	def define_telescopes_metadata(self):

		for telescope in self.telescopes:

			telescope.define_the_bokeh_metadata()



	def plot_the_lightcurve_data(self):
		
		legendary = []
		for telescope in self.telescopes:
			
			plot_bokeh_segments_for_errorbars(self.figure_lightcurve, telescope.time, telescope.mag, telescope.err_mag, telescope.couleur)
			r0 = plot_bokeh_points_for_errorbars(self.figure_lightcurve, telescope.time, telescope.mag,telescope.couleur,telescope.shape, telescope.alpha)
			#metapoints = plot_bokeh_points_with_metadata(self.figure_lightcurve, telescope.metadata, hover_name = 'points')
			#import pdb; pdb.set_trace()		
			legendary.append((telescope.name,[r0]))
	
		legend = Legend(legends=legendary, location=(0,-30))
		self.figure_lightcurve.add_layout(legend,'right')
		self.figure_lightcurve.select_one(HoverTool).tooltips = [
			('Observatoire','@observatory'),
			('Date','@time'),
			('Mag','@mag'),
			('Err_Mag','@err_mag'),
			('Color','@color')
			]

	def plot_the_residuals_data(self, model_name):
		
		legendary = []
		for telescope in self.telescopes:
			
			residuals = telescope.residuals[model_name]
					
			

			plot_bokeh_segments_for_errorbars(self.figure_residuals, residuals.time, residuals.delta_mag, residuals.err_mag, telescope.couleur)
			plot_bokeh_points_for_errorbars(self.figure_residuals, residuals.time, residuals.delta_mag, telescope.couleur, telescope.shape, telescope.alpha)

	def plot_the_model(self, model_name):
				
		good_model = [model for model in self.models if model.name == model_name][0]
		
		self.figure_lightcurve.line(good_model.time, good_model.mag, line_width=0.7, color='red')
		
		

		


	def assemble_lightcurve_and_residual_plots(self):

		self.finalplot = gridplot([[self.figure_lightcurve],[self.figure_residuals]],toolbar_location="right")
	
	def generate_the_plot(self, model_name):

		output_file(self.event_name+'_'+model_name+'.html',title=self.event_name)

		self.define_telescopes_metadata()
		self.plot_the_lightcurve_data()
		self.plot_the_residuals_data(model_name)
		self.plot_the_model(model_name)	

		
		
		self.assemble_lightcurve_and_residual_plots()
		show(self.finalplot)
		

class QualityControlPlot():


	def __init__(self, quantity_name, quantity_time, quantity_data, couleur='red') :

		self.name = quantity_name
		self.time = quantity_time
		self.data = quantity_data
		self.color = couleur

	def plot_histogram(self):

		figure_histogram = figure(title=self.name+" distribution")
		hist, edges = np.histogram(self.data, density=True, bins=50)

		figure_histogram.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],fill_color=self.color, line_color=self.color)
		figure_histogram.xaxis.axis_label = self.name		
		figure_histogram.yaxis.axis_label = 'Counts'
		return figure_histogram

	def plot_quantity_vs_time(self):

		figure_quantity_time = figure(title=self.name)
	

		figure_quantity_time.scatter(self.time, self.data, color= self.color)
		figure_quantity_time.xaxis.axis_label = 'Time'
		figure_quantity_time.yaxis.axis_label = self.name

		return figure_quantity_time

	def generate_the_plot(self):

		self.quantity_time_plot = self.plot_quantity_vs_time()
		self.quantity_histogram = self.plot_histogram()
		self.finalplot = gridplot(self.quantity_time_plot,self.quantity_histogram,ncols=2, toolbar_location="right")
		show(self.finalplot)
