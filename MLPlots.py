import numpy as np
import matplotlib.pyplot as plt
import glob
from bokeh.plotting import figure, show, output_file,gridplot,vplot


class MLplots():

	def __init__(self,name,choice) :

		
		self.lightcurves_path=''
		self.models_path=''
		self.telescopes=[]
		self.lightcurves=[]
		self.aligned_lightcurves=[]
		self.models=[]
		self.plot_limits=[]
		self.survey=[]
		self.colors=[]
		self.name=name
		self.choice=choice
		self.data_limits=[]

	def path_lightcurves(self,directory):

		self.lightcurves_path=directory

	def path_models(self,directory):

		self.models_path=directory

	def load_data(self):

			
		self.telescopes_names=glob.glob(self.lightcurves_path+'/*.dat')
		for i in self.telescopes_names :
			
			data=np.loadtxt(i)
			index=np.where(data[:,2]>self.data_limits)[0]
			self.lightcurves.append(data[index])
			self.telescopes.append([i.replace(self.lightcurves_path,'')[0],i.replace(self.lightcurves_path,'')])

	def load_models(self):
		
		
		model=glob.glob(self.models_path+self.name+'*'+'.model')
		models=[]
		for i in model :

			models.append(i.replace(self.models_path,''))
		
		
		for i in models :
			data=np.loadtxt(self.models_path+i,dtype='string')
			if '_' in i :
				data=data[[4,6,8,10,12,14,16]].astype(float)
				self.models.append(['BINARY',data.tolist()])
				
			else :

				
				data=data[[3,5,6]].astype(float)
				self.models.append(['PSPL',data.tolist()])
		
	def get_colors(self) :
		def rd_tables(file_in):
    			out_table = [] # set up blank array to hold the elements
    			elements = 0
    			infile = open(file_in,'r')   # open the input file for reading
    			rows = infile.readlines()    # read the whole thing into a list

			for this_element in rows:
				
        			element_values = this_element.split()
        			out_table.append(element_values[:3])
    			elements = len(out_table)
    			return out_table, elements

		# Read in the colour definitions
		(col_codes, nr_col) = rd_tables('colourdef.cfg')
		translation=dict(col_codes)
		#import pdb; pdb.set_trace()    			
		# Convert to dictionary
		#col_cod = dict(col_codes)

		# Read in the observatory colour tags
		(obs_col, nr_obs) = rd_tables('colours_sig.cfg')
		couleurs=np.array(obs_col)
		couleurs=couleurs[:,[0,2]]
		couleurs=couleurs.tolist()

		#couleurs=dict(couleurs[::1])
		cooleurs=[[i[0],translation[i[1]]] for i in couleurs]
				

		#import pdb; pdb.set_trace()
		self.colors=dict(cooleurs)
		
	def find_survey(self) :

		letters=np.array([i[0] for i in self.telescopes])
		
		if 'O' in letters :

			index=np.where(letters=='O')[0][0]
			self.survey.append(['O',index])
		else :
			index=np.where(letters=='K')[0][0]
			self.survey.append(['K',index])	
	
	def set_data_limits(self):

		number_of_days=300
		offset_magnitude=0.2
		choices=np.array([i[0] for i in self.models])
		index=np.where(choices==self.choice)[0]
		
		parameters=self.models[index][1]
		to=parameters[0]
		
		self.data_limits=to-number_of_days

	def set_plot_limits(self):
		
		factor_time=3
		offset_magnitude=0.2
		choices=np.array([i[0] for i in self.models])
		index=np.where(choices==self.choice)[0]
		
		parameters=self.models[index][1]
		to=parameters[0]
		tE=parameters[1]
		self.plot_limits=[[to-factor_time*tE,to+factor_time*tE],[np.max(self.lightcurves[self.survey[0][1]][:,0])+offset_magnitude,np.min(self.lightcurves[self.survey[0][1]][:,0])-offset_magnitude]]
	
	def magnification(self,t):

		choices=np.array([i[0] for i in self.models])
		index=np.where(choices==self.choice)[0]
		
		parameters=self.models[index][1]
		if self.choice=='PSPL' :

			to=parameters[0]
			tE=parameters[1]
			uo=parameters[2]
			
			u=np.sqrt(uo**2+(t-to)**2/tE**2)
			ampli=(u**2+2)/(u*np.sqrt(u**2+4))

		if self.choice=='BINARY' :
			
			ampli=self.VB_binary(t)
		return ampli


		print blabla
	def VB_binary(self,t):
		import VBBinaryLensing
		choices=np.array([i[0] for i in self.models])
		index=np.where(choices==self.choice)[0]
		
		parameters=self.models[index][1]
		to=parameters[0]
		tE=parameters[1]
		uo=parameters[2]	
		rho=parameters[3]
		s=parameters[4]
		q=parameters[5]		
		alpha=parameters[6]
			
		
		amp=[]  
		salpha=np.sin(alpha)
    		calpha=np.cos(alpha)  			
		for i in t :

			
    			
    			y1=-uo*salpha+(i-to)/tE*calpha
    			y2=uo*calpha+(i-to)/tE*salpha    
			amp.append(VBBinaryLensing.BinaryLightCurve(s,q,y2,0.0,y1,rho,10**-3))
    		return np.array(amp)

	def align_data(self):
		
		parameters=self.models[0][1]
		to=parameters[0]
		tE=parameters[1]
		uo=parameters[2]
		
		t=self.lightcurves[self.survey[0][1]][:,2]
		flux=10**((18-self.lightcurves[self.survey[0][1]][:,0])/2.5)
		err_flux=flux*self.lightcurves[self.survey[0][1]][:,1]*np.log(10)/-2.5
		
		ampli=self.magnification(t)
		FS,FB=np.polyfit(ampli,flux,1,w=1/err_flux)
		
		for i in xrange(len(self.telescopes)):
			if i!=self.survey[0][1] :
				
				t=self.lightcurves[i][:,2]
				flux=10**((18-self.lightcurves[i][:,0])/2.5)
				err_flux=flux*self.lightcurves[i][:,1]*np.log(10)/-2.5
				u=np.sqrt(uo**2+(t-to)**2/tE**2)
				ampli=self.magnification(t)
				fs,fb=np.polyfit(ampli,flux,1,w=1/err_flux)
				
				if fs<0 :
					
					delta_flux=flux-FS*ampli+FB
					
					normalized_flux=flux-np.median(delta_flux)

					normalized_errflux=err_flux/flux*normalized_flux
					
				else :
	
					normalized_flux=(flux-fb)/fs*FS+FB
					normalized_errflux=err_flux/flux*normalized_flux
				
				#normalized_flux=(flux-fb)/fs*FS+FB
				#normalized_errflux=err_flux/flux*normalized_flux
				normalized_mag=18-2.5*np.log10(normalized_flux)
				normalized_errmag=-2.5*normalized_errflux/(normalized_flux*np.log(10))
				
				
				normalisation=np.array([normalized_mag,normalized_errmag,t]).T
			else :
				normalisation=self.lightcurves[i]
			
			self.aligned_lightcurves.append(normalisation)
	
	
		
	
	def plot_data(self):
		
		
		
		output_file(self.name+'_'+self.choice+'.html',title=self.name)
		tools = "pan,wheel_zoom,box_zoom,reset,save"
		
		fig1=figure(title=self.name+'_'+self.choice,title_text_align='center',y_axis_label='I',x_range=self.plot_limits[0],y_range=self.plot_limits[1],min_border_left=50,min_border_bottom=10,tools=tools,width=800,plot_height=380)

		T=np.arange(self.models[0][1][0]-3*self.models[0][1][1],self.models[0][1][0]+3*self.models[0][1][1],0.01)
		
		fig1.line(T,18-2.5*np.log10(self.lightcurve_model(T)),line_width=0.7,color='red')
		
		for i in xrange(len(self.telescopes)):

			
			
			#fig1.scatter(self.aligned_lightcurves[i][:,2],self.aligned_lightcurves[i][:,0],fill_color='#'+self.colors[self.telescopes[i][0]],line_color=None,legend=self.telescopes[i][1],size=5/10**np.abs(self.aligned_lightcurves[i][:,1]))
			fig1.segment(self.aligned_lightcurves[i][:,2],self.aligned_lightcurves[i][:,0]+np.abs(self.aligned_lightcurves[i][:,1]),self.aligned_lightcurves[i][:,2],self.aligned_lightcurves[i][:,0]-np.abs(self.aligned_lightcurves[i][:,1]),color='#'+self.colors[self.telescopes[i][0]],line_alpha=0.3)
			
			fig1.scatter(self.aligned_lightcurves[i][:,2],self.aligned_lightcurves[i][:,0],fill_color='#'+self.colors[self.telescopes[i][0]],line_color=None,legend=self.telescopes[i][1],size=4)
		
		
		fig1.xaxis.minor_tick_line_color=None
		fig1.xaxis.major_tick_line_color=None
		fig1.xaxis.major_label_text_font_size='0pt'
		fig1.legend.label_width=190
		fig1.legend.label_width=50
		fig1.legend.background_fill_alpha=0.5
		fig1.legend.label_text_alpha=0.5
		
		fig2=figure(width=800,plot_height=120,x_range=fig1.x_range,y_range=(-0.5,0.5),x_axis_label='HJD-2450000',y_axis_label='Delta_I',min_border_left=70,min_border_top=10)
		for i in xrange(len(self.telescopes)):
			
			residuals=self.residuals(i)
			
			fig2.segment(self.lightcurves[i][:,2],residuals+np.abs(self.lightcurves[i][:,1]),self.lightcurves[i][:,2],residuals-np.abs(self.lightcurves[i][:,1]),color='#'+self.colors[self.telescopes[i][0]],line_alpha=0.3)
			fig2.scatter(self.lightcurves[i][:,2],residuals,fill_color='#'+self.colors[self.telescopes[i][0]],line_color=None,size=4)
		fig2.xaxis.minor_tick_line_color=None
		p=gridplot([[fig1],[fig2]],toolbar_location="right")	
		
		show(p)

	def Flux_model(self,choice):

		number=int(choice)
		
		t=self.lightcurves[choice][:,2]
		flux=10**((18-self.lightcurves[choice][:,0])/2.5)
		err_flux=flux*self.lightcurves[choice][:,1]*np.log(10)/-2.5
		if self.choice=='PSPL' :
	
			
			
			ampli=self.magnification(t)
			fs,fb=np.polyfit(ampli,flux,1,w=1/err_flux)
			
			

		if self.choice=='BINARY' :
			ampli=self.VB_binary(t)
			fs,fb=np.polyfit(ampli,flux,1,w=1/err_flux)
			
			
			
		return fs*ampli+fb

					
			
			
		

	def lightcurve_model(self,T):

		choices=np.array([i[0] for i in self.models])
		index=np.where(choices==self.choice)[0]
		
		parameters=self.models[index][1]
		t=self.lightcurves[self.survey[0][1]][:,2]
		flux=10**((18-self.lightcurves[self.survey[0][1]][:,0])/2.5)
		err_flux=flux*self.lightcurves[self.survey[0][1]][:,1]*np.log(10)/-2.5
		
		if self.choice=='PSPL' :
	
			
			
			ampli=self.magnification(t)
			fs,fb=np.polyfit(ampli,flux,1,w=1/err_flux)
			
			
			ampli=self.magnification(T)

		if self.choice=='BINARY' :

			ampli=self.VB_binary(t)
			
			fs,fb=np.polyfit(ampli,flux,1,w=1/err_flux)
			
			
			ampli=self.magnification(T)
		return fs*ampli+fb

	def residuals(self,number) :

		flux_data=10**((18-self.lightcurves[number][:,0])/2.5)
		flux_model=self.Flux_model(number)

		if flux_model=='Bad fit' :
			print flux_model
			delta_flux=flux_data-10**((18-self.aligned_lightcurves[number][:,0])/2.5)
		
		else :

			delta_flux=flux_data-flux_model
			
		res_mag=-2.5*np.log10(1-delta_flux/flux_data)
		
			
		return res_mag


