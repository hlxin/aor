from catmap import ReactionModel
from string import Template
from catmap import analyze

mkm_file = 'AOR.mkm'
model = ReactionModel(setup_file=mkm_file)
model.output_variables+=['production_rate', 'free_energy', 'selectivity']
model.run()

#sa = analyze.ScalingAnalysis(model)
#sa.plot(save='scaling.pdf')

vm = analyze.VectorMap(model)
vm.plot_variable = 'rate' #tell the model which output to plot
vm.log_scale = True #rates should be plotted on a log-scale
vm.min = 1e-25 #minimum rate to plot
vm.max = 1e10 #maximum rate to plot
vm.colormap = "jet"
vm.plot(save='rate.pdf') #draw the plot and save it as "rate.pdf"

vm = analyze.VectorMap(model)
vm.plot_variable = 'production_rate'
vm.log_scale = True
vm.min = 1e-20
vm.max = 1e5
vm.threshold = 1e-20 #anything below this is considered to be 0
vm.colormap = "jet"
fig = vm.plot(save='prodrate.png')


vm = analyze.VectorMap(model)
vm.plot_variable = 'coverage'
vm.min = 0
vm.max = 1
vm.log_scale = False
fig = vm.plot(save='coverage.png')

