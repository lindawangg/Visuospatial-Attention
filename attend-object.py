import nengo 
import numpy as np
from nengo.dists import Uniform

model = nengo.Network()

with model:
    visual_stim = nengo.Node([0.5, 0, 0.5])
    positions = nengo.Node([-0.5,0,0.5])
    v1_column1 = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1, max_rates=Uniform(90,120))
    v1_column2 = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1, max_rates=Uniform(90,120))
    v1_column3 = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1, max_rates=Uniform(90,120))
    nengo.Connection(positions[0], v1_column1)
    nengo.Connection(positions[1], v1_column2)
    nengo.Connection(positions[2], v1_column3)
    
    controls = nengo.Node([0.5, 0.75])
    control_neurons = nengo.Ensemble(n_neurons=400, dimensions=2, radius=1, max_rates=Uniform(90,120))
    nengo.Connection(controls, control_neurons)
    
    MT_terminal1 = nengo.Ensemble(n_neurons=600, dimensions=3, radius=1, max_rates=Uniform(90,120))
    MT_terminal2 = nengo.Ensemble(n_neurons=600, dimensions=3, radius=1, max_rates=Uniform(90,120))
    MT_terminal3 = nengo.Ensemble(n_neurons=600, dimensions=3, radius=1, max_rates=Uniform(90,120))
    
    nengo.Connection(v1_column1, MT_terminal1[0])
    nengo.Connection(v1_column2, MT_terminal2[0])
    nengo.Connection(v1_column3, MT_terminal3[0])
    nengo.Connection(control_neurons, MT_terminal1[1:])
    nengo.Connection(control_neurons, MT_terminal2[1:])
    nengo.Connection(control_neurons, MT_terminal3[1:])
    
    gating1 = nengo.Ensemble(n_neurons=900, dimensions=3, radius=2, max_rates=Uniform(90,120))
    gating2 = nengo.Ensemble(n_neurons=900, dimensions=3, radius=2, max_rates=Uniform(90,120))
    gating3 = nengo.Ensemble(n_neurons=900, dimensions=3, radius=2, max_rates=Uniform(90,120))
    
    def gating_func(x):
        pos = x[0]
        center = x[1]
        width = x[2]
        if pos > center + width or pos < center - width:
            return 0
        else:
            return 1

    def strength_func(x):
        pos = x[0]
        center = x[1]
        width = x[2]
        diff = (center-pos)
        f = np.exp(-(diff)**2/(2*width**2))
        return f
    
    nengo.Connection(visual_stim[0], gating1[1])
    nengo.Connection(visual_stim[1], gating2[1])
    nengo.Connection(visual_stim[2], gating3[1])
    nengo.Connection(MT_terminal1, gating1[0], function=gating_func)
    nengo.Connection(MT_terminal2, gating2[0], function=gating_func)
    nengo.Connection(MT_terminal3, gating3[0], function=gating_func)
    nengo.Connection(MT_terminal1, gating1[2], function=strength_func)
    nengo.Connection(MT_terminal2, gating2[2], function=strength_func)
    nengo.Connection(MT_terminal3, gating3[2], function=strength_func)
    
    MT_column1 = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1, max_rates=Uniform(90,120))
    MT_column2 = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1, max_rates=Uniform(90,120))
    MT_column3 = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1, max_rates=Uniform(90,120))
    
    def MT_column_func(x):
        gating = x[0]
        stim = x[1]
        f = x[2]
        if gating > 0.5:
            return stim*f
        else:
            return 0
    
    nengo.Connection(gating1, MT_column1, function=MT_column_func)
    nengo.Connection(gating2, MT_column2, function=MT_column_func)
    nengo.Connection(gating3, MT_column3, function=MT_column_func)
    
    MT = nengo.Ensemble(n_neurons=100, dimensions=1, radius=1,max_rates=Uniform(90,120))
    
    def connect_MT(x):
        return x[0]+x[1]+x[2]
    
    intermediate = nengo.Ensemble(n_neurons=600, dimensions=3, radius=1, max_rates=Uniform(90,120))
    nengo.Connection(MT_column1, intermediate[0])
    nengo.Connection(MT_column2, intermediate[1])
    nengo.Connection(MT_column3, intermediate[2])
    nengo.Connection(intermediate, MT, function=connect_MT)
    
