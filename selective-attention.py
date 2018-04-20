import nengo 
import numpy as np
from nengo.dists import Uniform

model = nengo.Network()
min_fr = 90
max_fr = 150

with model:
    visual_stim = nengo.Node([0.5, 0, 0.5])
    positions = nengo.Node([-0.5,0,0.5])
    v1_column1 = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1, max_rates=Uniform(min_fr,max_fr))
    v1_column2 = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1, max_rates=Uniform(min_fr,max_fr))
    v1_column3 = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1, max_rates=Uniform(min_fr,max_fr))
    nengo.Connection(positions[0], v1_column1)
    nengo.Connection(positions[1], v1_column2)
    nengo.Connection(positions[2], v1_column3)
    
    controls = nengo.Node([0.5, 0.75])
    control_neurons = nengo.Ensemble(n_neurons=400, dimensions=2, radius=2, max_rates=Uniform(min_fr,max_fr))
    nengo.Connection(controls, control_neurons)
    
    routing_guide1 = nengo.Ensemble(n_neurons=600, dimensions=3, radius=2, max_rates=Uniform(min_fr,max_fr))
    routing_guide2 = nengo.Ensemble(n_neurons=600, dimensions=3, radius=2, max_rates=Uniform(min_fr,max_fr))
    routing_guide3 = nengo.Ensemble(n_neurons=600, dimensions=3, radius=2, max_rates=Uniform(min_fr,max_fr))
    
    nengo.Connection(v1_column1, routing_guide1[0])
    nengo.Connection(v1_column2, routing_guide2[0])
    nengo.Connection(v1_column3, routing_guide3[0])
    nengo.Connection(control_neurons, routing_guide1[1:])
    nengo.Connection(control_neurons, routing_guide2[1:])
    nengo.Connection(control_neurons, routing_guide3[1:])
    
    feedforward1 = nengo.Ensemble(n_neurons=600, dimensions=2, radius=2, max_rates=Uniform(min_fr,max_fr))
    feedforward2 = nengo.Ensemble(n_neurons=600, dimensions=2, radius=2, max_rates=Uniform(min_fr,max_fr))
    feedforward3 = nengo.Ensemble(n_neurons=600, dimensions=2, radius=2, max_rates=Uniform(min_fr,max_fr))
    
    def gating_func(x):
        pos = x[0]
        center = x[1]
        width = x[2]
        if pos > center + width or pos < center - width:
            return 0
        else:
            return 1
    
    nengo.Connection(visual_stim[0], feedforward1[1])
    nengo.Connection(visual_stim[1], feedforward2[1])
    nengo.Connection(visual_stim[2], feedforward3[1])
    nengo.Connection(routing_guide1, feedforward1[0], function=gating_func)
    nengo.Connection(routing_guide2, feedforward2[0], function=gating_func)
    nengo.Connection(routing_guide3, feedforward3[0], function=gating_func)

    gating1 = nengo.Ensemble(n_neurons=300, dimensions=1, radius=2, max_rates=Uniform(min_fr,max_fr))
    gating2 = nengo.Ensemble(n_neurons=300, dimensions=1, radius=2, max_rates=Uniform(min_fr,max_fr))
    gating3 = nengo.Ensemble(n_neurons=300, dimensions=1, radius=2, max_rates=Uniform(min_fr,max_fr))
    
    def MT_column_func(x):
        gating = x[0]
        stim = x[1]
        if gating > 0.5:
            return stim
        else:
            return 0
    
    nengo.Connection(feedforward1, gating1, function=MT_column_func)
    nengo.Connection(feedforward2, gating2, function=MT_column_func)
    nengo.Connection(feedforward3, gating3, function=MT_column_func)
    
    combined1 = nengo.Ensemble(n_neurons=900, dimensions=4, radius=2, max_rates=Uniform(min_fr,max_fr))
    combined2 = nengo.Ensemble(n_neurons=900, dimensions=4, radius=2, max_rates=Uniform(min_fr,max_fr))
    combined3 = nengo.Ensemble(n_neurons=900, dimensions=4, radius=2, max_rates=Uniform(min_fr,max_fr))
    
    nengo.Connection(gating1, combined1[0])
    nengo.Connection(routing_guide1, combined1[1:])
    nengo.Connection(gating2, combined2[0])
    nengo.Connection(routing_guide2, combined2[1:])
    nengo.Connection(gating3, combined3[0])
    nengo.Connection(routing_guide3, combined3[1:])
    
    MT_column1 = nengo.Ensemble(n_neurons=300, dimensions=1, radius=1, max_rates=Uniform(min_fr,max_fr))
    MT_column2 = nengo.Ensemble(n_neurons=300, dimensions=1, radius=1, max_rates=Uniform(min_fr,max_fr))
    MT_column3 = nengo.Ensemble(n_neurons=300, dimensions=1, radius=1, max_rates=Uniform(min_fr,max_fr))
    
    def strength_func1(x):
        stim = x[0]
        pos = -0.5
        center = x[2]
        width = x[3]
        diff = (center-pos)
        f = np.exp(-(diff)**2/(2*width**2))
        return stim*f
        
    def strength_func2(x):
        stim = x[0]
        pos = 0
        center = x[2]
        width = x[3]
        diff = (center-pos)
        f = np.exp(-(diff)**2/(2*width**2))
        return stim*f
    
    def strength_func3(x):
        stim = x[0]
        pos = 0.5
        center = x[2]
        width = x[3]
        diff = (center-pos)
        f = np.exp(-(diff)**2/(2*width**2))
        return stim*f
    
    nengo.Connection(combined1, MT_column1, function=strength_func1)
    nengo.Connection(combined2, MT_column1, function=strength_func1)
    nengo.Connection(combined3, MT_column1, function=strength_func1)
    
    nengo.Connection(combined1, MT_column2, function=strength_func2)
    nengo.Connection(combined2, MT_column2, function=strength_func2)
    nengo.Connection(combined3, MT_column2, function=strength_func2)
    
    nengo.Connection(combined1, MT_column3, function=strength_func3)
    nengo.Connection(combined2, MT_column3, function=strength_func3)
    nengo.Connection(combined3, MT_column3, function=strength_func3)