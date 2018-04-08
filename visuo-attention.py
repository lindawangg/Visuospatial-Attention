import nengo 
import numpy as np

model = nengo.Network()

with model:
    stim_controls = nengo.Node([0,0,0]) # x,y, radius
    # controls the receptive field size
    control = nengo.Ensemble(n_neurons=100, dimensions=3)
    nengo.Connection(stim_controls, control)
    
    stim_signal = nengo.Node([0,0,0]) # x,y, radius
    # controls the receptive field of the signal size 
    signal = nengo.Ensemble(n_neurons=100,dimensions=3)
    nengo.Connection(stim_signal,signal)
    
    intermediate = nengo.Ensemble(n_neurons=100, dimensions=4)
    nengo.Connection(control[:2], intermediate[:2])
    nengo.Connection(signal[:2], intermediate[2:])
    
    # TO-DO: change this placeholder function 
    # Want: depending on the location of the signal in the
    # receptive field, response will differ
    # This depends on sigma_att
    def response_func(x):
        sigma_att = 1
        return np.exp(-(x[0]-x[2])**2/(2*sigma_att**2))*np.exp(-(x[1]-x[3])**2/(2*sigma_att**2))
    response = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(intermediate, response, function=response_func)
    
    
    