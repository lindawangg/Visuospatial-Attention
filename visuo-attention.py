import nengo 

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
    
    intermediate = nengo.Ensemble(n_neurons=100, dimensions=6)
    nengo.Connection(control, intermediate[:3])
    nengo.Connection(signal, intermediate[3:])
    
    # TO-DO: change this placeholder function 
    # Want: depending on the location of the signal in the
    # receptive field, response will differ
    # This depends on sigma_att
    def response_func(x):
        if x[2] > x[3]:
            return x[2]*x[3]
        else:
            return 0
    response = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(intermediate, response, function=response_func)
    
    
    