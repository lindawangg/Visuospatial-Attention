import nengo 
import numpy as np

model = nengo.Network()

with model:
    stim_controls = nengo.Node([0,0,0]) # x,y,attention
    # controls the receptive field size
    control = nengo.Ensemble(n_neurons=300, dimensions=3, radius=2)
    nengo.Connection(stim_controls, control)
    
    stim_signal = nengo.Node([0,0,0]) # x,y,strength of signal
    # controls the receptive field of the signal size 
    signal = nengo.Ensemble(n_neurons=600,dimensions=3, radius=1)
    nengo.Connection(stim_signal,signal)
    
    pos_rf = nengo.Ensemble(n_neurons=300,dimensions=3, radius=2)
    control_signal = nengo.Ensemble(n_neurons=600, dimensions=6, radius=2)
    nengo.Connection(control,control_signal[:3])
    nengo.Connection(signal,control_signal[3:])
    
    def pos_func(x):
        if x[2] > 0:
            return x[3],x[4],0.75
        else:
            return x[0],x[1],1.0

    nengo.Connection(control_signal,pos_rf,function=pos_func)
    
    intermediate = nengo.Ensemble(n_neurons=1200, dimensions=6)
    nengo.Connection(pos_rf, intermediate[:3])
    nengo.Connection(signal, intermediate[3:])
    
    # depending on the location of the signal in the receptive field, 
    # response will differ. This depends on sigma_att
    def response_func(x):
        a = x[0]-x[3]
        b = x[1]-x[4]
        c = np.sqrt(a**2+b**2)
        f = np.exp(-(c)**2/(2**2))
        return f*x[5]
    response = nengo.Ensemble(n_neurons=1000, dimensions=1)
    nengo.Connection(intermediate, response, function=response_func)
    
    
    