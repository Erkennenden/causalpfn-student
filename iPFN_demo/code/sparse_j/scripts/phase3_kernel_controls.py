"""Radial-basis kernel features."""
import numpy as np


def rbf(x,c,length):
    return np.exp(-np.maximum(np.sum(x*x,1)[:,None]+np.sum(c*c,1)[None,:]-2*x@c.T,0)/(2*length**2))

