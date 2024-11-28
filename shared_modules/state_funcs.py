import numpy as np
from numba import njit
import torch

@njit
def post_decision_state(a, m):

    b = a*m

    return b

@njit
def state_transition(par, b, R, p, xiR, xip):

    Rplusone = (R**par.rhoR)*xiR
    pplusone = (p**par.rhop)*xip

    mplusone = b*Rplusone + pplusone

    return mplusone, Rplusone, pplusone

def state_transition_notjitted(par, b, R, p, xiR, xip):

    Rplusone = (R**par.rhoR)*xiR
    pplusone = (p**par.rhop)*xip

    mplusone = b*Rplusone + pplusone

    return mplusone, Rplusone, pplusone

def post_decision_state_notjitted(a, m):

    b = a*m

    return b