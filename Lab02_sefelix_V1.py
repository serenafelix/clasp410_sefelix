#!/usr/bin/env python3
"""
This is my code for Lab 02 for CLIMATE410.

This script contains code to implement and compare two numerical integrators (the Euler and eighth order solvers).

The integrators are applied to Lotka-Volterra competition and predator-prey models.

Along with the integrators, there is code to run simulations and plot the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

############RUN THIS CODE BEFORE ANY QUESTIONS#################

####Set Up Competition Equations
def comp_model(t, N, a=1, b=2, c=1, d=3):
    """
    This function solves the Lotka-Volterra competition equations for two species.
    Parameters:
    t : time (will not be used but is required for ODE solvers)
    N : list of values [N1, N2], where N1 and N2 are the populations of the two species
    a : coefficient representing reproduction rate of species 1
    b : coefficient representing effect of species 2 on species 1
    c : coefficient representing reproduction rate of species 2
    d : coefficient representing effect of species 1 on species 2
    Returns:
    dN1, dN2 : time derivatives for N1 and N2 respectively.
    """
    N = np.asarray(N, dtype=float) #convert N to a numpy array
    N1, N2 = N[0], N[1]
    dN1 = a*N1*(1.0 - N1)-b*N1*N2 #Lotka-Volterra competition equation
    dN2 = c*N2*(1.0 - N2)-d*N1*N2 #Lotka-Volterra competition equation
    return np.array([dN1, dN2]) #return time derivatives as an array

####Set up Predator-Prey Equations
def predprey_model(t, N, a=1, b=2, c=1, d=3):
    """
    This function solves the Lotka-Volterra predator-prey equations for two species.
    Parameters:
    t : time (will not be used but is required for ODE solvers)
    N : list of values [N1, N2], where N1 and N2 are the populations of the two species
    a : coefficient representing reproduction rate of prey species 1
    b : coefficient representing effect of predator species 2 on prey species 1
    c : coefficient representing reproduction rate of predator species 2 per prey eaten
    d : coefficient representing natural death rate of predator species 2
    Returns:
    dN1, dN2 : time derivatives for N1 and N2 respectively.
    """
    N = np.asarray(N, dtype=float) #convert N to a numpy array
    N1, N2 = N[0], N[1] 
    dN1 = a*N1-b*N1*N2 #Lotka-Volterra predator-prey equation
    dN2 = -c*N2+d*N1*N2 #Lotka-Volterra predator-prey equation
    return np.array([dN1, dN2]) #return time derivatives as an array


#Model the Euler method
def euler(compute, N1_init=0.5, N2_init=0.5, t_step=10, t_end=100.0, a=1, b=2, c=1, d=3):
    """
    This function implements the Euler method for solving Loktka-Volterra competition equations.
    Parameters:
    compute : function that computes the derivatives, should accept (t, [N1, N2], a, b, c, d)
    N1_init : initial population of species 1
    N2_init : initial population of species 2
    t_step : time step for the Euler method
    t_end : end time for the simulation
    a, b, c, d : coefficients for the Lotka-Volterra equations (specific descriptions can be found in the comp_models function)
    Returns:
    times : array of times
    N1 : array of populations for species 1 over time
    N2 : array of populations for species 2 over time
    """

    #create time steps
    times = np.arange(0.0, t_end + 1, t_step) #create time array
    num_steps = times.size #find number of time steps

    N1 = np.zeros(num_steps, dtype=float) #create N1 array
    N2 = np.zeros(num_steps, dtype=float) #create N2 array
    N1[0] = N1_init #set initial condition
    N2[0] = N2_init #set initial condition

    for i in range(1, num_steps): #loop over time steps
        dN1, dN2 = compute(times[i-1], [N1[i-1], N2[i-1]], a, b, c, d) #compute derivatives
        N1[i] = N1[i-1] + t_step * dN1 #update N1 using Euler step
        N2[i] = N2[i-1] + t_step * dN2 #update N2 using Euler step

    return times, N1, N2

#Model the DOP853 method
def rk(compute, N1_init=0.5, N2_init=0.5, t_steps=0.01, t_end=100.0,
              a=1, b=2, c=1, d=3):
    """
    This function implements an 8th order Runge-Kutta solver for two species' competition.
    Parameters:
    compute : function that computes the derivatives, should accept (t, [N1, N2], a, b, c, d)
    N1_init : initial population of species 1
    N2_init : initial population of species 2 
    t_steps : maximum time step
    t_end : end time for the simulation
    a, b, c, d : coefficients for the Lotka-Volterra equations (specific descriptions can be found in the comp_models function)
    Returns:
    times : array of times
    N1 : array of populations for species 1 over time
    N2 : array of populations for species 2 over time
    """
    #use solve_ivp with the 'DOP853' method
    result = solve_ivp(compute, [0.0, t_end], [N1_init, N2_init], args=(a, b, c, d), method='DOP853', max_step=t_steps)
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :] #unpack results into variables
    return time, N1, N2

############BEGIN QUESTIONS#################

#Q1: How does the performance of the Euler method solver compare to the 8th-order DOP853 method for both sets of equations? 
#Create plots to validate and compare the two methods
if __name__ == "__main__":
    #run both models and assign variables to their outputs
    t_e, N1_e, N2_e = euler(comp_model, N1_init=0.3, N2_init=0.6, t_step=1, t_end=100.0) #euler method
    t_rk, N1_rk, N2_rk = rk(comp_model, N1_init=0.3, N2_init=0.6, t_steps=1, t_end=100.0) #DOP853 method
    #make the comparison plot
    plt.figure(figsize=(8, 4))
    #add Euler data (make the lines solid)
    plt.plot(t_e, N1_e, label='N1 Euler', linestyle='-')
    plt.plot(t_e, N2_e, label='N2 Euler', linestyle='-')
    #add RK data (make the lines dashed)
    plt.plot(t_rk, N1_rk, label='N1 RK', linestyle='--')
    plt.plot(t_rk, N2_rk, label='N2 RK', linestyle='--')
    plt.xlabel('Time (years)') #label x axis
    plt.ylabel('Carrying Capacity') #label y axis
    plt.title('Comparison of Euler and DOP853 Methods for Lotka-Volterra Competition Model')
    plt.legend() #add legend
    plt.show() #show the plot without saving it (I saved plots from the pop out window once previewed)

#Q2: How do the initial conditions and coefficient values affect the final result and general behavior of the two species? 
if __name__ == "__main__":
    #run both models and assign variables to their outputs
    t_e, N1_e, N2_e = euler(comp_model, N1_init=0.5, N2_init=0.5, t_step=1, t_end=100.0,
                            a=2, b=2, c=3, d=3)
    t_rk, N1_rk, N2_rk = rk(comp_model, N1_init=0.5, N2_init=0.5, t_steps=1, t_end=100.0,
                            a=2, b=2, c=3, d=3)
    #make the comparison plot
    plt.figure(figsize=(8, 4))
    #add Euler data (make the lines solid)
    plt.plot(t_e, N1_e, label='N1 Euler', linestyle='-')
    plt.plot(t_e, N2_e, label='N2 Euler', linestyle='-')
    #add RK data (make the lines dashed)
    plt.plot(t_rk, N1_rk, label='N1 RK', linestyle='--')
    plt.plot(t_rk, N2_rk, label='N2 RK', linestyle='--')
    plt.xlabel('Time (years)') #label x axis
    plt.ylabel('Carrying Capacity') #label y axis
    plt.title('Reaching Equilibrium with Lokta Volterra Competition Equations') #add title
    plt.legend() #add legend
    plt.show() #show the plot without saving it (I saved plots from the pop out window once previewed)

#Q3: How do the initial conditions and coefficient values affect the final result and general behavior of the two species? 
if __name__ == "__main__":
    #run both models and assign variables to their outputs
    t_e, N1_e, N2_e = euler(predprey_model, N1_init=0.5, N2_init=0.5, t_step=0.05, t_end=100.0,
                            a=1, b=1, c=1, d=0.5)
    t_rk, N1_rk, N2_rk = rk(predprey_model, N1_init=0.5, N2_init=0.5, t_steps=0.05, t_end=100.0,
                            a=1, b=1, c=1, d=0.5)
    
    #make the comparison plot (carrying capacity vs. time)
    plt.figure(figsize=(8, 4))
    #add Euler data (make the lines solid)
    plt.plot(t_e, N1_e, label='N1 Euler', linestyle='-')
    plt.plot(t_e, N2_e, label='N2 Euler', linestyle='-')
    #add RK data (make the lines dashed)
    plt.plot(t_rk, N1_rk, label='N1 RK', linestyle='--')
    plt.plot(t_rk, N2_rk, label='N2 RK', linestyle='--')
    plt.xlabel('Time (years)') #label x axis
    plt.ylabel('Carrying Capacity') #label y axis
    plt.title('Experimenting with Lokta Volterra Predator-Prey Equations') #add title
    plt.legend() #add legend
    plt.show() #show the plot without saving it (I saved plots from the pop out window once previewed)

    #make the phase diagram (N1 vs N2)
    plt.figure(figsize=(8, 4)) #create figure and set size
    plt.plot(N1_e, N2_e, label='Euler') #add Euler data
    plt.plot(N1_rk, N2_rk, label='RK') #add RK data
    plt.xlabel('N1 Population') #label x axis
    plt.ylabel('N2 Population') #label y axis
    plt.title('Phase Diagram of Predator-Prey Model') #add title
    plt.legend() #add legend
    plt.show() #show the plot without saving it (I saved plots from the pop out window once previewed)
