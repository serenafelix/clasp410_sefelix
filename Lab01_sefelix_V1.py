#!/usr/bin/env python3
"""
This is the code portion of my lab 01 assignment for CLIMATE410.
It contains code to generate an energy balance model expressed as an array,
and how to using that model to answer the lab questions.

Additionally, there is code for a series of plots to visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt


#constants to be used further down
sbc = 5.67e-8 #stefan-boltzmann constant W/m^2/K^4
albedo = 0.33 #Earth's albedo value

#Set up the matrix:
def energy_balance_model(layers, emissivity=1.0, albedo=0.33, solar_flux=1350):
    """
    This function sets up and runs a simple energy balance model, returning the surface
    temperature of a planet with N layers of atmosphere.
    
    Parameters:
    layers : The number of layers in the simulated atmosphere
    emissivity : The emissivity of the atmosphere
    albedo : The albedo of the Earth
    solar_flux : The amount of shortwave solar radiation in W/m^2
    
    Returns:
    surface_temp: Array of temperatures through a planet's atmosphere
    """
#Add in the dimensions of the matrix(N+1 x N+1):
    A = np.zeros([layers+1, layers+1]) #matrix of coefficients
    b = np.zeros(layers+1) #vector of constants
    b[0] = -solar_flux/4 * (1-albedo)
    
    # Populate A using our rules
    for i in range(layers+1):
        for j in range(layers+1):
            if i == j:
                # for diagonals, -1 for surface or -2 for atmosphere layers
                A[i, j] = -1 if i == 0 else -2
            else: #for non diagnonals, using the pattern from class
                m = np.abs(j - i) - 1
                A[i, j] = emissivity * (1 - emissivity) ** m

    # adjust for surface emissivity
    if emissivity != 0:
        A[0, 1:] /= emissivity

    # solve for fluxes
    InverseA = np.linalg.inv(A) #solve for inverse of A
    AtmosFlux = np.matmul(InverseA, b) #matrix multiplication to get fluxes

    # convert flux (W/m^2) to temperature using Stefan-Boltzmann law
    with np.errstate(invalid='ignore', divide='ignore'): #needed to add this line to avoid warnings and NaNs
        surface_temp = (AtmosFlux / sbc / emissivity) ** 0.25 #find the temperature of each layer
        # solve for surface temperature separately to avoid divide by zero if emissivity=0
        surface_temp[0] = (AtmosFlux[0] / sbc) ** 0.25

    return surface_temp
print(energy_balance_model(4, 0.5, albedo, 1350))

# How does the surface temperature of the Earth depend on emissivity and number of layers?

#Experiment One: Varying Emissivity with 1 layer atmosphere
E = [0.2, 0.5, 0.8, 1.0] #different emissivity values to test
surface_temps = [] #list to hold surface temperatures
for e in E:
    temps = energy_balance_model(1, e, albedo, 1350) #1 layer atmosphere with varied emissivites compared to just one
    print(f'emissivity={e}:', temps) #print a table of results
    surface_temps.append(temps[0]) #separate list of just surface temperatures
#Plot the surface temperatures
plt.plot(E, surface_temps, marker='o', linestyle='-') #plot surface temps vs emissivity with data points included
plt.xlabel('Emissivity') #label x axis
plt.ylabel('Surface Temperature (K)') #label y axis
plt.title('1 Layer Atmosphere Surface Temperature vs Emissivity') #give the plot a title
plt.grid(True) #add grid lines
plt.show() #show the plot without saving it (I saved plots from the pop out window once previewed)

#Experiment Two: Varying Atmospheric Layers with Constant Emissivity
layers = [1, 2, 3, 4, 5] #different number of layers to test
surface_temps_layers = [] #list to hold surface temperatures
for l in layers:
    temps = energy_balance_model(l, 0.255, albedo, 1350) #constant emissivity of 0.255 with varied layers
    print(f'layers={l}:', temps) #print a table of results
    surface_temps_layers.append(temps[0]) #separate list of just surface temperatures
#Plot the surface temperatures
plt.plot(layers, surface_temps_layers, marker='o', linestyle='-') #plot surface temps vs layers with data points included
plt.xlabel('Number of Atmospheric Layers (Altitude)') #label x axis
plt.ylabel('Surface Temperature (K)') #label y axis
plt.title('Surface Temperature vs Number of Atmospheric Layers (Emissivity=0.255)') #give the plot a title
plt.grid(True) #add grid lines
plt.show() #show the plot without saving it (I saved plots from the pop out window once previewed)

#How mant atmospheric layers do we expect on the planet Venus?

Venus_layers = [1,5,10,15,20,25,30] #different number of layers to test
surface_temps_venus = [] #list to hold surface temperatures
for v in Venus_layers:
    temps = energy_balance_model(v, 1, albedo, 2600) #Venus parameters (assuming albedo is the same as Earth)
    print(f'layers={v}:', temps) #print a table of results
    surface_temps_venus.append(temps[0]) #separate list of just surface temperatures
#Plot the surface temperatures
plt.plot(Venus_layers, surface_temps_venus, marker='o', linestyle='-') #plot surface temps vs layers with data points included
plt.xlabel('Number of Layers with Emissivity of One') #label x axis
plt.ylabel('Surface Temperature (K)') #label y axis
plt.title('Surface Temperature vs Number of Layers on Venus') #give the plot a title
plt.grid(True) #add grid lines
plt.show() #show the plot without saving it (I saved plots from the pop out window once previewed)

#What would the Earth’s surface temperature be under a nuclear winter scenario?

#Set up the new matrix:
def energy_balance_model_nuclear(N=5, emis=1.0, a=0.33, solar=1350):
    """
    This function sets up and runs a simple energy balance model, returning the surface
    temperature of a planet with N layers of atmosphere.
    
    Parameters:
    N : The number of layers in the simulated atmosphere (5)
    emis : The emissivity of the atmosphere (1.0)
    a : The albedo of the Earth
    solar : The amount of shortwave solar radiation in W/m^2
    
    Returns:
    nuclear_temp: Array of temperatures through Earth's atmosphere, where the top layer has an emissivity of one
    """
#Add in the dimensions of the matrix(N+1 x N+1):
    A = np.zeros([N+1, N+1]) #matrix of coefficients
    b = np.zeros(N+1) #vector of constants
    b[0] = -solar/4 * (1-a)
    
    # Populate A using our rules
    for i in range(N+1):
        for j in range(N+1):
            if i == j:
                # for diagonals, -1 for surface or -2 for atmosphere layers
                A[i, j] = -1 if i == 0 else -2
            else: #for non diagnonals, using the pattern from class
                m = np.abs(j - i) - 1
                A[i, j] = emis * (1 - emis) ** m

    # adjust for surface emissivity
    if emis != 0:
        A[0, 1:] /= emis

    # solve for fluxes
    InverseA = np.linalg.inv(A) #solve for inverse of A
    AtmosFlux = np.matmul(InverseA, b) #matrix multiplication to get fluxes

    # convert flux (W/m^2) to temperature using Stefan-Boltzmann law
    # Initialize the output array first
    nuclear_temp = np.zeros(N+1)
    with np.errstate(invalid='ignore', divide='ignore'):
        # compute baseline temperatures using the provided emissivity for atmosphere/surface
        if emis != 0:
            nuclear_temp = (AtmosFlux / sbc / emis) ** 0.25
        else:
            # avoid division by zero if emis == 0
            nuclear_temp = (AtmosFlux / sbc) ** 0.25

        # top layer (index N) is assumed to have emissivity = 1 (absorbs all incoming)
        # use emissivity=1 for that layer when converting flux to temperature
        nuclear_temp[N] = (AtmosFlux[N] / sbc) ** 0.25

    return nuclear_temp
print(energy_balance_model_nuclear(5, 1.0, albedo, 1350))