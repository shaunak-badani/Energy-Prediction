"""
get_aev_params(device)
get_complete_network(device)
"""
import numpy as np
import torch
import torchani
import os
import math
import torch.utils.tensorboard
import tqdm
import pickle

import matplotlib
import matplotlib.pyplot as plt
from sys import getsizeof
import h5py

def get_aev_params(device):
	Rcr = 5.2000e+00
	Rca = 3.5000e+00
	EtaR = torch.tensor([1.6000000e+01], device=device)
	ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
	Zeta = torch.tensor([3.2000000e+01], device=device)
	ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
	EtaA = torch.tensor([8.0000000e+00], device=device)
	taA = torch.tensor([8.0000000e+00], device=device)
	ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
	species_order = ['H', 'C', 'N', 'O']
	num_species = len(species_order)
	aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
	energy_shifter = torchani.utils.EnergyShifter(None)

	return energy_shifter, aev_computer


def get_complete_network(aev_computer, device, return_networks=False):
	aev_dim = aev_computer.aev_length
	H_network = torch.nn.Sequential(
	    torch.nn.Linear(aev_dim, 160),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(160, 128),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(128, 96),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(96, 1)
	)

	C_network = torch.nn.Sequential(
	    torch.nn.Linear(aev_dim, 144),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(144, 112),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(112, 96),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(96, 1)
	)

	N_network = torch.nn.Sequential(
	    torch.nn.Linear(aev_dim, 128),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(128, 112),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(112, 96),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(96, 1)
	)

	O_network = torch.nn.Sequential(
	    torch.nn.Linear(aev_dim, 128),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(128, 112),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(112, 96),
	    torch.nn.CELU(0.1),
	    torch.nn.Linear(96, 1)
	)

	nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
	model = torchani.nn.Sequential(aev_computer, nn).to(device)
	if return_networks:
		return H_network, C_network, N_network, O_network, model, nn
	return model, nn

if __name__ == "__main__":
	device = 'cpu'
	energy_shifter, aev_computer = get_aev_params(device)
	model, nn = get_complete_network(aev_computer, device)
	print(model, nn)
