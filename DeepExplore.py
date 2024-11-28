# Global modules
import torch
import os
import pickle
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from types import SimpleNamespace
from torch.autograd import gradcheck
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
from consav import quadrature

local_modules_path = [os.getcwd() + '/shared_modules', os.getcwd() + '/VFI_modules']
for path in local_modules_path:
    sys.path.append(path)

# Local modules
import utility
import state_funcs
import load_and_dump
import simulate
import euler_errors

class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, Nneurons):

        super(Policy, self).__init__()

        self.layers = nn.ModuleList([None]*(Nneurons.size + 1))

        # 1. Input layer taking the state
        self.layers[0] = nn.Linear(state_dim, Nneurons[0])

        # 2. Hidden layers
        for i in range(1, len(self.layers)-1):
            self.layers[i] = nn.Linear(Nneurons[i-1], Nneurons[i])

        # 3. Output layer
        self.layers[-1] = nn.Linear(Nneurons[-1], action_dim)

    def forward(self, state):
        # 1. Compute neurons after input layer
        s = torch.relu(self.layers[0](state))

        # 2. Compute neurons in all hidden layers
        for i in range(1, len(self.layers)-1):
            s = torch.relu(self.layers[i](s))

        # 3. Compute neurons in output layer
        s = torch.sigmoid(self.layers[-1](s))

        return s

class DeepExplore():

    def __init__(self):

        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.sim = SimpleNamespace()

        self.define_parameters()

    def define_parameters(self):

        par = self.par

        # 1. Preferences
        par.beta = 0.99

        # 2. Shocks
        par.muR = -0.00005
        par.mup = -0.00005
        par.sigmaR = 0.01
        par.sigmap = 0.01
        par.rhoR = 0.9
        par.rhop = 0.9

        # 3. Number of states, shocks, periods and households
        par.st = 3
        par.sh = 2
        par.T = 50
        par.N_eval = 50_000
        par.N_training = 5_000

        # 4. Simulation specifications
        par.m_int_min = 1.0
        par.m_int_max = 3.0
        par.R_int_min = 1.05
        par.R_int_max = 1.15
        par.p_int_min = 1.0
        par.p_int_max = 3.0

        # 5. Training specifications
        par.n_eval_iterations = 1000
        par.n_iterations = 50

        par.neurons = np.array([450, 450])
        par.lr_list = [0.00001]*par.n_eval_iterations
        
        par.early_stopping_threshold = 1e-8
        par.early_stopping_check_eval_iteration = 20

        par.stateclamp = True
        par.statebounds = 1e-5, 1e3
        par.clamp_a = 0.0, 1 - 1e-3 # Very small number as upper bound for a, a=1 means infinite disutility to be avoided

        par.grad_clip_bool = True
        par.grad_clip_value = 10

        par.seed_int_weights = 0

        # 6. Data type
        par.dtype = torch.float32 # GPUs are optimized for 32bit linear algebra
    
    def int_nn(self, seed, device, **kwargs):

        par = self.par
        sol = self.sol

        neurons = kwargs.get('neurons', par.neurons)
        assert isinstance(neurons, np.ndarray)

        assert isinstance(seed, int)

        print(f'A neural network with the following structure is to be initialized:')
        for layer, neurons_layer in enumerate([*neurons]):
            print(f'Hidden layer {layer}:\t\t {neurons_layer} neurons')
        
        # 1. Set seeds and dtype
        torch.set_default_dtype(par.dtype)
        torch.manual_seed(seed)

        # 2. Initialize nn on device
        nn = Policy(par.st + 1, 1, neurons).to(device)

        # 3. Store nn as initial nn
        sol.trained_nns['int_nn'] = nn
    
    def train_nn_iterations(self, nn, lr, n_iterations, eval_iteration, train_dict, test_dict, device):

        """
        Args:

            nn:                     nn to be trained

            lr:                     Learning rate used in training throughout iterations

            n_iterations:           Number of iterations to do

            eval_iteration:         Which evaluation iteration, the iterations are done in

            train_dict:             Dict with training seeds
                                    Key                     Value
                                    seed_initial_states     list with len(n_iterations)
                                    seed_shocks             list with len(n_iterations)

            test_dict:              Dict containing two unique test seeds
                                    Key                     Value
                                    seed_initial_states     list with one unique seed
                                    seed_shocks             list with one unique seed
            
            device:                 Device (always specify)

        Returns:

            in_sample_losses        (n_iterations,) torch.tensor with in-sample losses

            out_of_sample_losses    (n_iterations,) torch.tensor with out-of-sample losses based on unique seeds

            comp_time               (n_iterations,) torch.tensor with total computation time after each iteration
        
        """

        par = self.par
        sol = self.sol

        # 1. Memory
        in_sample_losses = torch.zeros(n_iterations, dtype=par.dtype, device=device)
        out_of_sample_eval = torch.zeros(1, dtype=par.dtype, device=device)
        comp_time = torch.zeros(n_iterations, dtype=par.dtype, device=device)

        # 2. nn in training mode
        nn.train()

        # 3. Call optimizer
        optimizer = torch.optim.Adam(nn.parameters(), lr = lr)

        # 3.1 Import ADAM from earlier epochs
        if eval_iteration > 0:
            sol.ADAM_temp['param_groups'][0]['lr'] = lr
            optimizer.load_state_dict(sol.ADAM_temp)

        # 4. Timestamp
        timestamp = time.time()

        # 5. Iterate
        for iteration in range(n_iterations):

            # 5.1 Zero out grads
            optimizer.zero_grad()

            # 5.2 Compute losses + store
            in_sample_loss = simulate.simulate_model(par, train_dict, nn, iteration=iteration, eval_iteration=eval_iteration, test=False, device=device, tensor_bool=True)[0]

            in_sample_losses[iteration] = in_sample_loss.detach()

            # 5.3 Backpropagate
            in_sample_loss.backward()

            # 5.4 Clip gradient
            if par.grad_clip_bool:
                torch.nn.utils.clip_grad_value_(nn.parameters(), par.grad_clip_value)
            
            # 5.5 Take step
            optimizer.step()

            # 5.6 Store time
            with torch.no_grad():
                comp_time[iteration] = time.time() - timestamp

            # 5.7 Print (make function)
            print(f'\n\tIteration {iteration}\t   Eval_iteration {eval_iteration}\n\t{"-"*37}')
            print(f'\tIn-sample loss:\t\t\t{in_sample_loss:.2f}')
            print(f'\tTime gone:\t\t\t{comp_time[iteration]:.2f}')

            if eval_iteration > 0:
                print(f'\tLast out-of-sample eval:\t{sol.out_of_sample_evals[eval_iteration-1]:.2f}')

            if iteration > 0:
                print(f'\n\tIn-sample progress:\t\t{in_sample_losses[iteration] - in_sample_losses[iteration - 1]:.2f}')
                print(f'\tTime for iteration:\t\t{comp_time[iteration] - comp_time[iteration - 1]:.2f}\n\n')
            
        # 6. Save ADAM optimizer
        sol.ADAM_temp = optimizer.state_dict()

        # 7. Training done
        nn.train(mode=False)

        # 8. Make evaluation
        with torch.no_grad():
            out_of_sample_eval = simulate.simulate_model(par, test_dict, nn, iteration=iteration, eval_iteration=eval_iteration, eval=True, device=device, tensor_bool=True)[0]
            print(f'\tOut-of-sample evaluation:\t\t{out_of_sample_eval:.2f}')

        # 8. Store nn and print
        print(f'\tStoring trained nn of eval_iteration {eval_iteration} as trained_nn in dict trained_nns namespace sol')
        sol.trained_nns['trained_nn'] = nn

        return in_sample_losses, out_of_sample_eval, comp_time

    def train_nn_evals(self, seed_dict, device, **kwargs):

        """
        Args:
        
            seed_dict:                Dict with seeds
                                      Key                 Value
                                      int_weights         1 seed
                                      train               two dicts with list with len(n_eval_iterations) of lists with len(n_iterations) of unique seeds
                                      test                one dict
            
            device:                   Device (always specify)
        
        Kwargs:
            n_iterations:             Specify number of iterations per evaluation, otherwise use par

            n_eval_iterations:        Specify number of iterations where model is evaluated, otherwise use par

            lr_list:                  List of len(n_eval_iterations) with learning rates for EACH eval, otherwise use par

            old_nn:                   if None, nn is initialized
                                      if specified, specified nn is used

            dump_eval_iteration:      Specify indices of evals to dump:
                                      Tensor                    size
                                      in_samle_losses           (n_eval_iterations, n_iterations)
                                      out_of_sample_eval        (n_eval_iterations, n_iterations)
                                      comp_time                 (n_eval_iterations, n_iterations)

            euler_errors_eval_iteration: List of booleans with len(n_eval_iterations), 1 = True means that Euler Errors are computed (takes long time)

            run_name:                 Name of session

            neurons:                  Specified number of neurons, format of np.array([neurons_hidden_layer_1, neurons_hidden_layer_2, ..., neurons_hidden_layer_n])
        
        Guide to making dict:

        dict:
            'int_weights':
                seed for weights
            'train': dict
                'seed_initial_states': list of len(n_eval_iterations)
                    list of len(n_iterations)
                        seed for initial states
                'seed_shocks': list of len(n_eval_iterations)
                    list of len(n_iterations)
                seed for shocks
            'test': dict
                'seed_initial_states' : unique test seed for initial states 
                'seed_shocks' : unique test seed for shocks
        """

        par = self.par
        sim = self.sim
        sol = self.sol

        # 1. Allocate memory by using kwargs
        sol.trained_nns = {}
        sol.ADAM_temp = None

        n_iterations = kwargs.get('n_iterations', par.n_iterations)
        n_eval_iterations = kwargs.get('n_eval_iterations', par.n_eval_iterations)

        sol.in_sample_losses = torch.zeros((n_eval_iterations, n_iterations), dtype=par.dtype, device=device)
        sol.out_of_sample_evals = torch.zeros(n_eval_iterations, dtype=par.dtype, device=device)
        sol.comp_time = torch.zeros((n_eval_iterations, n_iterations), dtype=par.dtype, device=device)

        # 2. Check kwargs

        # 2.1 seed_dict
        assert isinstance(seed_dict['train'], dict)
        assert len(seed_dict['train']['seed_initial_states']) == n_eval_iterations
        assert all(len(sublist) == n_iterations for sublist in seed_dict['train']['seed_initial_states'])
        assert len(seed_dict['train']['seed_shocks']) == n_eval_iterations
        assert all(len(sublist) == n_iterations for sublist in seed_dict['train']['seed_shocks'])

        assert isinstance(seed_dict['test'], dict)
        assert len(seed_dict['test']) == 2
        assert isinstance(seed_dict['test']['seed_initial_states'], int) and isinstance(seed_dict['test']['seed_shocks'], int)
        assert seed_dict['test']['seed_initial_states'] != seed_dict['test']['seed_shocks']

        # 2.2 lr
        lr_list = kwargs.get('lr_list', par.lr_list)
        assert isinstance(lr_list, list) and len(lr_list) == n_eval_iterations
        if lr_list != par.lr_list:
            print('\nChanges in learning rates will happen like the following')
            for lr, change in zip(*np.unique(lr_list, return_counts=True)):
                print(f'Evaluation iteration {change}:\t {lr}')
        else:
            print('\nlr_list was not specified, why learning rates in par.lr_list is used for all evaluation iterations')

        # 2.3 nn
        nn = kwargs.get('old_nn', None)

        if nn == None:
            neurons = kwargs.get('neurons', par.neurons)
            self.int_nn(seed_dict['int_weights'], device, neurons=neurons)
            nn = sol.trained_nns['int_nn']
            print('\nNew nn was succesfully initialized and is located in dict trained_nn in namespace sol')

        else:
            assert isinstance(nn, torch.nn.Module)
            print('\nold_nn was succesfully located')
        
        # 2.4 Extensions (Not done)
        dump_eval_iteration  = kwargs.get('dump_eval_iteration', [0]*n_eval_iterations)
        euler_errors_eval_iteration = kwargs.get('euler_errors_eval_iteration', [0]*n_eval_iterations)
        run_name = kwargs.get('run_name', 'unspecified_session' + datetime.now().strftime("%d-%m %H:%M:%S"))

        # 2.5 Inform user if many euler errors computations
        if sum(euler_errors_eval_iteration) > 3:
            print(f'{sum(euler_errors_eval_iteration)} Euler Errors evaluations are computed, this will take approximately {15*sum(euler_errors_eval_iteration):.2f} minutes')
            print('Interupt kernel if this is unintended')

        # 2.6 Initialize bools for computation after early stopping
        dump_and_done_bool = False
        compute_ee_and_done_bool = False

        # 3. Train for eval_iterations
        timestamp_outer = time.time()

        for eval_iteration in range(n_eval_iterations):

            # 3.1 Find just trained nn
            if eval_iteration > 0:
                nn = sol.trained_nns[f'trained_nn']

            print(f'\n{"-"*50}Starting evaluation iteration {eval_iteration}{"-"*50}')

            # 3.2 Acess lr
            lr = lr_list[eval_iteration]
            print(lr)

            # 3.3 Run iterations
            eval_iteration_tuple = self.train_nn_iterations(nn, lr, n_iterations, eval_iteration, seed_dict['train'], seed_dict['test'], device)

            # 3.4 Unpack results
            sol.in_sample_losses[eval_iteration] = eval_iteration_tuple[0]
            sol.out_of_sample_evals[eval_iteration] = eval_iteration_tuple[1]
            sol.comp_time[eval_iteration] = eval_iteration_tuple[2]

            # 3.5 Print progres
            if eval_iteration > 0:
                progress_between_evals = sol.out_of_sample_evals[eval_iteration] - sol.out_of_sample_evals[eval_iteration - 1]
                print(f'\nOut-of-sample progress:\t{progress_between_evals:.2f}')
            
            if eval_iteration > 9:
                progress_between_ten_evals = sol.out_of_sample_evals[eval_iteration] - sol.out_of_sample_evals[eval_iteration - 10]
                print(f'\nOut-of-sample progress over 10 evals:{progress_between_evals:.2f}')
        
            # 3.6 Check progress
            if eval_iteration >= par.early_stopping_check_eval_iteration and abs(progress_between_ten_evals) < par.early_stopping_threshold:
                print(f'\n Out-of-sample progres under threshold, stopping training')

                # 3.6.1 If dump was to be made, make it and break
                if sum(dump_eval_iteration) > 0:
                    dump_and_done_bool = True
                
                # 3.6.2 If euler errors should be computed, compute now and break
                if sum(euler_errors_eval_iteration) > 0:
                    compute_ee_and_done_bool = True
                
                # 3.6.3 If no operations, just break
                if dump_and_done_bool == False and compute_ee_and_done_bool == False:
                    break
                
            # 3.7 Dump results
            if dump_eval_iteration[eval_iteration] or dump_and_done_bool:

                # 3.7.1 Dump nn
                nn = nn = sol.trained_nns['trained_nn']

                nn_name = run_name + f'_nn_eval_iteration_{eval_iteration}'

                load_and_dump.dump_nn(nn, nn_name)

                # 3.7.2 Dump sol tensors
                list_of_names = ['_in_sample_losses', '_out_of_sample_evals', '_comp_time']

                for i, tensor in enumerate([sol.in_sample_losses, sol.out_of_sample_evals, sol.comp_time]):
                    
                    tensor_name = run_name + list_of_names[i] + f' nn_eval_iteration_{eval_iteration}'

                    load_and_dump.dump_arrays(tensor, tensor_name, tensor=True)

            # 3.8 Compute euler errors
            if euler_errors_eval_iteration[eval_iteration] or compute_ee_and_done_bool:

                nn = sol.trained_nns[f'trained_nn']

                self.simulate_test(seed_dict, device, nn=nn)

                self.compute_euler_errors(device, nn=nn)

                ee_tensor_name = run_name + f' Euler_Errors_{eval_iteration}'

                load_and_dump.dump_arrays(sim.euler_errors, ee_tensor_name, tensor=True)

            # 4. Stop if no progress
            if dump_and_done_bool or compute_ee_and_done_bool:
                print(f'Dumping or computing Euler Errors is done, breaking loop')
                break
            
            # 5 Print
            print(f'\nEvaluation iteration was succesfull')
            print(f'\nThere are now {len(sol.trained_nns)} nns in the specified DeepExplore namespace\n')

    def sim_dict(self, **kwargs):

        """
        Kwargs:
            
            int_weights:        Seed for weight initialization in nn

            n_epochs:           Number of epochs

            n_iterations:       Number of iterations

            test_seeds:         tuple of unique (seed_initial_states, seed_shocks) for testing
        
        Returns:
            dict:
                'int_weights':
                    seed for weights
                'train': dict
                    'seed_initial_states': list of len(n_epochs)
                        list of len(n_iterations)
                            seed for initial states
                    'seed_shocks': list of len(n_epochs)
                        list of len(n_iterations)
                            seed for shocks
                'test': dict
                    'seed_initial_states' : unique test seed for initial states 
                    'seed_shocks' : unique test seed for shocks
        """

        par = self.par
        # 1. Check kwargs
        assert all(isinstance(kwargs[kwarg], int) for kwarg in kwargs)

        # 2. Extract kwarg or set from par
        int_weights = kwargs.get('n_iterations', par.seed_int_weights)
        n_iterations = kwargs.get('n_iterations', par.n_iterations)
        n_eval_iterations = kwargs.get('n_eval_iterations', par.n_eval_iterations)
        test_seeds = kwargs.get('test_seeds', (1, 2))

        assert isinstance(test_seeds, tuple)
        assert test_seeds[0] != test_seeds[1]

        total = n_eval_iterations*n_iterations

        # 1. Make train dict
        train_initial_states_seeds = [list(range(3 + i*n_iterations, 3 + (1 + i)*n_iterations)) for i in range(n_eval_iterations)]

        train_shocks_seeds = [[total + i for i in iter_list] for iter_list in train_initial_states_seeds]

        train_dict = {'seed_initial_states' : train_initial_states_seeds, 'seed_shocks' : train_shocks_seeds}

        # 2. Make total dict
        dict = {'int_weights' : int_weights, 'train' : train_dict, 'test' : {'seed_initial_states' : test_seeds[0], 'seed_shocks' : test_seeds[1]}}

        return dict
    
    def generate_lr_list(self, lr_changes):

        """
        Function that creates list of learning rates for all epochs

        Args:

            lr_changes:             List of tuples with [(epoch_indx, new_lr), ..]'
        
        Returns:
            list_of_lr:             List of learning rates
        """

        par = self.par

        # 1. Static list of lr
        list_of_lr = par.lr_list

        # 2. Change baseline lr in indices
        for eval_iter_indx, new_lr in lr_changes:
            for i in range(len(list_of_lr)):
                if i >= eval_iter_indx:
                    list_of_lr[i] = new_lr
        
        return list_of_lr
    
    def generate_bool_list(self, indx_list):

        """
        Function that creates list of booleans of 0 = False, unless instructed to 1 = True

        Args:

            indx_list           List of indices that is to be converted to 1's, i.e. true booleans

        Returns:
            list_of_bools:      List of booleans
        """

        par = self.par
        
        # 1. Make list of False booleans
        list_of_bools = [0]*par.n_eval_iterations

        # 2. Change from False to True in indices
        for i in indx_list:
            list_of_bools[i] = 1

        return list_of_bools

    def simulate_test(self, seed_dict, device, nn=None):
    
        par = self.par
        sol = self.sol
        sim = self.sim

        # 1. Locate nn if not specified
        if nn == None:
            key_of_last_nn = list(sol.trained_nns)[-1]

            nn = sol.trained_nns[key_of_last_nn]
        
        # 1.1 Check specified
        else:
            isinstance(nn, torch.nn.Module)

        # 2. Set nn in eval mode
        nn.eval()

        # 3. Locate test dict
        test_dict = seed_dict['test']

        # 2. Simulate
        with torch.no_grad():
            loss, state, action = simulate.simulate_model(par, test_dict, nn, eval=True, device=device, tensor_bool=True)

        # 3. Store in namespace
        sim.state = state
        sim.action = action
        sim.obj = -loss
    
    def compute_euler_errors(self, device, n_batches=10, nn=None):

        par = self.par
        sol = self.sol
        sim = self.sim

        sim.euler_errors = torch.zeros((par.T - 1, par.N_eval))

        # Convert Gauss-Hermite quadrature to arrays
        xR_array = quadrature.log_normal_gauss_hermite(n=10, sigma=par.sigmaR)
        xp_array = quadrature.log_normal_gauss_hermite(n=10, sigma=par.sigmap)

        xR = torch.tensor(xR_array[0], dtype=par.dtype, device=device)
        wR = torch.tensor(xR_array[1], dtype=par.dtype, device=device)

        xp = torch.tensor(xp_array[0], dtype=par.dtype, device=device)
        wp = torch.tensor(xp_array[1], dtype=par.dtype, device=device)

        GH_tuple = (xR, wR, xp, wp)

        # 1. Locate nn if not specified
        if nn == None:
            key_of_last_nn = list(sol.trained_nns)[-1]

            nn = sol.trained_nns[key_of_last_nn]
        
        # 1.1 Check specified
        else:
            isinstance(nn, torch.nn.Module)

        if torch.all(sim.state == 0) or torch.all(sim.action == 0):
            print('No simulated data found, make simulation and then compute Euler Errors')
            return # Stop function

        # 2. Make batches
        batch_size = int(par.N_eval/n_batches)
        list_batch = [batch_size]*n_batches

        # 3. Compute euler errors
        for t in range(par.T-1):

            print(f'Computing Euler Errors for period {t}')

            state_batches = torch.split(sim.state[t], list_batch, dim=0)
            action_batches = torch.split(sim.action[t], list_batch, dim=0)

            for batch in range(n_batches):
                print(f'Computing Euler Errors in period {t} for batch {batch}')

                euler_errors.euler_errors_nn_linalg(par, sim.euler_errors, nn, batch, batch_size, state_batches[batch].to(device), action_batches[batch].to(device), t, device, GH_tuple)