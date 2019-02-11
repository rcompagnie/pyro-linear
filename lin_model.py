
import os
import numpy as np
import torch
import torch.nn as nn

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# for CI testing
smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)

import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab



# Model parameters
subsample_size = 10000
lr = 0.05
num_iterations = 1000 if not smoke_test else 2
test_size = 1000
num_samples_predict = 50



# Import data
rev_df = pd.read_csv("../Data/rev_ext_small.csv", index_col=0, dtype=np.float32)
N = rev_df.shape[0]
p = rev_df.shape[1] - 1
rev_ts = torch.tensor(rev_df.values).type(torch.Tensor)
var_y_data = 1.69666648605591

# Correct subsample size
if subsample_size > N:
    subsample_size = N



class RegressionModel(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
    
    def forward(self, x):
        return self.linear(x)



regression_model = RegressionModel(p)



# Model
def model(data):
    
    # Create normal priors over the parameters
    loc, scale = torch.zeros(1, p), 10 * torch.ones(1, p)
    bias_loc, bias_scale = 3 * torch.ones(1), 10 * torch.ones(1)
    w_prior = Normal(loc, scale).independent(1)
    b_prior = Normal(bias_loc, bias_scale).independent(1)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module_pyro", regression_model, priors)
    
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    
    with pyro.iarange("map", N, subsample=data):
        
        x_data = data[:, :-1]
        y_data = data[:, -1]
        
        # run the regressor forward conditioned on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        
        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, .5 * torch.ones(data.size(0))),
                    obs=y_data)



softplus = torch.nn.Softplus()

# Guide
def guide(data):
    
    # define our variational parameters
    w_loc = torch.randn(1, p)
    # note that we initialize our scales to be pretty narrow
    w_log_sig = torch.tensor(-3.0 * torch.ones(1, p) + 0.05 * torch.randn(1, p))
    b_loc = torch.randn(1)
    b_log_sig = torch.tensor(-3.0 * torch.ones(1) + 0.05 * torch.randn(1))
    
    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_loc)
    sw_param = softplus(pyro.param("guide_log_scale_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_loc)
    sb_param = softplus(pyro.param("guide_log_scale_bias", b_log_sig))

    # guide distributions for w and b
    w_dist = Normal(mw_param, sw_param).independent(1)
    b_dist = Normal(mb_param, sb_param).independent(1)
    dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
    
    # overload the parameters in the module with random samples
    # from the guide distributions
    lifted_module = pyro.random_module("module_pyro", regression_model, dists)
    
    return lifted_module()



data = rev_ts
optim = Adam({"lr": lr})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

def inference():

    pyro.clear_param_store()

    for j in range(num_iterations):
        
        # Subsample the data
        ind = random.sample(range(N), subsample_size)
        sub_data = data[ind,:]

        # Perform inference step
        loss = svi.step(sub_data)
        
        # Print evolution
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / float(N)))
    print()



def predict(x):
    sampled_models = [guide(None) for _ in range(num_samples_predict)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean



if __name__ == '__main__':
    
    # Print model parameters
    print("Size of subsamples used to train the model: ", subsample_size)
    print("Learning rate: ", lr)
    print("Number of iterations for inference: ", num_iterations)
    print("Number of samples used for prediction: ", num_samples_predict)
    print("Numner of samples for the test: ", test_size)
    print()
    
    # Perform inference
    inference()
    
    print('Pyro parameters: ', list(pyro.get_param_store().get_all_param_names()))
    print('Inputs: ', list(rev_df.columns.values)[:-1])
    print('Output: ', list(rev_df.columns.values)[-1])
    for name in pyro.get_param_store().get_all_param_names():
        print("[%s]: %s" % (name, pyro.param(name).data.numpy()))
    print()

    # Test prediction accuracy
    MSE = 0
    ind = random.sample(range(N), test_size)
    x_data = data[ind, :-1]
    y_data = data[ind, -1]
    y_pred = predict(x_data).squeeze()
    MSE += ((y_data - y_pred)**2).mean().item()
      
    print("MSE: ", MSE)
    print("Var: ", var_y_data)
    print("MSE/Var: %d %%" % (100 * MSE / var_y_data))
    print()

    # Plot a prediction
    i = random.sample(range(N), 1)
    x_data = data[i, :-1]
    y_data = data[i, -1]
    sampled_models = [guide(None) for _ in range(200)]
    y_sample = [model(x_data).data + np.random.normal(0, .5) for model in sampled_models]
    plt.hist(y_sample, range=(0., 5.), bins=20)
    plt.axvline(x=y_data, color='#d62728')
    plt.title("P(y)")
    plt.xlabel("y")
    plt.ylabel("#")
    plt.show()

    # Plot bias
    mu = pyro.param("guide_mean_bias").data.numpy()
    sigma = softplus(pyro.param("guide_log_scale_bias").data).numpy()
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x,mlab.normpdf(x, mu, sigma))
    plt.title("P(beta)")
    plt.xlabel("beta")
    plt.ylabel("#")
    plt.show()
