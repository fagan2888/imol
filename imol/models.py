"""network models

observation about motor delay 2016-06-24: if the delay is greater than one time-step
and the system has memory, the causal connection and thus "learnability"
between the motor and sensor signal will be weakened by redundancy ()

"""

from __future__ import print_function

import argparse
import numpy as np
import pylab as pl
import scipy.sparse as spa
import matplotlib.gridspec as gridspec
import rlspy

import cPickle, time, sys
# import tensorflow as tf

from smp_base.models_reservoirs import res_input_matrix_random_sparse, res_input_matrix_disjunct_proj
from smp_base.models_reservoirs import create_matrix_reservoir, normalize_spectral_radius

def relu(x):
    return np.clip(x, 0, np.inf)
    

class LinearNetwork(object):
    def __init__(self, modelsize = 100, idim = 1, odim = 1, alpha = 1.0, eta = 1e-3, theta_state = 1e-2, input_scaling = 1.0,
                 g = 0.0, tau = 1.0, mtau=False, bias_scaling = 0.0):
        self.modelsize = modelsize
        self.idim = idim
        self.odim = odim
        self.input_scaling = input_scaling
        self.output_scaling = 1e-1

        # FIXME: params: nonlin, 
                            
        # input to hidden
        # self.W_i = np.random.normal(0, self.input_scaling, (self.modelsize, self.idim))
        self.W_i = res_input_matrix_random_sparse(idim = self.idim, odim = self.modelsize, sparsity = 0.2) * self.input_scaling
        # self.W_i = res_input_matrix_disjunct_proj(idim = self.idim, odim = self.modelsize) * self.input_scaling
        # self.W_i = np.eye(self.modelsize) * 0.5
        self.b_i = np.random.normal(0, 1e-3, (self.modelsize, 1))

        # hidden to output
        # self.W_o = np.random.normal(0, 1e-2, (self.odim, self.modelsize))
        self.W_o = np.zeros((self.odim, self.modelsize))
        self.b_o = np.random.normal(0, 1e-3, (self.odim, 1))

        # state
        self.x = np.zeros((self.idim, 1))
        self.r = np.zeros((self.modelsize, 1))
        self.h = np.zeros((self.modelsize, 1))
        self.y = np.zeros((self.odim, 1))

        # bias
        self.bias = np.random.uniform(-1., 1., size=self.h.shape)
        self.bias_scale = bias_scaling
        
        # learning
        self.eta = 1e-1 # for 1D io, modelsize 100

        self.eta = 5e-2 # for 1D io, modelsize 1000
        self.eta = 1e-2 # for 1D io, modelsize 1000
        self.eta = 1e-3 # for 1D io, modelsize 1000
        # self.eta = 1e-2 # for 1D io, modelsize 1000
        # self.eta = 2.5e-3 # for 1D io, modelsize 1000
        # self.eta = 1e-3 # for 3D io, modelsize 1000
        # self.eta = 1e-4 # for 3D io, modelsize 1000
        self.theta_state = theta_state

        # RLS        
        self.rlsi = [rlspy.data_matrix.Estimator(np.random.uniform(0, 0.01, size=(self.modelsize, 1)) , np.eye(self.modelsize)) for i in range(self.odim)]
        # FORCE
        self.alpha = alpha
        self.P = (1.0/self.alpha)*np.eye(self.modelsize)
        
    def predict(self, x):
        self.x = x.copy()
        self.r = np.dot(self.W_i, self.x) + (self.bias * self.bias_scale) # + self.M.dot(self.h) # + self.b_i
        self.h = relu(self.r)
        self.h += (np.random.normal(0, 1, self.h.shape)) * self.theta_state
        self.y = np.dot(self.W_o, self.h) # + self.b_o
        return self.y, self.h

    def fit(self, x, y):
        # print(x.shape, y.shape)
        error = y - self.y
        # delta rule
        # eta * error * activation * input
        # dw = self.eta * error * self.y * self.h.T
        # dw_o_1 = error * self.h.T
        dw_o = np.dot(error, self.h.T)
        # print("dw_o.shape", dw_o.shape)
        self.W_o += self.eta * dw_o

        # backprop error
        dh = np.dot(self.W_o.T, error)
        dw_i = dh * self.x.T #
        dw_i = np.dot(dh, self.x.T)
        # print("dw_i.shape", dw_i.shape)
        self.W_i += self.eta * dw_i
        
        # print("|W_i|", np.linalg.norm(self.W_i))
        # print("|W_o|", np.linalg.norm(self.W_o))
        # print("error", error)
        # y = x.copy()
        return error

    def fitRLS(self, x, y):
        """Recursive Least Squares learning rule"""
        error = y - self.y
        for i in range(self.odim):
            self.rlsi[i].update(self.h.T, y[[i],0], 1e-2)
            self.W_o[[i],:] = self.rlsi[i].x.T
        return error

    def fitFORCE(self, x, y, reverse = False, activate = False):
        """FORCE learning rule"""
        # print("x", x, "y", y)
        if activate:
            r_ = self.r.copy()
            h_ = self.h.copy()
            self.predict(x)
                    
        k = np.dot(self.P, self.h)
        rPr = np.dot(self.h.T, k)
        c = 1.0/(1.0 + rPr)
        # print("x.shape", x.shape, "y.shape", y.shape, "self.y.shape", self.y.shape, "self.h.shape", self.h.shape, "k.shape", k.shape)
        # print "r.shape", self.r.shape
        # print "k.shape", k.shape, "P.shape", self.P.shape, "rPr.shape", rPr.shape, "c.shape", c.shape
        self.P = self.P - np.dot(k, (k.T*c))

        # TODO: vectorize loop
        for i in range(self.odim):
            # print "self.P", self.P
            # print "target.shape", target.shape
            e = y[i,0] - self.y[i,0]
            # e = self.y[i,0] - y[i,0]
            # print("y[i,0], self.y[i,0], e", i, y[i,0], self.y[i,0], e)
            # print "error e =", e, self.z[i,0]

            # print "err", e, "k", k, "c", c
            # print "e.shape", e.shape, "k.shape", k.shape, "c.shape", c.shape
            # dw = np.zeros_like(self.wo)
            if reverse:
                dw = -e * k * c
            else:
                dw = e * k * c
            # dw = -e * np.dot(k, c)
            # print("dw", dw.shape, self.W_o.shape)
            # print "shapes", self.wo.shape, dw.shape # .reshape((self.N, 1))
            # print i
            # print "shapes", self.wo[:,0].shape, dw[:,0].shape # .reshape((self.N, 1))
            # print "types", type(self.wo), type(dw)
            self.W_o[[i],:] += dw[:,0]
            # self.wo[:,i] = self.wo[:,i] + dw[:,0]
            # self.wo[:,i] += dw[:]
        # print "FORCE", LA.norm(self.wo, 2)
        # return 0

        if activate:
            self.r = r_
            self.h = h_
        
        return y - self.y
    
class ReservoirNetwork(LinearNetwork):
    def __init__(self, modelsize = 100, idim = 1, odim = 1, alpha = 1.0, eta = 1e-3, theta_state = 1e-2, input_scaling = 1.0,
                 g = 0.0, tau = 1.0, mtau=False, bias_scaling = 0.0):
        LinearNetwork.__init__(self, modelsize = modelsize, idim = idim, odim = odim, alpha = alpha, eta = eta,
                               theta_state = theta_state, input_scaling = input_scaling, bias_scaling = bias_scaling)
        
        self.g = g # spectral radius
        self.p = 0.1 # recurrent matrix density
        # leak rate
        self.mtau = mtau
        if self.mtau:
            self.tau = np.exp(np.random.uniform(-10, 0, (self.N, 1)))
        else:
            self.tau = tau

        # recurrent (reservoir) connection matrix
        self.M = create_matrix_reservoir(self.modelsize, self.p)

        normalize_spectral_radius(self.M, self.g)
        self.M = spa.csr_matrix(self.M)

    def predict(self, x):
        self.x = x.copy()
        r1_tp1 = (1 - self.tau) * self.r
        r2_tp1 = self.tau * (np.dot(self.W_i, self.x) + self.M.dot(self.h) + (self.bias * self.bias_scale))
        self.r = r1_tp1 + r2_tp1 # + self.b_i
        # self.h = relu(self.r)
        self.h = np.tanh(self.r)
        self.h += (np.random.normal(0, 1, self.h.shape)) * self.theta_state
        self.y = np.dot(self.W_o, self.h) # + self.b_o
        return self.y, self.h
        
# class: ReLU network

def gen_dataset_simple_1d(args):
    s = np.zeros((10000, 1))
    u = np.zeros((10000, 1))
    for i in range(s.shape[0]-1):
        u[i,0] = np.random.uniform(-1, 1)
        s[i+1,0] = s[i,0] + 0.5 * u[i,0]
    # pl.subplot(211)
    # pl.plot(s)
    # pl.subplot(212)
    # pl.plot(u)
    # pl.show()
    dataset = {
        "s": s.copy(),
        "m": u.copy()
        }
    return dataset
    
def gen_dataset_simple_nd(args):
    s = np.zeros((10000, args.dim))
    u = np.zeros((10000, args.dim))
    for i in range(s.shape[0]-1):
        u[[i],:] = np.random.uniform(-1, 1, (1, args.dim))
        # print("blub", u[[i],:])
        # print("blub", 0.5 * u[[i],:])
        s[[i+1],:] = s[[i],:] + 0.5 * u[[i],:]
    # pl.subplot(211)
    # pl.plot(s)
    # pl.subplot(212)
    # pl.plot(u)
    # pl.show()
    dataset = {
        "s": s.copy(),
        "m": u.copy()
        }
    return dataset
    

def gen_dataset_pm3d(args):
    from robots.pointmass import SysPM3D

    generator = SysPM3D(args.lag)
    for i in range(generator.numsteps - 1):
        # generator.step(i, )
        u = np.random.uniform(-10, 10, (3, 1))
        # phase = np.ones((3, 1)) * (i / 10.0)
        # u = np.sin(phase)
        # u += np.random.normal(0, 1e-2, u.shape)
        generator.step(i, u)

    pl.subplot(411)
    pl.plot(generator.sys.x)
    pl.subplot(412)
    pl.plot(generator.sys.v)
    pl.subplot(413)
    pl.plot(generator.sys.a)
    pl.subplot(414)
    pl.plot(generator.sys.u)
    pl.show()

    # dataset = {"s": np.roll(generator.sys.v, -1, axis = 0), # need to roll because of double integration
    #            "m": generator.sys.u.copy()}
    dataset = {"s": generator.sys.v.copy(),
               "m": generator.sys.u.copy()}
    return dataset

def standardize_data(ds):
    mean_s = np.mean(ds["s"], axis=0)
    std_s  = np.std(ds["s"], axis=0)
    mean_m = np.mean(ds["m"], axis=0)
    std_m  = np.std(ds["m"], axis=0)
    print("stat shapes", mean_s, std_s, mean_m, std_m)
    ds["s"] -= mean_s
    ds["s"] /= std_s
    ds["m"] -= mean_m
    ds["m"] /= std_m
    return ds, (mean_s, std_s, mean_m, std_m)

def destandardize_data(ds, stats):
    ds_ = ds.copy()
    ds_ *= stats[1] # times std
    ds_ += stats[0] # plus mean
    return ds_

def prepare_prediction_data(ds, args):
    lag = args.lag
    dslen = min(ds["s"].shape[0], ds["m"].shape[0])
    s_t_offset   = 0
    s_t_end      = dslen - 2 - lag # -(lag + 1)
    s_tp1_offset = lag + 1
    s_tp1_end    = dslen - 1 # lag
    u_t_offset   = 0
    u_t_end      = dslen - 2 - lag # -(lag + 1)
    print("shift operators", s_t_offset, s_t_end, s_tp1_offset, s_tp1_end, u_t_offset, u_t_end)
    # s_t   = ds["s"][0:-1]
    # s_tp1 = ds["s"][1:]
    # u_t   = ds["m"][0:-1]
    # s_t   = ds["s"][1:-2]
    # s_tp1 = ds["s"][2:-1]
    # u_t   = ds["m"][:-3]
    s_t   = ds["s"][s_t_offset:s_t_end]
    s_tp1 = ds["s"][s_tp1_offset:s_tp1_end]
    u_t   = ds["m"][u_t_offset:u_t_end]
    print("s_t.shape, s_tp1.shape, u_t.shape", s_t.shape, s_tp1.shape, u_t.shape)

    X_inv = np.hstack((s_t, s_tp1))
    X_fwd = np.hstack((s_t, u_t))

    Y_inv = u_t.copy()
    Y_fwd = s_tp1.copy()
    return (X_fwd, Y_fwd, X_inv, Y_inv)


def main(args):
    # print("bla")

    # test dataset
    if args.mode == "gen_dataset_pm3d":
        """generate dataset from 3D pointmass"""
        ds = gen_dataset_pm3d(args)
        cPickle.dump(ds, open("%s/dataset_pm3d_%s.bin" % (sys.argv[0][:-3], time.strftime("%Y%m%d_%H%M%S")), "wb"))

    # modes
    # SOM
    # GNG
    # goal babbling rolf / explauto
    # IM / explauto
        
    elif args.mode == "linear":
        # load data
        # ds_raw = cPickle.load(open(args.datafile))
        ds_raw = np.load(args.datafile)
        # ds_raw = gen_dataset_pm3d(args)
        # ds_raw = gen_dataset_simple_1d(args)
        # ds_raw = gen_dataset_simple_nd(args)
        # standardize data
        ds_raw, stats = standardize_data(ds_raw)
        print("stats", stats)
                
        # prepare predidiction dataset
        X_fwd, Y_fwd, X_inv, Y_inv = prepare_prediction_data(ds_raw, args)

        print("X_inv.shape", X_inv.shape)

        pl.subplot(211)
        pl.title("X_inv")
        pl.plot(X_inv)
        pl.subplot(212)
        pl.title("Y_inv")
        pl.plot(Y_inv)
        pl.show()
        
        # create network
        modelsize = 100 # X_inv.shape[1]
        ln = LinearNetwork(modelsize = modelsize, idim = X_inv.shape[1], odim = Y_inv.shape[1])
        numiter = 1
        numsteps = X_inv.shape[0] # ds_raw["s"].shape[0] - 1
        err_ = np.zeros((numsteps*numiter, 1))
        W_o_ = np.zeros((numsteps*numiter, 1))
        h_ = np.zeros((numsteps*numiter, modelsize))
        y_ = np.zeros((numsteps*numiter, Y_inv.shape[1]))
        # train network
        for j in range(numiter):
            for i in range(numsteps):
                fullidx = (j*numsteps)+i
                y, h = ln.predict(X_inv[[i],:].T)
                h_[fullidx,:] = h.T
                y_[fullidx,:] = y.T
                # print("error", np.linalg.norm(Y_inv[[i]].T - y, 2))
                # print("error 1", Y_inv[[i]].T - y)
                # print("y", y.shape)
                # fiterr = ln.fit(X_inv[[i]].T, Y_inv[[i]].T)
                # fiterr = ln.fitRLS(X_inv[[i]].T, Y_inv[[i]].T)
                fiterr = ln.fitFORCE(X_inv[[i]].T, Y_inv[[i]].T)
                err_[fullidx] = np.linalg.norm(fiterr, 2)
                W_o_[fullidx] = np.linalg.norm(ln.W_o, 2)
                if fullidx % 1000 == 0:
                    print("step %d: err = %f, |W_i| = %f, |W_o| = %f" % (fullidx,
                                                                         np.mean(err_[fullidx-1000:fullidx]),
                                                                         np.linalg.norm(ln.W_i, 2),
                                                                         W_o_[fullidx],
                                                                         ))

        print("h_.shape", h_.shape)
        # batch regression comparison
        from sklearn import linear_model
        from sklearn import kernel_ridge
        # lm = linear_model.LinearRegression()
        lm = linear_model.Ridge(alpha = 1.0)
        # lm.fit(X_inv, Y_inv)
        # y_batch = lm.predict(X_inv)
        lm.fit(h_, np.vstack([Y_inv] * numiter))
        y_batch = lm.predict(h_)

        # destandardize data (rescale back to original values)
        # print("stats", stats)
        y__ = destandardize_data(y_, (stats[2], stats[3]))

        gs = gridspec.GridSpec(Y_inv.shape[1] + 2, 1)
        for i in range(Y_inv.shape[1]):
            pl.subplot(gs[i])
            pl.title("Y_inv[%d] (target)" % i)
            pl.plot(np.vstack([Y_inv[:,[i]]] * numiter), linewidth=0.5)
            pl.plot(y_[:,[i]])
            # pl.plot(y__[:,[i]])
            pl.plot(y_batch[:,[i]])
        pl.subplot(gs[i+1])
        pl.title("err")
        pl.plot(err_)
        pl.subplot(gs[i+2])
        pl.title("|W_o|")
        pl.plot(W_o_)
        pl.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dim",  dest="dim", default=1, type=int, help="input data dimension [1]")
    parser.add_argument("-l", "--lag",  dest="lag",  default=0, type=int, help="Additional motor delay on top of the minimum 1 [0]")
    parser.add_argument("-m", "--mode", dest="mode", default="linear", type=str, help="script execution mode [linear]")
    parser.add_argument("-df", "--datafile", dest="datafile", default=None, type=str, help="datafile to load for testing models [None]")
    args = parser.parse_args()
    main(args)
