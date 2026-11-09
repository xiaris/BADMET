import numpy as np
from autograd import grad
from multiprocessing import Pool
import emcee

from .baddat_probability_map import log_tau0_to_log_tau1, log_tau0_to_log_tau1_derivative

class MassEstimator:
    def __init__(self):
        self.param_list = []
        self.Mparam_list = []
        self.tau_list = []
        self.Mest_list = []

    def add_Mest(self, LOG_MBH, LOG_MBH_ERR):
        self.Mest_list.append({
            'LOG_MBH': LOG_MBH,
            'LOG_MBH_ERR': LOG_MBH_ERR
        })
    
    def add_Mparam(self, func, func_err, args=None, args_err=None):
        if args is None: args = []
        if args_err is None: args_err = []
        self.Mparam_list.append({
            'func': func,
            'func_err': func_err,
            'args': args,
            'args_err': args_err
        })
        
    def add_param(self, param, param_err, func, func_err, args=None, args_err=None):
        if args is None: args = []
        if args_err is None: args_err = []
        self.param_list.append({
            'param': param,
            'param_err': param_err,
            'func': func,
            'func_err': func_err,
            'args': args,
            'args_err': args_err
        })

    
    def add_tau(self, log_tau_mle, baseline, ncadence, func, func_err, args=None, args_err=None):
        if args is None: args = []
        if args_err is None: args_err = []
        self.tau_list.append({
            'log_tau_mle': log_tau_mle,
            'baseline': baseline,
            'ncadence': ncadence,
            'func': func,
            'func_err': func_err,
            'args': args,
            'args_err': args_err
        })

    def log_likelihood(self, theta):
        log_M_BH = theta
        ll = 0.0
        # MBH likelihood
        for p in self.Mest_list:
            sigma2 = p['LOG_MBH_ERR']**2
            ll += -0.5 * ((p['LOG_MBH'] - log_M_BH)**2 / sigma2 + np.log(sigma2))
            
        # Mparam likelihood
        for p in self.Mparam_list:
            sigma2 = p['func_err']**2
            for i, err in enumerate(p['args_err']):
                df = grad(p['func'], i)(*p['args'])
                sigma2 += (df * err)**2
            ll += -0.5 * ((p['func'](*p['args']) - log_M_BH)**2 / sigma2 + np.log(sigma2))
            
        # param likelihood
        for p in self.param_list:
            sigma2 = p['param_err']**2 + p['func_err']**2
            for i, err in enumerate(p['args_err']):
                df = grad(p['func'], i+1)(log_M_BH, *p['args'])
                sigma2 += (df * err)**2
            ll += -0.5 * ((p['param'] - p['func'](log_M_BH, *p['args']))**2 / sigma2 + np.log(sigma2))


        # tau likelihood
        for t in self.tau_list:
            tau0 = t['func'](log_M_BH, *t['args'])
            log_tau1_mean, log_tau1_std = log_tau0_to_log_tau1(tau0, t['baseline'], t['ncadence'])
            sigma2 = log_tau1_std**2 + t['func_err']**2
            for i, err in enumerate(t['args_err']):
                deriv = log_tau0_to_log_tau1_derivative(tau0, t['baseline'], t['ncadence'])
                df = grad(t['func'], i+1)(log_M_BH, *t['args'])
                sigma2 += (deriv * df * err)**2
            ll += -0.5 * ((t['log_tau_mle'] - log_tau1_mean)**2 / sigma2 + np.log(sigma2))
        return ll
        
    def log_prior(self, theta):
        log_M_BH = theta
        if 0 < log_M_BH < 12:
            return 0.0
        return -np.inf

    
    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def estimate(self, initial=None, nwalkers=16, nsteps=1000, pool_size=1, progress=True):
        if initial is None:
            initial = np.array([5.0])
        pos = initial + 1e-4 * np.random.randn(nwalkers, 1)

        if pool_size == 1:
            sampler = emcee.EnsembleSampler(nwalkers, 1, self.log_probability)
            sampler.run_mcmc(pos, nsteps, progress=progress)
        else:
            with Pool(pool_size) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, 1, self.log_probability, pool=pool)
                sampler.run_mcmc(pos, nsteps, progress=progress)

        flat_samples = sampler.get_chain(
            discard=max(1, nsteps//5),
            thin=max(1, nsteps//100),
            flat=True
        )
        return flat_samples
