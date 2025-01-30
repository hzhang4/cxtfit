import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import scipy.stats as stats

from .stocde import StoCDE

class CXTmodel(StoCDE):
    """
    Calculate deterministic equilibrium & nonequilibrium CDE
    """
    # Converted from DIRECT and MODEL function in MODEL.FOR original Fortran code
    def model(self,simparms):
        """Assign coefficients"""
        if self.mode in (1,2):
            self.init_detcde(simparms)
        else:
            self.init_stocde(simparms)
        
        # Loop over observation data points
        csim1=[0.0]*self.nob
        csim2=[0.0]*self.nob
        cvar1=[0.0]*self.nob
        cvar2=[0.0]*self.nob
        for i,rec in self.cxtdata.iterrows():
            if self.nredu in (0,1,3): # DIMENSIONLESS Distance
                self.zz = rec['z'] / self.zl
            else :
                self.zz = rec['z']

            if self.nredu in (0,1): # DIMENSIONLESS Time
                self.tt = rec['t'] * self.v / self.zl
            else :
                self.tt = rec['t']

            if self.mode ==1:
                c1,c2 = self.DetCDE()
                if c1 > self.ctol: csim1[i]=c1 
            elif self.mode==2:
                c1,c2 = self.DetCDE()
                if self.inverse == 1 and (self.modc == 4 or self.modc == 6):
                    c1 = self.beta * self.r * c1
                    c2 = (1 - self.beta) * self.r * c2
                if c1 > self.ctol: csim1[i]=c1 
                if c2 > self.ctol: csim2[i]=c2 
            else:
                c1,c2,v1,v2 = self.StoCDE()
                if c1 > self.ctol: csim1[i]=c1 
                if v1 > self.ctol: cvar1[i]=c1 
                if self.mode in (4,6,8):
                    if c2 > self.ctol: csim2[i]=c2 
                    if v2 > self.ctol: cvar2[i]=c2 
        return csim1,csim2,cvar1,cvar2

    def direct(self,simparms, verbose=False):
        csim1,csim2,cvar1,cvar2 = self.model(simparms)
        self.cxtdata['csim'] = csim1
        if self.mode in (2,4,6,8):
            self.cxtdata['csim2'] = csim2
        if self.mode > 2:
            self.cxtdata['cvar'] = cvar1
        if self.mode in (4,6,8):
            self.cxtdata['cvar2'] = cvar2
        
        if verbose:
            print(self.cxtdata)

    # Define the residuals function to be used in the scipy.optimize.leastsq method
    def residuals(self,params):
        current_params = self.binit.copy() 
        # Update parameters based on the fit flags 
        idx = 0
        for key, flag in self.bfit.items():
            if flag == 1:
                current_params[key] = params[idx]
                idx += 1

        csim,_,_,_ = self.model(current_params)

        return csim - self.cobs
    
    # Replaced the curve fitting algorithm in the original fortran code with 
    # the scipy.optimize.leastsq function to fit model to data
    def curvefit(self,verbose=False):
        if self.nfit > self.nob:
            raise ValueError("Number of parameters to fit exceeds number of observations")
        
        self.cobs = self.cxtdata['cobs'].values
        self.binit = self.parms.loc['binit'].to_dict()
        self.bfit = self.parms.loc['bfit'].to_dict()
        initial_guess = [self.binit[key] for key, flag in self.bfit.items() if flag == 1]

        fitkeys = [key for key, flag in self.bfit.items() if flag == 1]

        if self.ilmt : # Use bounds
            bmax = self.parms.loc['bmax'].to_dict()
            bmin = self.parms.loc['bmin'].to_dict()
            lower_bounds = [bmin[key] for key, flag in self.bfit.items() if flag == 1]
            upper_bounds = [bmax[key] for key, flag in self.bfit.items() if flag == 1]
            # Perform the least squares optimization, default setting works great
            leastsq = least_squares(self.residuals, initial_guess, 
                                        bounds=(lower_bounds, upper_bounds), verbose=verbose)
        else:
            leastsq = least_squares(self.residuals, initial_guess, verbose=verbose)

        params_optimized = leastsq.x
        J = leastsq.jac  # Jacobian matrix
        residuals = leastsq.fun
        m = len(residuals)
        p = len(params_optimized)

        # Covariance matrix (using the formula: Cov = (J.T @ J)^{-1} * residuals variance)
        # First, calculate the residuals variance
        residual_variance = np.sum(residuals**2) / (m - p)  # m - p is the degrees of freedom

        # Calculate the covariance matrix
        covariance_matrix = residual_variance * np.linalg.inv(J.T @ J)

        # Standard errors
        std_errors = np.sqrt(np.diagonal(covariance_matrix))

        # t-values
        t_values = params_optimized / std_errors

        # Confidence intervals (95% confidence)
        alpha = 0.05  # 95% confidence
        t_critical = stats.t.ppf(1 - alpha / 2, m - p)  # t-distribution critical value
        CI_Lower = params_optimized - t_critical * std_errors
        CI_Upper = params_optimized + t_critical * std_errors

        self.stats_dict = pd.DataFrame({
            'Final_Parameters': dict(zip(fitkeys,params_optimized)),
            'standard_errors': dict(zip(fitkeys, std_errors)),
            't_values': dict(zip(fitkeys, t_values)),
            'CI_Lower': dict(zip(fitkeys, CI_Lower)),
            'CI_Upper': dict(zip(fitkeys, CI_Upper)),
        })

        if verbose:
            print(f"{self.stats_dict}")

        # Update the simulated data with the optimized parameters
        self.bfinal = self.binit.copy()
        idx = 0
        for key, flag in self.bfit.items():
            if flag == 1:
                self.bfinal[key] = leastsq.x[idx]
                idx += 1
        
        self.direct(self.bfinal, verbose=verbose)

        return
