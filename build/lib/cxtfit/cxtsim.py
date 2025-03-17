import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import scipy.stats as stats

from .stocde import StoCDE, limit

class CXTsim(StoCDE):
    """ Simulation Case Class of CXTFIT

    Parameters
    ----------
    inverse : int (0,1), default = 0
        flag of inverse problem (0 = forward problem, 1 = inverse problem)
    mode : int (1,2,3,4,5,6,8), default = 1  
     flag of model mode ( 1 = equilibrium  model, 2 = nonequilibrium  model, 
     stochastic models: 3 = kd&v equilibrium      4= kd&v nonequilibrium
         5 = d&v equilibrium      6 = d&v nonequilibrium 8 = alpha & v nonequilibrium 
    nredu : int (0,1,2,3), default = 1
        flag of reduction of variables (0 = real t ,x ,<c2>  for mode=4,6,8,
        1 = real t & x  c2=<c2/k>, 2 = dimensionless t & z, 3 = dimensionless t & real x)
    modc : int (1,2,3,4,5,6), default = 1 
        flag of model concentration (1 = flux concentration or area-averaged flux concentration,
        2 = field-scale flux concentration, 3 = third-type resident concentration,
        4 = third-type total resident concentration, 5 = first-type resident concentration,
        6 = first-type total resident concentration)   
        MODC= 1,2,3,5  PHASE 1 CONC. IS USED FOR PARAMETER ESTIMATION.
    zl : float, default = 1.0
        Characteristic length for dimensionless parameters
    mit: int, default = 0
        Maximum number of iterations (the inverse part is bypassed if MIT = 0, the
        program calculates concentrations at specified Z(I) and T(I) using the initial
        estimates as model parameters).
    ilmt: int, default = 0
        Parameter constraint code(0 = no constraints for parameter estimation, 
        1 = use minimum and maximum constraints)
    mass: int, default = 0
        Total mass estimation code. This option is only available for the BVP in case of
        a Dirac, step-type, or single pulse input. 0 = No estimation for total mass.
        1 = Total mass included in estimation procedure.
    massst: int, default = 0
        Mass distribution index for the stochastic CDE (
        0 = Amount of solute in each tube is proportional to v)
        1 = Amount of solute in each tube is constant regardless of v)
    mneq: int, default = 1
        Nonequilibrium model code MNEQ=l for the stochastic one-site model 
        0 = Two-region physical nonequilibrium model; 
        1 = One-site chemical nonequilibrium model; 
        2 = Two-site chemical nonequilibrium model
        3 = Two-region physical nonequilibrium model with internal constraints
    mdeg: int, default = 0
        Degradation estimation code for the nonequilibrium CDE
        0 = Solution and adsorbed phase degradation rates are independent.
        1 = Degradation everywhere the same (μ 1=μ.).
        2 = Degradation only in the liquid phase (μ1>0, μ.=O).
        3 = Degradation only in the adsorbed phase (μ1=0, μ.>0).
    phim: float, default = 0
        Mobile water fraction
    parms: pandas dataframe (nvar = number of columns)
        initial estimates with constrains (min, max) of transport parameters
    rhoth: float, default = 1.0
        ROU/THETA
    nvar:total number of parameters including mass estimation
    modb: int (0,1,2,3,4,5,6), default = 0
        flag of boundary mode (0 = zero input, 1 = dirac delta input, 2 = step input,
        3 = single pulse input, 4 = multiple pulse input, 5 = exponential input,
        6 = arbitrary input)
    pulse: list of dicts {'conc': float,'time': float}, default = []
        Input concentrations and times for each pulse
    modi: int (0,1,2,3,4), default = 0
        flag of initial concentration (0 = zero initial concentration, 1 = constant initial concentration,
        2 = stepwise initial concentration, 3 = exponential initial concentration, 
        4 = delta(z-z1) & constant initial concentration)
    cini: list of dicts {'conc': float,'z': float}, default = []
        Intitial concentrations and positions
    modp: int (0,1,2,3), default = 0
        Production value problem code (0 = zero production, 1 = constant production, 2 = stepwise production,
        3 = exponential production)
    mpro: int (0,1), default = 0
        Production function code for a nonequilibrium phase (
        0 = Same conditions for equilibrium and nonequilibrium phases, 
        1 = Different conditions for equilibrium and nonequilibrium phases)
    prodval1 : list of dicts {'gamma': float,'zpro': float}, default = []
        Production values and positions for the equilibrium phase
    prodval2 : list of dicts {'gamma': float,'zpro': float}, default = []
        Production values and positions for the nonequilibrium phase
    obsdata : pandas dataframe (nob=number of rows), default = []
        Observation data (t, c, z)
    """
    # Based on the CHECK function in DATA.FOR original Fortran code
    def __init__(self,
                 title="", 
                 inverse=0, 
                 mode=1,
                 nredu=1,
                 modc=1,
                 zl=1.0,
                 mit=0,
                 ilmt=0,
                 mass=0,
                 massst=0,
                 mneq=1,
                 mdeg=0,
                 phim=0,
                 rhoth=1.0,
                 parms=None,
                 modb=0,
                 pulse=[],
                 modi=0,
                 cini=[],
                 modp=0,
                 mpro=0,
                 prodval1=[],
                 prodval2=[],
                 obsdata=[],
                 **kwargs):
        
        self.title = title
        self.inverse = inverse
        self.mode = mode
        self.nredu = nredu
        self.modc = modc
        self.zl = zl
        self.mit = mit
        self.ilmt = ilmt
        self.mass = mass
        self.massst=massst,
        self.mneq = mneq
        self.mdeg = mdeg
        self.phim = phim
        self.phiim = 1.0 - self.phim # IMMOBILE WATER FRACTION
        self.rhoth = rhoth
        
        self.__set_parm(parms)
        
        self.modb = modb
        self.__set_bvp(pulse)
        
        self.modi = modi
        self.__set_ivp(cini)
        
        self.modp = modp
        self.__set_pvp(prodval1,prodval2,mpro)
        
        self.__set_obs(obsdata)

        self.__set_const(**kwargs)

        if self.mode in (3,4,5,6,8):
            self.update_stomode()
    
    def run(self,verbose=False):
        """ Run CXTFIT simulation """
        if self.inverse == 0 or self.mit == 0 or self.nfit == 0 :
            simparms = self.parms.loc['binit'].to_dict()
            self.direct(simparms, verbose=verbose)
        else:
            self.curvefit(verbose=verbose)
    
    def __set_bvp(self,pulse) :
        """ Check and set boundary value problem (BVP) Input """
        if self.modb in (1,2,3,4,5,6) :
            self.npulse = len(pulse)
            self.pulse = pulse
            self.tpulse = [item['time'] for item in pulse]
            self.cpulse = [item['conc'] for item in pulse]
        else:
            self.npulse = 0
            self.pulse = []
    
    def __set_ivp(self,cini) :
        """ Check and set initial value problem (BVP) Input """
        if self.modi in (1,2,3,4) :
            self.cini = cini
            self.nini = len(cini)
        else :
            self.nini = 0
            self.cini = []
    
    def __set_pvp(self, prodval1,prodval2,mpro):
        """ Check and set production value problem (PVP) Input """
        if self.modp in (1,2,3) :
            self.prodval1 = prodval1
            self.npro1 = len(prodval1)
            
            if self.mode in (2,4,6,8) : #NONEQUILIBRIUM MODEL
                self.mpro = mpro
                if self.mpro == 0: #SAME CONDITION FOR PHASE 1 & 2
                    self.prodval2 = prodval1
                    self.npro2 = self.npro1
                else: #DIFFERENT CONDITION FOR PHASE 1 & 2
                    self.prodval2 = prodval2
                    self.npro2 = len(prodval2)
            else: #EQUILIBRIUM MODEL
                self.npro2 = 0
                self.prodval2 = []
                self.mpro = 0
        else : #NO PRODUCTION
            self.npro1 = 0
            self.prodval1 = []
            self.npro2 = 0
            self.prodval2 = []
            self.mpro = 0
        
        self.gamma1 = [item['gamma'] for item in self.prodval1]
        self.zpro1 = [item['zpro'] for item in self.prodval1]
        self.gamma2 = [item['gamma'] for item in self.prodval2]
        self.zpro2 = [item['zpro'] for item in self.prodval2]
    
    def __set_obs(self,obsdata) :
        """ dataframe of observed and simulated data """
        self.nob = obsdata.shape[0] 
        self.cxtdata = obsdata 
    
    # Converted from CHECK function in DATA.FOR Fortran code
    def __set_parm(self,parms) :
        """ Dataframe of initial and estimated parameters with statistics  """

        # total number of variable parameters
        self.nvar = parms.shape[1] 
        
        self.parms = parms 

        # Number of estimated parameters
        self.nfit = self.parms.loc['bfit'].sum()

    def __set_const(self, **kwargs):
        """ Set constants and default values """
        defaults = {
            "mprint": 0, # print mode
            "ctol": 1.0e-10, # minimum value for the concentration
            "dtmin": 1.0e-7, # minimum time step size
            "dzmin": 1.0e-7, # minimum length step size
            "parmin": 1.0e-10, # minimum parm value
        }

        for key, value in defaults.items():
            setattr(self, key, kwargs.get(key, value))

    def direct(self,simparms, verbose=False):
        """ Forward CXT simulation """
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

    def residuals(self,params):
        """Residuals function for least squares optimization""" 
        cpar = self.binit.copy() 
        # Update parameters based on the fit flags 
        idx = 0
        for key, flag in self.bfit.items():
            if flag == 1:
                cpar[key] = params[idx]
                idx += 1

        if self.mode == 2:
            if self.mneq == 1: # ONE-SITE CHEMICAL NONEQUILIBRIUM MODEL beta = 1/r
                cpar['beta'] = 1/cpar['r']
            if self.mdeg == 1: # DEGRADATION EVERYWHERE THE SAME (μ1=μ2)
                cpar['mu2'] = (1.0 - cpar['beta']) / cpar['beta'] * cpar['mu1']
            elif self.mdeg == 2: # DEGRADATION ONLY IN THE LIQUID PHASE (μ1>0, μ2=0)
                if self.mneq in (0,3): # TWO-REGION PHYSICAL NONEQUILIBRIUM MODEL
                    cpar['mu2'] = (1.0 - self.phiim) / self.phim * cpar['mu1']
                elif self.mneq in (1,2): # ONE-SITE OR TWO-SITE CHEMICAL NONEQUILIBRIUM MODEL
                    cpar['mu2'] = 0.0
            elif self.mdeg == 3: # DEGRADATION ONLY IN THE ADSORBED PHASE (μ1=0, μ2>0)
                if self.mneq in (0,3): # TWO-REGION PHYSICAL NONEQUILIBRIUM MODEL
                    cpar['mu2'] = cpar['mu1'] / (cpar['beta'] * cpar['r'] - self.phiim) * ((1.0 - cpar['beta']) * cpar['r'] - self.phiim)
                elif self.mneq == 2: # TWO-SITE CHEMICAL NONEQUILIBRIUM MODEL
                    cpar['mu2'] = cpar['mu1'] / (cpar['beta'] * cpar['r'] - 1.0) * (1.0 - cpar['beta']) * cpar['r']
                elif self.mneq == 1: # ONE-SITE CHEMICAL NONEQUILIBRIUM MODEL
                    cpar['mu1'] = 0.0
        elif self.mode in (4, 6, 8):
            if self.mdeg == 1: # DEGRADATION EVERYWHERE THE SAME (μ1=μ2)
                cpar['mu2'] = (cpar['R'] - 1) * cpar['mu1']
            elif self.mdeg == 2: # DEGRADATION ONLY IN THE LIQUID PHASE (μ1>0, μ2=0)
                cpar['mu2'] = 0.0
            elif self.mdeg == 3: # DEGRADATION ONLY IN THE ADSORBED PHASE (μ1=0, μ2>0)
                cpar['mu1'] = 0.0

        if self.mode > 2:
            if self.modd == 1 : # SDLND=SDLNV
                cpar['SD.D'] = cpar['SD.v']
            if self.modk == 1 : # SDLNK=SDLNV
                cpar['SD.Kd'] = cpar['SD.v']
            if self.mal8 == 1 : # SDLNAL=SDLNV
                cpar['SD.alp'] = cpar['SD.v']

        csim,_,_,_ = self.model(cpar)
        return csim - self.cobs
    
    # Replaced the curve fitting algorithm in the original fortran code with 
    # the scipy.optimize.leastsq function to fit model to data
    def curvefit(self,verbose=False):
        """ Curve fitting using least squares optimization """
        
        self.cobs = self.cxtdata['cobs'].values
        self.binit = self.parms.loc['binit'].to_dict()
        self.bfit = self.parms.loc['bfit'].to_dict()

        if self.mode == 2:
            if self.mneq == 1: 
                self.bfit['beta'] = 0
            if self.mdeg in (1,2): 
                self.bfit['mu2'] = 0
            elif self.mdeg == 3: 
                if self.mneq in (0,2,3):
                    self.bfit['mu2'] = 0
                elif self.mneq == 1:
                    self.bfit['mu1'] = 0
        elif self.mode in (4,6,8):
            if self.mdeg in (1,2): 
                self.bfit['mu2'] = 0
            elif self.mdeg == 3: 
                self.bfit['mu1'] = 0

        self.modd = self.modk = self.mal8 = 0
        if self.mode > 2:
            if self.binit['SD.D'] < 1.0e-7: # D=CONSTANT
                self.binit['SD.D'] = 0
                self.bfit['SD.D'] = 0
            elif abs(self.binit['SD.v']- self.binit['SD.D']) < 1.0e-5:
                self.bfit['SD.D'] = 0 # SDLND=SDLNV
                self.modd = 1
            
            if self.binit['SD.Kd'] < 1.0e-7: # Kd=CONSTANT
                self.binit['SD.Kd'] = 0
                self.bfit['SD.Kd'] = 0
            elif abs(self.binit['SD.v']- self.binit['SD.Kd']) < 1.0e-5:
                self.bfit['SD.Kd'] = 0 # SDLNK=SDLNV
                self.modk = 1

        if self.mode == 8:
            if abs(self.binit['SD.v']- self.binit['SD.alp']) < 1.0e-5:
                self.bfit['SD.alp'] = 0 # SDLNAL=SDLNV 
                self.mal8 = 1

        initial_guess = [self.binit[key] for key, flag in self.bfit.items() if flag == 1]

        fitkeys = [key for key, flag in self.bfit.items() if flag == 1]

        self.nfit = len(fitkeys)

        if self.nfit > self.nob:
            raise ValueError("Number of parameters to fit exceeds number of observations")
        
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

    # Converted from DIRECT and MODEL function in MODEL.FOR original Fortran code
    def model(self,simparms):
        """CXT computation with the given parameters"""
        self.init_parms(simparms)
        
        # Loop over data points
        csim1=[0.0]*self.nob
        csim2=[0.0]*self.nob
        cvar1=[0.0]*self.nob
        cvar2=[0.0]*self.nob
        self.iskip = 0
        for i,rec in self.cxtdata.iterrows():
            if self.nredu in (0,1): # DIMENSIONLESS Time
                self.tt = rec['t'] * self.v / self.zl
            else :
                self.tt = rec['t']

            if self.nredu in (0,1,3): # DIMENSIONLESS Distance
                self.zz = rec['z'] / self.zl
            else :
                self.zz = rec['z']

            if self.mode ==1:
                c1,c2 = self.detcde()
                if c1 > self.ctol: csim1[i]=c1 
            elif self.mode==2:
                c1,c2 = self.detcde()
                if self.inverse == 1 and (self.modc == 4 or self.modc == 6):
                    c1 = self.beta * self.r * c1
                    c2 = (1 - self.beta) * self.r * c2
                if c1 > self.ctol: csim1[i]=c1 
                if c2 > self.ctol: csim2[i]=c2 
            else:
                c1,c2,v1,v2 = self.StoCDE()
                self.iskip = 1
                if c1 > self.ctol: csim1[i]=c1 
                if v1 > self.ctol: cvar1[i]=v1 
                if self.mode in (4,6,8):
                    if c2 > self.ctol: csim2[i]=c2 
                    if v2 > self.ctol: cvar2[i]=v2 
        return csim1,csim2,cvar1,cvar2

    def init_parms(self,simparms):
        """Initialize model parameters before computation"""

        # Velocity for dimensionless variables, value may change for each iteration
        if self.mode in (1,2): # Deterministic
            self.v = simparms['V'] # VELOCITY
        else: # Stochastic
            self.v = simparms['<V>'] # VELOCITY

        if self.modb == 3 and self.mass == 1:
            self.tpulse[1] = simparms['T2']

        if self.nredu in (0,1):
            # CHANGE pulse TO DIMENSIONLESS VARIABLES
            if self.modb in (3,4):
                self.tpulse = [i * self.v / self.zl for i in self.tpulse]
            if (self.modb == 5):
                self.tpulse[0] = self.tpulse[0] / self.v * self.zl

            # CHANGE gamma TO DIMENSIONLESS VARIABLES FOR MODE=1,3,5
            if self.mode in (1, 3, 5) and self.modp in (1, 2, 4) :
                    self.gamma1 = [i * self.zl / self.v for i in self.gamma1]

        # PARAMETER FOR TOTAL MASS
        if self.mass == 1:
            if self.modb ==2:
                self.cpulse[0] = simparms['MASS']
            if self.modb in (1,3):
                self.cpulse[0] = simparms['Cin']

        self.beta = 1.0 # NONEQUILIBRIUM PARTITIONING COEFFICIENT
        self.omega = 0.0 # NONEQUILIBRIUM MASS TRANSFER COEFFICIENT
        self.dmu2 = 0.0 # DEGRADATION RATE IN IMMOBILE PHASE
        self.alpha = 0.0 # ALPHA Kinetic rate coefficient
        
        if self.mode in (1,2): # deterministic model
            self.d = simparms['D'] # DISPERSIVITY
            self.p = self.v * self.zl / self.d # PECLET NUMBER
            self.r = simparms['R'] # RETARDATION FACTOR
            self.dmu1 = simparms['mu1'] # DEGRADATION RATE
        else: # stochastic model
            self.d = simparms['<D>']# DISPERSIVITY
            self.p = self.v * self.zl / self.d # PECLET NUMBER
            self.dk = simparms['<Kd>'] # DISTRIBUTION COEFFICIENT
            self.r = 1.0 + self.rhoth * self.dk # RETARDATION FACTOR
            self.dmu1 = simparms['mu1']# DEGRADATION RATE
            self.sdlnv = simparms['SD.v']
        
        if self.mode == 1: # deterministic equilibrium model
            if self.nredu <= 1:
                self.dmu1 = self.dmu1 * self.zl / self.v
        elif self.mode == 2: # deterministic nonequilibrium models
            self.beta = simparms['beta'] # nonequilibrium partitioning coefficient 
            self.omega = simparms['omega'] # nonequilibrium mass transfer coefficient
            self.dmu2 = simparms['mu2'] # DEGRADATION RATE IN IMMOBILE PHASE
        elif self.mode == 3: # bivariate distribution for V and Kd equilibrium
            if self.nredu in (0,1): # change to dimensionless variables
                self.dmu1 *= self.zl / self.v
            self.y = self.dk
            self.sdlny = simparms['SD.Kd'] # standard deviation of ln(Kd)
            self.sdln_d = simparms['SD.D'] # standard deviation of ln(D)
            self.corr = simparms['RhovKd'] # correlation coefficient
        elif self.mode == 4: # bivariate distribution for V and Kd nonequilibrium
            self.y = self.dk
            self.sdlny = simparms['SD.Kd']
            self.sdln_d = simparms['SD.D']
            self.corr = simparms['RhovKd']

            self.beta = 1.0 / self.r
            self.omega = simparms['omega']
            self.dmu2 = simparms['mu2']
        elif self.mode == 5: # bivariate distribution for V and D equilibrium
            if self.nredu in (0,1): # change to dimensionless variables
                self.dmu1 *= self.zl / self.v
            self.y = self.d
            self.sdlny = simparms['SD.D']
            self.sdlnk = simparms['SD.Kd']
            self.corr = simparms['RhovD']
        elif self.mode == 6: # bivariate distribution for V and D nonequilibrium
            self.y = self.d
            self.sdlny = simparms['SD.D']
            self.sdlnk = simparms['SD.Kd']
            self.corr = simparms['RhovD']
            self.beta = 1.0 / self.r
            self.omega = simparms['omega']
            self.dmu2 = simparms['mu2']
        elif self.mode == 8: # bivariate distribution for V and alpha nonequilibrium
            self.alpha = simparms['alpha'] # Kinetic rate coefficient
            self.y = self.alpha
            self.omega = self.alpha * (self.r - 1.0) * self.zl / self.v
            self.beta = 1.0 / self.r
            self.dmu2 = simparms['mu2']
            self.sdlny = simparms['SD.alp'] # standard deviation of ln(alpha)
            self.sdlnk = simparms['SD.Kd']
            self.sdln_d = simparms['SD.D']
            self.corr = simparms['RhovAl'] # correlation coefficient
        else :
            raise ValueError("Invalid mode")
        
        if self.mode > 2: # STOCHASTIC MODELS
            if self.sdlny < 1.0e-7:
                self.mstoch = 1 # CONSTANT Y
                self.corr = 0.0
            elif self.sdlnv < 1.0e-7:
                self.mstoch = 2 # CONSTANT V
                self.corr = 0.0
            else:
                self.mstoch = 4 # VARIABLE V AND Y

            self.mcorr = 0 # NO CORRELATION
            if self.mstoch == 4:
                if self.corr < -0.999999:
                    self.mcorr = -1 # NEGATIVE CORRELATION
                    self.mstoch = 3 # NEGATIVE CORRELATION BETWEEN  V & Y
                elif self.corr > 0.999999:
                    self.mcorr = 1 # POSITIVE CORRELATION
                    self.mstoch = 3 # POSITIVE CORRELATION BETWEEN V & Y
                elif abs(self.corr) < 1.0e-20:
                    self.mcorr = 0 # NO CORRELATION
                else:
                    self.mcorr = 2 # BIVARIATE DISTRIBUTION V & Y

            if self.mstoch != 2: # VARIABLE V 
                self.vmin, self.vmax = limit(self.v, self.sdlnv)

            if self.mstoch in (2,4): # VARIABLE Y
                self.ymin, self.ymax = limit(self.y, self.sdlny)
            
            print(self.vmin, self.vmax, self.ymin, self.ymax)

        return