import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
import scipy.stats as stats

from .stocde import StoCDE, limit

class CXTsim(StoCDE):
    """ CXT Simulation object

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
                 mneq=0,
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
                 verbose=False,
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
        self.massst=massst
        self.mneq = mneq
        self.mdeg = mdeg
        self.phim = phim
        self.phiim = 1.0 - self.phim # IMMOBILE WATER FRACTION
        self.rhoth = rhoth
        
        self.__set_parm(parms,verbose = verbose)
        
        self.modb = modb
        self.__set_bvp(pulse ,verbose = verbose)
        
        self.modi = modi
        self.__set_ivp(cini)
        
        self.modp = modp
        self.__set_pvp(prodval1, prodval2, mpro ,verbose = verbose)
        
        self.__set_obs(obsdata ,verbose = verbose)

        self.__set_const(verbose = verbose, **kwargs)

    def run(self,verbose=False):
        """ Run CXTFIT simulation """
        if self.inverse == 0 or self.mit == 0 or self.nfit == 0 :
            simparms = self.parms.loc['binit'].to_dict()
            self.direct(simparms, verbose=verbose)
        else:
            self.curvefit(verbose=verbose)
    
    def __set_bvp(self,pulse,verbose=False) :
        """ Check and set boundary value problem (BVP) Input """
        if self.modb in (1,2,3,4,5,6) :
            self.npulse = len(pulse)
            self.tpulse = [item['time'] for item in pulse]
            self.cpulse = [item['conc'] for item in pulse]
        else:
            self.npulse = 0
            self.tpulse = [0]
            self.cpulse = [0]
    
    def __set_ivp(self,cini,verbose=False) :
        """ Check and set initial value problem (BVP) Input """
        if self.modi in (1,2,3,4) :
            self.nini = len(cini)
            self.cini = [item['conc'] for item in cini]
            self.zini = [item['z'] for item in cini]
        else :
            self.nini = 0
            self.cini = [0]
            self.zini = [0]
    
    def __set_pvp(self, prodval1,prodval2,mpro,verbose=False) :
        """ Check and set production value problem (PVP) Input """
        if self.modp in (1,2,3) :
            self.npro1 = len(prodval1)
            self.gamma1 = [item['gamma'] for item in prodval1]
            self.zpro1 = [item['zpro'] for item in prodval1]
            
            if self.mode in (2,4,6,8) : #NONEQUILIBRIUM MODEL
                self.mpro = mpro
                if self.mpro == 0: #SAME CONDITION FOR PHASE 1 & 2
                    self.npro2 = self.npro1
                    self.gamma2 = [item['gamma'] for item in prodval1]
                    self.zpro2 = [item['zpro'] for item in prodval1]
                else: #DIFFERENT CONDITION FOR PHASE 1 & 2
                    self.npro2 = len(prodval2)
                    self.gamma2 = [item['gamma'] for item in prodval2]
                    self.zpro2 = [item['zpro'] for item in prodval2]
            else: #EQUILIBRIUM MODEL
                self.npro2 = 0
                self.mpro = 0
                self.gamma2 = [0]
                self.zpro2 = [0]
        else : #NO PRODUCTION
            self.npro1 = 0
            self.gamma1 = [0]
            self.zpro1 = [0]
            self.npro2 = 0
            self.mpro = 0
            self.gamma2 = [0]
            self.zpro2 = [0]

    def __set_obs(self,obsdata,verbose=False) :
        self.nob = obsdata.shape[0] 
        self.cxtdata = obsdata.copy() 
    
    # Converted from CHECK function in DATA.FOR Fortran code
    def __set_parm(self,parms,verbose=False):
        """ Intitalize model parameters  """

        # total number of variable parameters
        self.nvar = parms.shape[1] 
        self.parms = parms.copy()

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

        #   ----- DISPERSION COEFFICIENT FOR STOCHASTIC MODEL FOR MODE 3,4,8--------
        self.modd = 0
        if self.mode in (3, 4, 8) :
            self.bfit['SD.D'] = 0
            if self.binit['SD.D'] < 1.0e-07:
                self.modd = 0 # D=CONSTANT
            elif abs(self.binit['SD.v'] - self.binit['SD.D']) < 1.0e-05:
                self.modd = 2 # SDLND=SDLNV
            else:
                self.modd = 1 # POSITIVE CORRELATION V&D.

        #  --- DISPERSION COEFFICIENT FOR STOCHASTIC MODEL FOR MODE 5,6--------
        self.md56 = 0 # SDLND AND SALNV ARE INDEPENSENT
        if self.mode in (5,6) and abs(self.binit['SD.v'] - self.binit['SD.D']) < 1.0e-05:
            self.md56 = 1 # SDLND=SDLNV
            self.bfit['SD.D'] = 0

        #    ----- DISTRIBUTION COEF FOR STOCHASTIC MODEL FOR MODE 5,6,8--------
        #    ASSUME NEGATIVE CORRELATION V&Kd FOR MODE 5,6,8
        self.modk = 0 # R=CONSTANT
        if self.mode in (5, 6, 8):
            self.bfit['SD.Kd'] = 0
            if self.binit['SD.Kd'] < 1.0e-07:
                self.modk = 0
            elif abs(self.binit['SD.v'] - self.binit['SD.Kd']) < 1.0e-05:
                self.modk = -2 # SDLNK=SDLNV
            else:
                self.modk = -1 # NEGATIVE CORRELATION V&Kd

        #  --- DISTRIBUTION COEFFICIENT FOR MODE 3,4--------
        self.mk34 = 0 # SDLNK AND SDLNV ARE INDEPENSENT
        if self.mode in (3, 4) and abs(self.binit['SD.v'] - self.binit['SD.Kd']) < 1.0e-05:
            self.mk34 = 1 # SDLNK=SDLNV
            self.bfit['SD.Kd'] = 0

        #  --- ALPHA FOR MODE 8--------
        self.mal8 = 0 # SDLNAL AND SDLNV ARE INDEPENSENT
        if self.mode == 8 and abs(self.binit['SD.v'] - self.binit['SD.alp']) < 1.0e-05:
            self.mal8 = 1 # SDLNAL=SDLNV
            self.bfit['SD.alp'] = 0

        self.nfit = sum(self.bfit.values())
        if self.nfit and verbose:
            print(f"Initial parameters: {self.binit}")
            print(f"Number of parameters to fit: {int(self.nfit)}")
            print(f"Parameters to fit: {[key for key, flag in self.bfit.items() if flag == 1]}")

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
        cpar = self.full_parms(params) 
        # print(f"Parameters estimates \n {cpar}")
        csim,_,_,_ = self.model(cpar)

        return csim - self.cobs
    
    # Replaced the curve fitting algorithm in the original fortran code with 
    # the scipy.optimize.leastsq function to fit model to data
    def curvefit(self,verbose=False):
        """ Curve fitting using least squares optimization """
        if 'cobs' in self.cxtdata.columns:
            self.cobs = self.cxtdata['cobs'].values
            if self.nfit > self.nob:
                raise ValueError(f"Number of parameters to fit {self.nfit} exceeds number of observations {self.nob}")
        else:
            raise ValueError("Observed data 'cobs' not found in the input dataframe.")
        
        initial_guess = [self.binit[key] for key, flag in self.bfit.items() if flag == 1]
        if verbose:
            print(f"Initial parameters estimates \n {self.parms}")
        
        if self.ilmt : # Use bounds
            bmax = self.parms.loc['bmax'].to_dict()
            bmin = self.parms.loc['bmin'].to_dict()
            lower_bounds = [bmin[key] for key, flag in self.bfit.items() if flag == 1]
            upper_bounds = [bmax[key] for key, flag in self.bfit.items() if flag == 1]

            # Perform the least squares optimization, default setting works great
            leastsq = least_squares(self.residuals, initial_guess, 
                                        bounds=(lower_bounds, upper_bounds), diff_step=1e-3, verbose=verbose)
        else:
            leastsq = least_squares(self.residuals, initial_guess, diff_step=1e-3, verbose=verbose)

        self.bfinal = self.full_parms(leastsq.x)
        if verbose:
            print(f"Final parameters \n {self.bfinal}")

        if verbose:
            self.fit_stats(leastsq)

        self.direct(self.bfinal, verbose=verbose)
        return
    
    def full_parms(self, params):
        cpar = self.binit.copy()
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
            if self.modd : # SDLND=SDLNV
                cpar['SD.D'] = cpar['SD.v']
            if self.modk : # SDLNK=SDLNV
                cpar['SD.Kd'] = cpar['SD.v']
            if self.mal8 : # SDLNAL=SDLNV
                cpar['SD.alp'] = cpar['SD.v']
        return cpar
    
    def fit_stats(self, leastsq):
        fitkeys = [key for key, flag in self.bfit.items() if flag == 1]
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

        print(f"{self.stats_dict}")

    # Converted from DIRECT and MODEL function in MODEL.FOR original Fortran code
    def model(self,simparms):
        """CXT computation with the given parameters"""
        self.init_parms(simparms)

        # Loop over data points
        csim1=[0.0]*self.nob
        csim2=[0.0]*self.nob
        cvar1=[0.0]*self.nob
        cvar2=[0.0]*self.nob
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
                # print(f"V={self.v}, D={self.d}, R={self.r}, P={self.p}, Mu1={self.dmu1}, Mu2={self.dmu2}, Beta={self.beta}, Omega={self.omega}")
                # print(f"Gamma1={self.gamma1}, Gamma2={self.gamma2}, Y={self.y}, Corr={self.corr}, Alpha={self.alpha}")
                # print(f"SDlnV={self.sdlnv}, SDlnD={self.sdln_d}, SDlnK={self.sdlnk}")
                # print(f'MSTOCH={self.mstoch}, MCORR={self.mcorr}, MODD={self.modd}, MD56={self.md56}, MODK={self.modk}, MK34={self.mk34}, MAL8={self.mal8}')
                # print(f"cpulse={self.cpulse}, tpulse={self.tpulse}")
                c1,c2,v1,v2 = self.StoCDE()
                csim1[i]=c1 
                cvar1[i]=v1 
                # if c1 > self.ctol: csim1[i]=c1 
                # if v1 > self.ctol: cvar1[i]=v1 
                if self.mode in (4,6,8):
                    csim2[i]=c2 
                    cvar2[i]=v2
                    # if c2 > self.ctol: csim2[i]=c2 
                    # if v2 > self.ctol: cvar2[i]=v2 

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
                self.tpulse = [x * self.v / self.zl for x in self.tpulse]
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

        if self.mode in (1,2): # deterministic model
            self.d = simparms['D'] # DISPERSIVITY
            self.r = simparms['R'] # RETARDATION FACTOR
        else: # stochastic model
            self.d = simparms['<D>']# DISPERSIVITY
            self.dk = simparms['<Kd>'] # DISTRIBUTION COEFFICIENT
            self.r = 1.0 + self.rhoth * self.dk # RETARDATION FACTOR
            self.sdlnv = simparms['SD.v']
            self.sdln_d = simparms['SD.D']
            self.sdlnk = simparms['SD.Kd']        

        self.p = self.v * self.zl / self.d # PECLET NUMBER
        self.dmu1 = simparms['mu1'] # DEGRADATION RATE
        if self.mode in (1,3,5):
            if self.nredu <= 1:
                self.dmu1 *= self.zl / self.v

        if self.mode == 2: # deterministic nonequilibrium models
            self.beta = simparms['beta'] # NONEQUILIBRIUM PARTITIONING COEFFICIENT
        elif self.mode in (4,6,8):
            self.beta = 1.0 / self.r
        else :
            self.beta = 1.0

        if self.mode in (2,4,6,8): # NONEQUILIBRIUM MODELS
            self.omega = simparms['omega'] # nonequilibrium mass transfer coefficient
            self.dmu2 = simparms['mu2'] # DEGRADATION RATE IN IMMOBILE PHASE
        else :
            self.omega = 0.0
            self.dmu2 = 0.0
        
        if self.mode in (3, 4): # bivariate distribution for V and Kd
            self.y = self.dk
            self.sdlny = self.sdlnk  # standard deviation of ln(Kd)
            self.corr = simparms['RhovKd'] # correlation coefficient

        if self.mode in (5,6): # bivariate distribution for V and D 
            self.y = self.d
            self.sdlny = self.sdln_d
            self.corr = simparms['RhovD']

        self.alpha = 0.0 # ALPHA Kinetic rate coefficient
        if self.mode == 8: # bivariate distribution for V and alpha nonequilibrium
            self.alpha = simparms['alpha'] # Kinetic rate coefficient
            self.y = self.alpha
            self.sdlny = simparms['SD.alp'] # standard deviation of ln(alpha)
            self.corr = simparms['RhovAl'] # correlation coefficient
        
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
                # print(self.vmin, self.vmax)

            if self.mstoch in (2,4): # VARIABLE Y
                self.ymin, self.ymax = limit(self.y, self.sdlny)
                # print(self.ymin, self.ymax)
        return
    
    def plot_btc(self,**kwargs):
        """ Plot breakthrough curve """
        if 'cobs' in self.cxtdata.columns:
            self.cxtdata.plot.scatter(x='t', y='cobs', label='Observed',**kwargs)

        if self.mode == 1 : # One phase
            self.cxtdata.plot(x='t', y='csim', label='Phase 1', **kwargs)
        elif self.mode == 2 : # Two phase
            self.cxtdata.plot(x='t', y='csim', label='Phase 1', **kwargs)
            self.cxtdata.plot(x='t', y='csim2', label='Phase 2', 
                              linestyle='dashed', **kwargs)
        elif self.mode in (3, 5) :
            self.cxtdata.plot(x='t', y='csim', label='Phase 1', **kwargs)
            plt.fill_between(self.cxtdata['t'], 
                             self.cxtdata['csim'] - self.cxtdata['cvar'], 
                             self.cxtdata['csim'] + self.cxtdata['cvar'], 
                             alpha=0.2, label='Phase 1 ± Variance',**kwargs)
        elif self.mode in (4,6,8) :
            self.cxtdata.plot(x='t', y='csim', label='Phase 1', **kwargs)
            plt.fill_between(self.cxtdata['t'], 
                             self.cxtdata['csim'] - self.cxtdata['cvar'], 
                             self.cxtdata['csim'] + self.cxtdata['cvar'], 
                             alpha=0.2, label='Phase 1 ± Variance',**kwargs)
            self.cxtdata.plot(x='t', y='csim2', label='Phase 2',
                              linestyle='dashed', **kwargs)
            plt.fill_between(self.cxtdata['t'], 
                             self.cxtdata['csim2'] - self.cxtdata['cvar2'], 
                             self.cxtdata['csim2'] + self.cxtdata['cvar2'], 
                             alpha=0.2, label='Phase 2 ± Variance',**kwargs)
        
    def plot_profile(self,**kwargs):
        """ Plot concentration profile """
        if 'cobs' in self.cxtdata.columns:
            self.cxtdata.plot.scatter(x='z', y='cobs', label='Observed', **kwargs)
        if self.mode == 1 : # One phase
            self.cxtdata.plot(x='z', y='csim', label='Phase 1', **kwargs)
        elif self.mode == 2 : # Two phase
            self.cxtdata.plot(x='z', y='csim', label='Phase 1', **kwargs)
            self.cxtdata.plot(x='z', y='csim2', label='Phase 2',
                              linestyle='dashed', **kwargs)
        elif self.mode in (3, 5) :
            self.cxtdata.plot(x='z', y='csim', label='Phase 1', **kwargs)
            plt.fill_between(self.cxtdata['z'], 
                             self.cxtdata['csim'] - self.cxtdata['cvar'], 
                             self.cxtdata['csim'] + self.cxtdata['cvar'], 
                             alpha=0.2, label='Phase 1 ± Variance',**kwargs)
        elif self.mode in (2,4,6,8) :
            self.cxtdata.plot(x='z', y='csim', label='Phase 1', **kwargs)
            plt.fill_between(self.cxtdata['z'], 
                             self.cxtdata['csim'] - self.cxtdata['cvar'], 
                             self.cxtdata['csim'] + self.cxtdata['cvar'], 
                             alpha=0.2, label='Phase 1 ± Variance',**kwargs)
            self.cxtdata.plot(x='z', y='csim2', label='Phase 2', 
                              linestyle='dashed', **kwargs)
            plt.fill_between(self.cxtdata['z'], 
                             self.cxtdata['csim2'] - self.cxtdata['cvar2'], 
                             self.cxtdata['csim2'] + self.cxtdata['cvar2'], 
                             alpha=0.2, label='Phase 2 ± Variance',**kwargs)
