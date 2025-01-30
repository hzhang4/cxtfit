from .cxtmodel import CXTmodel

class CXTsim(CXTmodel):
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
    mprint: int (0,1,2), default = 0
        flag of print mode (0 = no print, 1 = print conc vs. time, 2 = conc vs. depth)
    maxtry: int, default = 50
        Maximum number of trials for the iteration
    stopcr: float, default = 0.0005
        Iteration criterion. The curve-fitting process stops when the relative
        change in the ratio of all coefficients becomes less than STOPCR.
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
        
        return
    
    def run(self,verbose=False):
        if self.inverse == 0 or self.mit == 0 or self.nfit == 0 :
            simparms = self.parms.loc['binit'].to_dict()
            self.direct(simparms, verbose=verbose)
        else:
            self.curvefit(verbose=verbose)
        return
    
    def __set_bvp(self,pulse) :
        # Check and set boundary value problem (BVP) Input
        if self.modb in (1,2,3,4,5,6) :
            self.npulse = len(pulse)
            self.pulse = pulse
            self.tpulse = [item['time'] for item in pulse]
            self.cpulse = [item['conc'] for item in pulse]
        else:
            self.npulse = 0
            self.pulse = []
    
    def __set_ivp(self,cini) :
        # Check and set initial value problem (BVP) Input
        if self.modi in (1,2,3,4) :
            self.cini = cini
            self.nini = len(cini)
        else :
            self.nini = 0
            self.cini = []
    
    def __set_pvp(self, prodval1,prodval2,mpro):
        # Check and set production value problem (PVP) Input
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

        ## dataframe of observed and simulated data
        self.nob = obsdata.shape[0] 
        self.cxtdata = obsdata 
    
    # Converted from CHECK function in DATA.FOR Fortran code
    def __set_parm(self,parms) :
        ## Dataframe of initial and estimated parameters with statistics 

        # total number of variable parameters
        self.nvar = parms.shape[1] 
        
        self.parms = parms 
        # Number of estimated parameters
        self.nfit = self.parms.loc['bfit'].sum()
    
    # Converted from CONST1 function in USER.FOR Fortran code
    def __set_const(self, **kwargs):
        defaults = {
            "maxtry": 50, # maximum number of trials for the iteration
            "stopcr": 0.0005, # iteration criterion
            "ga": 0.01, # parameters for the marquardt inversion method
            "gd": 10.0, # ga*gd = initial value for fudge factor
            "iderl": 1.0e-2, #derl: initial increment to evaluate vector derivatives 
            "stsq": 1.0e-6, # stop criteria for the iteration based on the improvement of ssq
            "chebymm": 100, # initial number of integration points for gauss chebychev
            "mmax": 6400, # max number of integration points for gauss chebychev
            "icheb": 0,  # integration mode for gauss chebychev
            "cheby_level": 11,  # number of levels in cherbychev integration
            "ommax": 100.0, # maximum constraint for omega
            "mprint": 0, # print mode
            "modjh": 1, # 0 = eq.(3.23) or (3.24); 1 = eq.(3.21) or (3.22) goldstein's j-function
            "phi_level": 11, # number of levels used for calculate series in exponential bvp
            "ctol": 1.0e-10, # minimum value for the concentration
            "dtmin": 1.0e-7, # minimum time step size
            "dzmin": 1.0e-7, # minimum length step size
            "pmin": 1.0e-10, # minimum parm value
            "stopch": 1.0e-3, # criteria for stopping the iteration in cherbychev integration
            "intm": 1, # intm: calculation control code for the numerical integration
            "modp1": 1, # modp1: flag for constant production 0; eq.(2.32) 1; eq.(2.33) or (2.34)
            "iskip": 0, # iskip: Calculation control code for the evaluation of the integral limits
        }

        for key, value in defaults.items():
            setattr(self, key, kwargs.get(key, value))
