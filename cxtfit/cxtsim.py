from .detcde import DetCDE
from .stocde import StoCDE

class CXTsim(DetCDE,StoCDE):
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
    modc : int (1,2,3,4,5,6), default = 1 
        flag of model concentration (1 = flux concentration or area-averaged flux concentration,
        2 = field-scale flux concentration, 3 = third-type resident concentration,
        4 = third-type total resident concentration, 5 = first-type resident concentration,
        6 = first-type total resident concentration)   
        MODC= 1,2,3,5  PHASE 1 CONC. IS USED FOR PARAMETER ESTIMATION.
        1 = real t & x  c2=<c2/k>, 2 = dimensionless t & z, 3 = dimensionless t & real x)
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
    massst: int, default = 0
        Mass distribution index for the stochastic CDE (
        0 = Amount of solute in each tube is proportional to v)
        1 = Amount of solute in each tube is constant regardless of v)
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
        if self.modb in (1,2,3,4,5,6) :
            self.__set_bvp(pulse)
        
        self.modi = modi
        if self.modi in (1,2,3,4) :
            self.__set_ivp(cini)
        
        self.modp = modp
        if self.modp in (1,2,3) :
            self.__set_pvp(prodval1,prodval2,mpro)
        
        self.__set_obs(obsdata)

        self.__set_const(**kwargs)

        self.curp = 'binit'
        if self.mode in (3,4,5,6,8):
            self.update_stomode()
        
        return
    
    def run(self,verbose=False):
        self.nit = 0
        if self.inverse == 0 or self.mit == 0 or self.np == 0 :
            self.inverse = 0
            self.__model(verbose=verbose)
        else:
            self.__inverse(verbose=verbose)
        return
    
    def __set_bvp(self,pulse) :
        # Check and set boundary value problem (BVP) Input

        self.npulse = len(pulse)
        self.pulse = pulse

        return
    
    def __set_ivp(self,cini) :
        # Check and set initial value problem (BVP) Input
        if self.modc >= 5 and abs(self.cini[1]) <= 1.0e-10:
            raise ValueError("Initial concentration must be greater 0 for first-type inlet")

        self.cini = cini
        self.ncini = len(cini)
        if self.nredu in (0,1) :
            if self.modb in (3,4) :
                for i in range(self.npulse) :
                    self.pulse[i]['time'] = self.pulse[i]['time'] * self.v/self.zl

        return
    
    def __set_pvp(self, prodval1,prodval2,mpro):
        # Check and set production value problem (PVP) Input

        self.prodval1 = prodval1
        self.npro1 = len(prodval1)
        self.prodval2 = prodval2
        self.npro2 = len(prodval2)
        
        self.mpro = mpro
        return
    
    def __set_obs(self,obsdata) :

        ## dataframe of observed and simulated data
        self.nob = obsdata.shape[0] 
        self.cxtdata = obsdata 

        return
    
    # Converted from CHECK function in DATA.FOR Fortran code
    def __set_parm(self,parms) :
        ## Dataframe of initial and estimated parameters with statistics 

        # total number of variable parameters
        self.nvar = parms.shape[1] 
        
        self.parms = parms 
        # Number of estimated parameters
        self.np = self.parms.loc['bfit'].sum()

        print(f"Total number of parameters: {self.nvar}")
        print(f"Number of estimated parameters: {self.np}")

        return
    
    # Converted from CONST1 function in USER.FOR Fortran code
    def __set_const(self, **kwargs):
        # Parameters for control numerical evaluations
        if "maxtry" in kwargs:
            self.maxtry = kwargs["maxtry"]
        else:
            self.maxtry = 50
        if "stopcr" in kwargs:
            self.stopcr = kwargs["stopcr"]
        else:
            self.stopcr = 0.0005
        
        # Parameters for the Marquardt inversion method
        # GA*GD = INITIAL VALUE FOR FUDGE FACTOR
        if "ga" in kwargs:
            self.ga = kwargs["ga"]
        else:
            self.ga = 0.01

        if "gd" in kwargs:
            self.gd = kwargs["gd"]
        else:
            self.gd = 10.0

        #DERL: INCREMENT TO EVALUATE VECTOR DERIVATIVES IN TERMS OF 
        #      MODEL PARAMETERS

        if "derl" in kwargs:
            self.derl = kwargs["derl"]
        else:
            self.derl = 1.0e-2

        # Stop criteria for the iteration based on the improvement of SSQ
        if "stsq" in kwargs:
            self.stsq = kwargs["stsq"]
        else:
            self.stsq = 1.0e-6

        # Initial number of integration points for Gauss Chebychev
        if "chebymm" in kwargs:
            self.mm = kwargs["chebymm"]
        else:
            self.mm = 100
        
        # Integration mode for Gauss Chebychev
        if "icheb" in kwargs:
            self.icheb = kwargs["icheb"]
        else:
            self.icheb = 0
        
        # Maximum constraint for OMEGA
        # OMMAX = 100 is recommended when L is equal to the observation scale
        if "ommax" in kwargs:
            self.ommax = kwargs["ommax"]
        else:
            self.ommax = 100.0

        # MPRINT flag for print mode
        if "mprint" in kwargs:
            self.mprint = kwargs["mprint"]
        else:
            self.mprint = 1

        # MODJH: Calculation control code for step input
        # MODJH=0; evaluate Eq.(3.23) or (3.24); 
        #      =1 Eq.(3.21) or (3.22) based on Goldstein's J-function
        if "modjh" in kwargs:
            self.modjh = kwargs["modjh"]
        else:
            self.modjh = 1 # default with Goldstein's J-function

        # Number of levels used for calculate series in exponential BVP
        if "phi_level" in kwargs:
            self.phi_level = kwargs["phi_level"]
        else:
            self.phi_level = 26
        
        if "ctol" in kwargs:
            self.ctol = kwargs["ctol"]
        else:
            self.ctol = 1.0e-10
        
        # Criteria for stopping the iteration in cherbychev integration
        if "stopch" in kwargs:
            self.stopch = kwargs["stopch"]
        else:
            self.stopch = 1.0e-3

        # Number of levels in cherbychev integration
        if "cheby_level" in kwargs:
            self.cheby_level = kwargs["cheby_level"]
        else:
            self.cheby_level = 11

        # INTM: Calculation control code for the numerical integration 
        #   1 log-transformed Romberg
        #   2 log-transformed Gauss-Chebyshev (current setting)
        if "intm" in kwargs:
            self.intm = kwargs["intm"]
        else:
            self.intm = 1

        # Minimum time step 
        if "dtmin" in kwargs:
            self.dtmin = kwargs["dtmin"]
        else:
            self.dtmin = 1.0e-7

        # minimum length step
        if "dzmin" in kwargs:
            self.dzmin = kwargs["dzmin"]
        else:
            self.dzmin = 1.0e-7

        # Minimum parm value
        if "pmin" in kwargs:
            self.pmin = kwargs["pmin"]
        else:
            self.pmin = 1.0e-10
        
        #MODP1:  Calculation control code for constant production 
        #                   for the equilibrium CDE 
        #   MODP1=0; evaluate the integral in Eq.(2.32) 
        #        =1   Eq.(2.33) or (2.34) (current setting)
        if "modp1" in kwargs:
            self.modp1 = kwargs["modp1"]
        else:
            self.modp1 = 1

        self.iskip = 0

    def __set_dimless(self):
        # Velocity for dimensionless variables, value may change for each iteration
        if self.mode in (1,2):
            self.v = self.parms.loc[self.curp,'V'] # VELOCITY
        else :
            self.v = self.parms.loc[self.curp,'<V>']
        
        if self.nredu <= 1:
            if (self.modb == 3 or self.modb == 4):
                for i in range(self.npulse):
                    self.pulse[i]['time'] = self.pulse[i]['time'] * self.v / self.zl
            if (self.modb == 5):
                self.pulse[0]['time'] = self.pulse[0]['time'] / self.v * self.zl

            # CHANGE gamma TO DIMENSIONLESS VARIABLES FOR MODE=1,3,5
            if ((self.mode % 2) == 1) and (self.modp in (1,2,4)):
                for i in range(self.npro1) :
                    self.prodval1[i]['gamma'] = self.prodval1[i]['gamma'] * self.zl / self.v

        # PARAMETER FOR TOTAL MASS
        if self.mass == 1:
            if self.modb ==2:
                self.pulse[0]['conc'] = self.parms.loc[self.curp,'MASS']
            if self.modb in (1,3):
                self.pulse[0]['conc'] = self.parms.loc[self.curp,'Cin']

    # Converted from DIRECT and MODEL function in MODEL.FOR original Fortran code
    def __model(self,verbose=False):
        """Assign coefficients"""
        self.__set_dimless()
        if self.mode in (1,2):
            self.init_detcde()
        else:
            self.init_stocde()
        
        csim1=csim2=cvar1=cvar2=[]
        for i,rec in self.cxtdata.iterrows():
            self.tt = rec['t']
            self.zz = rec['z']
            if self.nredu in (0,1,3):
                self.zz /= self.zl
            if self.nredu in (0,1):
                self.tt *= self.v / self.zl

            if self.mode ==1:
                c1,c2 = self.DetCDE()
                if c1 < self.ctol: c1 = 0            
                csim1.append(c1)
            elif self.mode==2:
                c1,c2 = self.DetCDE()
                if self.inverse == 1 and (self.modc == 4 or self.modc == 6):
                    c1 = self.beta * self.r * c1
                    c2 = (1 - self.beta) * self.r * c2
                if c1 < self.ctol:  c1 = 0            
                if c2 < self.ctol:  c2 = 0            
                csim1.append(c1)
                csim2.append(c2)
            else:
                c1,c2,v1,v2 = self.StoCDE()
                self.iskip = 1
                if c1 < self.ctol: c1 = 0
                if c2 < self.ctol: c2 = 0
                if v1 < self.ctol: v1 = 0
                if v2 < self.ctol: v2 = 0
                csim1.append(c1)
                csim2.append(c2)
                cvar1.append(v1)
                cvar2.append(v2)

        self.cxtdata[f'c1sim{self.nit}'] = csim1
        if self.mode in (2,4,6,8): # nonequilibrium model
            self.cxtdata[f'c2sim{self.nit}'] = csim2
        if self.mode in (3,4,5,6,8): # stochastic model
            self.cxtdata[f'c1var{self.nit}'] = cvar1
        if self.mode in (4,6,8): # stochastic nonequilibrium model
            self.cxtdata[f'c2var{self.nit}'] = cvar2

        print(f"Simulation {self.nit} completed")
        print(self.cxtdata)
        return

    def __inverse(self,verbose=False):
        # MCON: Calculation control code for concentrations 
        # 0, Calculate equilibrium and nonequilibrium concentrations;
        # 1, only equilibrium; 3, only nonequilibrium
        self.mcon = 0

        # if self.inverse == 1:
        #     k = 0
        #     nu1 = self.nvar + 1
        #     nu2 = self.nvar * 2
        #     for i in range(nu1, nu2 + 1):
        #         if self.index[i - self.nvar - 1] == 0:
        #             continue
        #         k += 1
        #         bn[i - 1] = bn[k - 1]

        return
