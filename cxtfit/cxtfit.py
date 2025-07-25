"""
Python class implementation of CXTFIT VERSION 2.1 original Fortran code 
developed by Nobuo Toride at U.S. SALINITY LABORATORY in April 17, 1999.
                                                                
Non-linear least-squares analysis of c(x,t) data for one-dimensional 
deterministic or stochastic equilibrium and nonequilibrium convective 
dispersive equation (CDE)       

N. Toride, F. J. Leij, and M. Th. van Genuchten. The CXTFIT Code for 
Estimating Transport Parameters from Laboratory or Field Tracer Experiments, 
Version 2.1 , Research Report No. 137, U. S. Salinity Laboratory, USDA, ARS, 
Riverside, CA. 1999.   

"""
__version__ = "1.0"
__author__ = "Hua Zhang @ SWFWMD"

from .cxtsim import CXTsim
import pandas as pd

class CXTfit:
    """ Main Class of CXTFIT Code
    Parameters
    ----------
    simcases: list of CXTFitcase
        list of simulation cases
    """

    def __init__(self, 
                 simcases=[],
                 ):
        
        if simcases is None:
            self.simcases = self.default_case()
        else:
            self.simcases = simcases
        
        self.ncase = len(simcases)
        return

    def run(self,verbose=False):
        """Run the CXTFIT code
        """
        for i,simcase in enumerate(self.simcases):
            if verbose:
                print(f"\n\n{'-' * 100}\nRunning Simulation Case {i+1}\n\n{simcase.title}\n{'-' * 100}\n")
            simcase.run(verbose=verbose)
        return
    
    # Based on the DATAIN function in DATA.FOR original Fortran code
    @classmethod
    def load(
        cls,
        f: str,
        verbose = False,
    ):
        """
        load data from existing cxtfit input file 
        Parameters
        ----------
        f : str
            Path to cxtfit input file to load.
        """
        if not hasattr(f, "read"):
            infile = open(f, "r")
        else:
            infile = f

        simcases = []
        nc = int(infile.readline().strip()) # Number of cases
        for i in range(nc):
            ### BLOCK A: MODEL DESCRIPTION ###
            infile.readline()
            title = infile.readline().strip()
            infile.readline()
            infile.readline()
            tline = infile.readline().split()

            inverse = int(tline[0]) # Inverse problem flag (0,1)
            mode = int(tline[1]) # Model flag (1-8)
            nredu = int(tline[2]) # Dimension reduction flag (0-3)

            if mode == 7: mode = 8

            infile.readline()  # Skip line
            tline = infile.readline().split()
            if mode % 2 == 1 and nredu <= 1:
                modc = int(tline[0]) # Concentration model flag (1-6)
                zl = 1.0 # Characteristic length for dimensionless parameters (m)
            else:
                modc = int(tline[0])
                zl = float(tline[1])
            if verbose:
                print(f"\n\n{'-' * 100}\nLoading Simulation Case {i+1}\n\n{title}\n{'-' * 100}\n")
                text1={0: 'Forward problem', 1: 'Inverse problem'}[inverse]
                print(f" {text1} (INVERSE={inverse})\n")
                text1={1 : 'DETERMINISTIC EQUILIBRIUM  MODEL', 
                       2 : 'DETERMINISTIC NONEQUILIBRIUM  MODEL',
                       3 : 'STOCHASTIC KD&V EQUILIBRIUM',
                       4 : 'STOCHASTIC KD&V NONEQUILIBRIUM',
                       5 : 'STOCHASTIC D&V EQUILIBRIUM',
                       6 : 'STOCHASTIC D&V NONEQUILIBRIUM',
                       8 : 'STOCHASTIC ALPHA & V NONEQUILIBRIUM'}[mode]
                print(f" {text1} (MODE = {mode})\n")
                text1={0: 'REAL T ,X , <C2>  FOR MODE=4,6,8',
                    1: 'REAL T & X  C2=<C2/K>',
                    2: 'DIMENSIONLESS T & Z',
                    3: 'DIMENSIONLESS T & REAL X'}[nredu]
                print(f" {text1} (NREDU = {nredu})\n")
                text1={1: "FLUX CONCENTRATION OR AREA-AVERAGED FLUX CONCENTRATION",
                        2: "FIELD-SCALE FLUX CONCENTRATION",
                        3: "THIRD-TYPE RESIDENT CONCENTRATION",
                        4: "THIRD-TYPE TOTAL RESIDENT CONCENTRATION",
                        5: "FIRST-TYPE RESIDENT CONCENTRATION",
                        6: "FIRST-TYPE TOTAL RESIDENT CONCENTRATION"
                }[modc]
                print(f" {text1} (MODC = {modc})\n")

            # READ PARAMETERS FOR AN INVERSE PROBLEM (BLOCK B)
            mit = 0
            ilmt = 0
            mass = 0
            massst = 0
            mneq = 0
            mdeg = 0
            phim = 0.0
            rhoth = 1.0
            mprint = 0

            if inverse >= 1:
                infile.readline()
                infile.readline()

                tline = infile.readline().split()
                mit = int(tline[0])  # Maximum number of iterations (MODAT common).
                ilmt = int(tline[1]) # Parameter constraint code (0 = No constraints,1 = Constraints).
                mass = int(tline[2]) # Total mass estimation code (0 = No estimation for total mass.
                                     # 1 = Total mass included in estimation procedure).
                
                if mode % 2 == 0:
                    infile.readline()  # Skip line
                    tline = infile.readline().split()
                    mneq = int(tline[0]) # Nonequilibrium model code (0-3)
                    mdeg = int(tline[1]) # Degradation estimation code (0-3).
                    if ilmt==0 and mneq ==3:
                        raise ValueError('ERROR! ILMT SHOULD BE 1 FOR MNEQ = 3')
                    if mneq == 3 or (mneq == 0 and mdeg >= 2) :
                        infile.readline()
                        phim = float(infile.readline().strip()) # Mobile water fraction
                
                if verbose:
                    print(f" MAXIMUM NUMBER OF ITERATIONS (MIT = {mit})\n")
                    text1={0: 'NO ', 1: ''}[ilmt]
                    print(f" {text1} PARAMETER CONSTRAINTS (ILMT = {ilmt})\n")
                    text1={0: 'NO', 1: ''}[mass]
                    print(f" {text1} TOTAL MASS ESTIMATION (MASS = {mass})\n")
                    TEXT1={0: 'TWO-REGION PHYSICAL NONEQUILIBRIUM MODEL ',
                           1: 'ONE-SITE CHEMICAL NONEQUILIBRIUM MODEL ',
                           2: 'TWO-SITE CHEMICAL NONEQUILIBRIUM MODEL ',
                           3: 'TWO-REGION PHYSICAL NONEQUILIBRIUM MODEL WITH INTERNAL CONSTRAINTS'}[mneq]
                    print(f" {TEXT1} (MNEQ = {mneq})\n")
                    text1 = {
                        0: 'SOLUTION AND ADSORBED PHASE DEGRADATION RATES ARE INDEPENDENT',
                        1: 'DEGRADATION EVERYWHERE THE SAME (MU1 = MU2)',
                        2: 'DEGRADATION ONLY IN THE LIQUID PHASE (MU1 > 0, MU2 = 0)',
                        3: 'DEGRADATION ONLY IN THE ADSORBED PHASE (MU1 = 0, MU2 > 0)'
                    }[mdeg]
                    print(f" {text1} (MDEG = {mdeg})\n")
                    print(f" MOBILE WATER FRACTION (PHIM = {phim})\n")
            else:
                mass=0
                # MODE=4,6,8; ONE-SITE MODEL
                if mode in (4,6,8) : mneq=1

            # READ TRANSPORT PARAMETERS  (BLOCK C)
            infile.readline()
            infile.readline()
            ntp, bname = {
                1: (4, ['V', 'D', 'R', 'mu1']),
                2: (7, ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']),
                3: (8, ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']),
                4: (10, ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']),
                5: (8, ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovD']),
                6: (10, ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovD']),
                8: (11, ['<V>', '<D>', '<Kd>', 'alpha', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'SD.alp', 'RhovAl'])
            }[mode]

            # READ INITIAL ESTIMATES
            tline = infile.readline().split()
            if mode <=2:
                binit = [float(x) for x in tline[:ntp]]
            else:
                binit = [float(x) for x in tline[:ntp]]
                rhoth = float(tline[ntp]) # ROU/THETA
            
            # READ INDICES
            bfit = [0] * ntp
            bmin = [0.0] * ntp
            bmax = [0.0] * ntp
            if inverse >= 1:
                # Parameter estimation index (1 = Estimate, 0 = Fixed)
                bfit = [int(x) for x in infile.readline().split()[:ntp]]
                # READ CONSTRAINTS ON PARAMETER VALUES
                if ilmt > 0:
                    bmin = [float(x) for x in infile.readline().split()[:ntp]]
                    bmax = [float(x) for x in infile.readline().split()[:ntp]]
            if verbose:
                print(f" INITIAL ESTIMATES\n {bname}\n {binit}\n")
                print(f" ESTIMATION INDEX\n {bfit}\n")
                if ilmt > 0:
                    print(f" CONSTRAINTS ON PARAMETER VALUES\n {bmin}\n {bmax}\n")
            
            # PARAMETERS FOR BOUNDARY VALUE PROBLEMS (BLOCK D)
            infile.readline()
            infile.readline()
            modb= int(infile.readline().strip()) # Boundary condition code (0-6)

            if modb ==0: #zero input
                pulse = [{'conc': 0.0,'time': 0.0}]
            elif modb == 1: #dirac delta input
                tline = infile.readline().split()
                if mode <=2:
                    pulse = [{'conc': float(tline[0]),'time': 0.0}]
                else:
                    pulse = [{'conc': float(tline[0]),'time': 0.0}]
                    massst = int(tline[1])
                if mass == 1:
                    binit.append(pulse[0]['conc'])
                    bname.append('MASS')
                    bfit.append(int(infile.readline().strip()))
                    if ilmt == 1:
                        bmin.append(float(infile.readline().split()[0]))
                        bmax.append(float(infile.readline().split()[0]))
                    else:
                        bmin.append(0.0)
                        bmax.append(0.0)
            elif modb == 2: #step input
                inconc = float(infile.readline().strip())
                pulse= [{'conc': inconc,'time': 0.0}]
                if inverse == 1 and mass == 1:
                    binit.append(pulse[0]['conc'])
                    bname.append('Cin')
                    bfit.append(int(infile.readline().strip()))
                    if ilmt == 1:
                        bmin.append(float(infile.readline().split()[0]))
                        bmax.append(float(infile.readline().split()[0]))
                    else:
                        bmin.append(0.0)
                        bmax.append(0.0)
            elif modb == 3: #single pulse input
                tline = infile.readline().split()
                if mode <= 2:
                    pulse=[{'conc':float(tline[0]),'time':0.0},
                          {'conc':0.0,'time':float(tline[1])}]
                else:
                    pulse=[{'conc':float(tline[0]),'time':0.0},
                          {'conc':0.0, 'time':float(tline[1])}]
                    massst = int(tline[2])
                npulse = 2
                if inverse == 1 and mass == 1:
                    binit.append(pulse[0]['conc'])
                    binit.append(pulse[1]['time'])
                    bname.append('Cin')
                    if massst == 0:
                        bname.append('T2')
                    else:
                        bname.append('Mo')
                    tline = infile.readline().split()
                    bfit.append(int(tline[0]))
                    bfit.append(int(tline[1]))
                    if ilmt == 1:
                        tline = infile.readline().split()
                        bmin.append(float(tline[0]))
                        bmin.append(float(tline[1]))
                        tline = infile.readline().split()
                        bmax.append(float(tline[0]))
                        bmax.append(float(tline[1]))
                    else:
                        bmin.append(0.0)
                        bmin.append(0.0)
                        bmax.append(0.0)
                        bmax.append(0.0)
            elif modb == 4: #multiple pulse input
                mass=0
                npulse = int(infile.readline().strip())
                pulse = []
                for i in range(npulse):
                    tline = infile.readline().split()
                    pulse.append({'conc': float(tline[0]),'time': float(tline[1])})
            elif modb == 5: #exponential input
                mass=0
                tline = infile.readline().split()
                pulse = [{'conc':float(tline[0]),'time':float(tline[2])},
                          {'conc':float(tline[1]),'time':0.0}]
            elif modb == 6: #arbitrary input
                pulse = []
            else:
                raise ValueError('ERROR! MODB SHOULD BE 0,1,2,3,4,5,6')
            
            # Create DataFrame for parameters
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = bfit
            parms.loc['bmin'] = bmin
            parms.loc['bmax'] = bmax

            if verbose:
                text1 = {0: 'ZERO INPUT (SOLUTE FREE WATER)',
                         1: 'DIRAC DELTA INPUT',
                         2: 'STEP INPUT',
                         3: 'SINGLE PULSE INPUT',
                         4: 'MULTIPLE PULSE INPUT',
                         5: 'EXPONENTIAL INPUT',
                         6: 'ARBITRARY INPUT (DEFINE FUNCTION CINPUT(T))'}[modb]
                print(f"{text1} (MODB = {modb})\n BOUNDARY CONDITION")
                if modb != 0:
                    print(pulse)
            
            # PARAMETERS FOR INITIAL VALUE PROBLEMS (BLOCK E)
            infile.readline()
            infile.readline()
            modi= int(infile.readline().strip()) # Initial condition code (0-4)

            if modi == 0: #zero initial concentration
                cini=[{'conc':0.0,'z':0.0}]
            elif modi == 1: #constant initial concentration
                inconc = float(infile.readline().strip())
                cini = [{'conc':inconc,'z':0.0}]
            elif modi == 2: #stepwise initial concentration
                nini = int(infile.readline().strip())
                cini=[]
                for i in range(nini):
                    tline = infile.readline().split()
                    cini.append({'conc':float(tline[0]),'z':float(tline[1])})
            elif modi == 3: #exponential initial concentration
                tline = infile.readline().split()
                cini=[{'conc':float(tline[0]),'z':float(tline[2])},
                      {'conc':float(tline[1]),'z':0.0}]
            elif modi == 4: #Dirac delta(z-z1) & constant initial concentration
                tline = infile.readline().split()
                cini=[{'conc':float(tline[2]),'z':0.0},
                      {'conc':float(tline[0]),'z':float(tline[1])}]
            else:
                raise ValueError('ERROR! MODI SHOULD BE 0,1,2,3,4')
            
            if verbose:
                text1 = {0: 'ZERO INITIAL CONCENTRATION',
                         1: 'CONSTANT INITIAL CONCENTRATION',
                         2: 'STEPWISE INITIAL CONCENTRATION',
                         3: 'EXPONENTIAL INITIAL CONCENTRATION',
                         4: 'DELTA(Z-Z1) & CONSTANT INITIAL CONCENTRATION'}[modi]
                print(f" \n {text1} (MODI = {modi})\n INITIAL CONCENTRATION\n")
                if modi != 0:
                    print(cini)

            # PARAMETERS FOR PRODUCTION VALUE PROBLEMS (BLOCK F)
            infile.readline()
            infile.readline()
            modp = int(infile.readline().strip()) # Production code (0-3)

            mpro=0
            if modp == 0: #zero production
                prodval1 =[]
                prodval2 =[]
            elif modp == 1: #constant production
                if mode % 2 == 0:
                    mpro=1 
                    prodval1 = [{'gamma':float(infile.readline().strip()),'zpro':0.0}]
                    prodval2 = [{'gamma':float(infile.readline().strip()),'zpro':0.0}]
                else:
                    prodval1 = [{'gamma':float(infile.readline().strip()),'zpro':0.0}]
                    prodval2 = prodval1
            elif modp == 2: #stepwise production
                mpro = int(infile.readline().strip()) # Nonequilibrium Production code (0-1)
                npro1 = int(infile.readline().strip())
                prodval1 = [] 
                for i in range(npro1):
                    tline = infile.readline().split()
                    prodval1.append({'gamma':float(tline[0]),'zpro':float(tline[1])})
                if mode % 2 == 0 and mpro == 1:
                    npro2 = int(infile.readline().strip())
                    prodval2 = [] 
                    for i in range(npro2):
                        tline = infile.readline().split()
                        prodval2.append({'gamma':float(tline[0]),'zpro':float(tline[1])})
                else :
                    prodval2 = prodval1
            elif modp == 3: #exponential production
                tline = infile.readline().split()
                prodval1 = [{'gamma':float(tline[0]),'zpro':float(tline[2])},
                            {'gamma':float(tline[1]),'zpro':0.0}]
                if mode % 2 == 0 and mpro == 1:
                    tline = infile.readline().split()
                    prodval2 = [{'gamma':float(tline[0]),'zpro':float(tline[2])},
                                {'gamma':float(tline[1]),'zpro':0.0}]
                else:
                    prodval2 = prodval1
            else:
                raise ValueError('ERROR! MODP SHOULD BE 0,1,2,3')
            if verbose:
                text1 = {0: 'ZERO PRODUCTION',
                         1: 'CONSTANT PRODUCTION',
                         2: 'STEPWISE PRODUCTION',
                         3: 'EXPONENTIAL PRODUCTION'}[modp]
                print(f" {text1} (MODP = {modp})\n PRODUCTION VALUES\n")
                if modp != 0:
                    print(prodval1)
                    print(prodval2)
            
            # READ DATA FOR INVERSE PROBLEM (BLOCK G)
            if inverse ==1 :
                infile.readline()
                infile.readline()
                inputm = int(infile.readline().strip()) # Input mode (0-3)
                # Position of the breakthrough curve.
                if inputm == 1 or inputm == 2:
                    dumtz = float(infile.readline().strip()) 
                infile.readline()
                obsdata = [] # Observation data
                nob = 0
                while True:
                    tline = infile.readline().split()
                    # AT THE END OF DATA SET, GIVE "0,0,0" TO INDICATE THE LAST LINE
                    if all(abs(float(x)) < 1e-7 for x in tline) and nob >=1:
                        break
                    if inputm == 0: #Z(I), T(I), C(I)
                        if mit >=1:
                            obsdata.append({'t':float(tline[1]),'z':float(tline[0]),'cobs':float(tline[2])})
                        else:
                            obsdata.append({'t':float(tline[1]),'z':float(tline[0]),'cobs':0.0})
                    elif inputm == 1: #T(I), C(I) FOR SAME Z (Breakthrough Curve)
                        if mit >=1 :
                            obsdata.append({'t':float(tline[0]),'z':dumtz,'cobs':float(tline[1])})
                        else:
                            obsdata.append({'t':float(tline[0]),'z':dumtz,'cobs':0.0})
                    elif inputm == 2: #Z(I), C(I) FOR SAME T (Depth Profile)
                        if mit >=1:
                            obsdata.append({'t':dumtz,'z':float(tline[0]),'cobs':float(tline[1])})
                        else:
                            obsdata.append({'t':dumtz,'z':float(tline[0]),'cobs':0.0})
                    elif inputm == 3: #C(I), Z(I), T(I)
                            obsdata.append({'t':float(tline[2]),'z':float(tline[1]),'cobs':float(tline[0])})
                    else :
                        raise ValueError('ERROR! INPUTM SHOULD BE 0,1,2,3')
                    nob += 1
            # T & Z FOR DIRECT PROBLEM (BLOCK H)
            else:
                infile.readline()
                infile.readline()
                tline = infile.readline().split()
                nz=int(tline[0])
                dz=float(tline[1])
                zi=float(tline[2])
                nt=int(tline[3])
                dt=float(tline[4])
                ti=float(tline[5])
                mprint=int(tline[6])
                zi_values = [zi + i * dz for i in range(nz)]
                ti_values = [ti + i * dt for i in range(nt)]
                obsdata = [{'t': t, 'z': z} 
                           for z in zi_values for t in ti_values]
            
            obsdata = pd.DataFrame(obsdata)

            # EQUILIBRIUM CDE WITH NREDU=1, ZL=MAX OF ZOBS
            if mode % 2 == 1 and nredu <= 1:
                zl = obsdata['z'].max()

            if verbose:
                print(obsdata)
                print(f" CHARACTERISTIC LENGTH (ZL = {zl})\n")

            simcases.append(CXTsim(title=title,
                                    inverse=inverse,
                                    mode=mode,
                                    nredu=nredu,
                                    modc=modc,
                                    zl=zl,
                                    mit=mit,
                                    ilmt=ilmt,
                                    mass=mass,
                                    massst=massst,
                                    mneq=mneq,
                                    mdeg=mdeg,
                                    phim=phim,
                                    rhoth=rhoth,
                                    parms=parms,
                                    modb=modb,
                                    pulse=pulse,
                                    modi=modi,
                                    cini=cini,
                                    modp=modp,
                                    mpro=mpro,
                                    prodval1=prodval1,
                                    prodval2=prodval2,
                                    obsdata=obsdata,
                                    mprint=mprint,
                                    verbose=verbose
                                    )) 
        
        if hasattr(f, "read"):
            f.close()

        return cls(simcases)
    
    def write(self, f: str, verbose=False):
        """
        Write data to a cxtfit input file
        Parameters
        ----------
        f : str
            Path to cxtfit input file to write.
        """
        with open(f, "w") as outfile:
            outfile.write(f"{len(self.simcases)}\n")
            for i, simcase in enumerate(self.simcases):
                if verbose:
                    print(f"\n\n{'-' * 100}\nWriting Simulation Case {i+1}\n\n{simcase.title}\n{'-' * 100}\n")
                
                outfile.write("\n")
                outfile.write(f"{simcase.title}\n")
                outfile.write("\n\n")
                outfile.write(f"{simcase.inverse} {simcase.mode} {simcase.nredu}\n")
                outfile.write("\n")
                if simcase.mode % 2 == 1 and simcase.nredu <= 1:
                    outfile.write(f"{simcase.modc}\n")
                else:
                    outfile.write(f"{simcase.modc} {simcase.zl}\n")
                
                if simcase.inverse >= 1:
                    outfile.write("\n\n")
                    outfile.write(f"{simcase.mit} {simcase.ilmt} {simcase.mass}\n")
                    if simcase.mode % 2 == 0:
                        outfile.write("\n")
                        outfile.write(f"{simcase.mneq} {simcase.mdeg}\n")
                        if simcase.ilmt == 0 and simcase.mneq == 3:
                            raise ValueError('ERROR! ILMT SHOULD BE 1 FOR MNEQ = 3')
                        if simcase.mneq == 3 or (simcase.mneq == 0 and simcase.mdeg >= 2):
                            outfile.write("\n")
                            outfile.write(f"{simcase.phim}\n")
                
                outfile.write("\n\n")
                ntp, bname = {
                    1: (4, ['V', 'D', 'R', 'mu1']),
                    2: (7, ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']),
                    3: (8, ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']),
                    4: (10, ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']),
                    5: (8, ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovD']),
                    6: (10, ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovD']),
                    8: (11, ['<V>', '<D>', '<Kd>', 'alpha', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'SD.alp', 'RhovAl'])
                }[simcase.mode]
                
                binit = simcase.parms.loc['binit'].values
                outfile.write(" ".join(map(str, binit)) + "\n")
                
                if simcase.inverse >= 1:
                    bfit = simcase.parms.loc['bfit'].values
                    outfile.write(" ".join(map(str, bfit)) + "\n")
                    if simcase.ilmt > 0:
                        bmin = simcase.parms.loc['bmin'].values
                        bmax = simcase.parms.loc['bmax'].values
                        outfile.write(" ".join(map(str, bmin)) + "\n")
                        outfile.write(" ".join(map(str, bmax)) + "\n")
                
                outfile.write("\n\n")
                outfile.write(f"{simcase.modb}\n")
                if simcase.modb == 0:
                    pass
                elif simcase.modb == 1:
                    outfile.write(f"{simcase.pulse[0]['conc']}\n")
                    if simcase.mode > 2:
                        outfile.write(f"{simcase.massst}\n")
                    if simcase.mass == 1:
                        outfile.write(f"{simcase.parms['MASS']['binit']}\n")
                        outfile.write(f"{simcase.parms['MASS']['bfit']}\n")
                        if simcase.ilmt == 1:
                            outfile.write(f"{simcase.parms['MASS']['bmin']}\n")
                            outfile.write(f"{simcase.parms['MASS']['bmax']}\n")
                elif simcase.modb == 2:
                    outfile.write(f"{simcase.pulse[0]['conc']}\n")
                    if simcase.inverse == 1 and simcase.mass == 1:
                        outfile.write(f"{simcase.parms['Cin']['binit']}\n")
                        outfile.write(f"{simcase.parms['Cin']['bfit']}\n")
                        if simcase.ilmt == 1:
                            outfile.write(f"{simcase.parms['Cin']['bmin']}\n")
                            outfile.write(f"{simcase.parms['Cin']['bmax']}\n")
                elif simcase.modb == 3:
                    outfile.write(f"{simcase.pulse[0]['conc']} {simcase.pulse[1]['time']}\n")
                    if simcase.mode > 2:
                        outfile.write(f"{simcase.massst}\n")
                    if simcase.inverse == 1 and simcase.mass == 1:
                        outfile.write(f"{simcase.parms['Cin']['binit']} {simcase.parms['T2']['binit']}\n")
                        outfile.write(f"{simcase.parms['Cin']['bfit']} {simcase.parms['T2']['bfit']}\n")
                        if simcase.ilmt == 1:
                            outfile.write(f"{simcase.parms['Cin']['bmin']} {simcase.parms['T2']['bmin']}\n")
                            outfile.write(f"{simcase.parms['Cin']['bmax']} {simcase.parms['T2']['bmax']}\n")
                elif simcase.modb == 4:
                    outfile.write(f"{len(simcase.pulse)}\n")
                    for pulse in simcase.pulse:
                        outfile.write(f"{pulse['conc']} {pulse['time']}\n")
                elif simcase.modb == 5:
                    outfile.write(f"{simcase.pulse[0]['conc']} {simcase.pulse[1]['conc']} {simcase.pulse[0]['time']}\n")
                elif simcase.modb == 6:
                    pass
                else:
                    raise ValueError('ERROR! MODB SHOULD BE 0,1,2,3,4,5,6')
                
                outfile.write("\n\n")
                outfile.write(f"{simcase.modi}\n")
                if simcase.modi == 0:
                    pass
                elif simcase.modi == 1:
                    outfile.write(f"{simcase.cini[0]['conc']}\n")
                elif simcase.modi == 2:
                    outfile.write(f"{len(simcase.cini)}\n")
                    for cini in simcase.cini:
                        outfile.write(f"{cini['conc']} {cini['z']}\n")
                elif simcase.modi == 3:
                    outfile.write(f"{simcase.cini[0]['conc']} {simcase.cini[1]['conc']} {simcase.cini[0]['z']}\n")
                elif simcase.modi == 4:
                    outfile.write(f"{simcase.cini[1]['conc']} {simcase.cini[1]['z']} {simcase.cini[0]['conc']}\n")
                else:
                    raise ValueError('ERROR! MODI SHOULD BE 0,1,2,3,4')
                
                outfile.write("\n\n")
                outfile.write(f"{simcase.modp}\n")
                if simcase.modp == 0:
                    pass
                elif simcase.modp == 1:
                    if simcase.mode % 2 == 0:
                        outfile.write(f"{simcase.prodval1[0]['gamma']}\n")
                        outfile.write(f"{simcase.prodval2[0]['gamma']}\n")
                    else:
                        outfile.write(f"{simcase.prodval1[0]['gamma']}\n")
                elif simcase.modp == 2:
                    outfile.write(f"{simcase.mpro}\n")
                    outfile.write(f"{len(simcase.prodval1)}\n")
                    for prodval in simcase.prodval1:
                        outfile.write(f"{prodval['gamma']} {prodval['zpro']}\n")
                    if simcase.mode % 2 == 0 and simcase.mpro == 1:
                        outfile.write(f"{len(simcase.prodval2)}\n")
                        for prodval in simcase.prodval2:
                            outfile.write(f"{prodval['gamma']} {prodval['zpro']}\n")
                elif simcase.modp == 3:
                    outfile.write(f"{simcase.prodval1[0]['gamma']} {simcase.prodval1[1]['gamma']} {simcase.prodval1[0]['zpro']}\n")
                    if simcase.mode % 2 == 0 and simcase.mpro == 1:
                        outfile.write(f"{simcase.prodval2[0]['gamma']} {simcase.prodval2[1]['gamma']} {simcase.prodval2[0]['zpro']}\n")
                else:
                    raise ValueError('ERROR! MODP SHOULD BE 0,1,2,3')
                
                if simcase.inverse == 1:
                    outfile.write("\n\n")
                    outfile.write(f"{simcase.inputm}\n")
                    if simcase.inputm == 1 or simcase.inputm == 2:
                        outfile.write(f"{simcase.dumtz}\n")
                    outfile.write("\n")
                    for _, row in simcase.obsdata.iterrows():
                        if simcase.inputm == 0:
                            outfile.write(f"{row['z']} {row['t']} {row['cobs']}\n")
                        elif simcase.inputm == 1:
                            outfile.write(f"{row['t']} {row['cobs']}\n")
                        elif simcase.inputm == 2:
                            outfile.write(f"{row['z']} {row['cobs']}\n")
                        elif simcase.inputm == 3:
                            outfile.write(f"{row['cobs']} {row['z']} {row['t']}\n")
                        else:
                            raise ValueError('ERROR! INPUTM SHOULD BE 0,1,2,3')
                else:
                    outfile.write("\n\n")
                    outfile.write(f"{len(simcase.obsdata['z'].unique())} {simcase.dz} {simcase.zi} {len(simcase.obsdata['t'].unique())} {simcase.dt} {simcase.ti} {simcase.mprint}\n")
                    for _, row in simcase.obsdata.iterrows():
                        outfile.write(f"{row['z']} {row['t']}\n")