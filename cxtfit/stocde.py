import numpy as np
from .detcde import DetCDE,dbexp
from scipy.integrate import quad

# Converted from functions in STOCDE.FOR original Fortran code
class StoCDE(DetCDE):
    def StoCDE(self):
        mods = self.mode % 2

        # MCON=0 CALUCULATE PHASE1 & PHASE2 FOR TOTAL RESIDENT CONC.
        # MCON=1            PHASE1 ONLY
        # MCON=2            PHASE2 ONLY
        if self.modc in [4, 6]:
            self.mcon = 0
        else:
            self.mcon = 1

        # MSD;  Calculation control code for ensemble averages 
        # MSD=0  <C>
        # MSD=1  <C*C>
        self.msd = 0
        if self.mstoch in [1, 3]:
            if self.intm == 2:
                c1 = self.chebylog2(self.conprov, self.vmin, self.vmax)
                if self.inverse == 0:
                    self.msd = 1
                    c12 = self.chebylog2(self.conprov, self.vmin, self.vmax)
            else:
                c1 , _ = quad(self.conprov, self.vmin, self.vmax)
                if self.inverse == 0:
                    self.msd = 1
                    c12, _ = quad(self.conprov, self.vmin, self.vmax)

            if self.modc in [4, 6]:
                return c1, 0.0, c12-c1*c1, 0.0

            if mods == 0:
                self.mcon = 2
                if self.intm == 2:
                    c2 = self.chebylog2(self.conprov, self.vmin, self.vmax)
                    if self.inverse == 0:
                        self.msd = 1
                        c22 = self.chebylog2(self.conprov, self.vmin, self.vmax)
                else:
                    c2 = self.romb(self.conprov, self.vmin, self.vmax)
                    if self.inverse == 0:
                        self.msd = 1
                        c22 , _ = quad(self.conprov, self.vmin, self.vmax)

        if self.mstoch in [2, 4]:
            if self.intm == 2:
                c1 = self.chebylog2(self.conproy, self.ymin, self.ymax)
                if self.inverse == 0:
                    self.msd = 1
                    c12 = self.chebylog2(self.conproy, self.ymin, self.ymax)
            else:
                c1 = self.romb2(self.conproy, self.ymin, self.ymax)
                if self.inverse == 0:
                    self.msd = 1
                    c12 = self.romb2(self.conproy, self.ymin, self.ymax)

            if self.modc in [4, 6]:
                return c1, c2, c12-c1*c1, c22-c2*c2

            if mods == 0:
                self.mcon = 2
                if self.intm == 2:
                    c2 = self.chebylog2(self.conproy, self.ymin, self.ymax)
                    if self.inverse == 0:
                        self.msd = 1
                        c22 = self.chebylog2(self.conproy, self.ymin, self.ymax)
                else:
                    c2 = self.romb2(self.conproy, self.ymin, self.ymax)
                    if self.inverse == 0:
                        self.msd = 1
                        c22 = self.romb2(self.conproy, self.ymin, self.ymax)

        return c1, c2, c12-c1*c1, c22-c2*c2
    
    def conprov(self, v):
        """Calculate argument of stochastic models for a log-normal velocity distribution"""

        mmode = self.mode
        sd = self.d
        sp = self.p
        sr = self.r
        sdk = self.dk
        sbeta = self.beta
        sbetr = self.betar
        salpha = self.alpha if self.mode == 8 else None
        somega = self.omega
        sdmu1 = self.dmu1
        sdmu2 = self.dmu2
        spulse = self.cpulse[0]

        stpulse = self.tpulse
        sgamma1 = self.gamma1
        sgamma2 = self.gamma2

        if abs(v) < 1.0e-30:
            v = self.v * 1.0e-30

        if self.mode in [5, 6]:
            m56 = 1
            self.sdln_d = self.sdlny
        else:
            m56 = 0

        if (m56 == 1 and self.mstoch != 3) or (m56 == 0 and self.modd == 0):
            self.p = self.p / self.v * v
        elif (m56 == 1 and self.mcorr == 1) or (m56 == 0 and self.modd >= 1):
            self.d = (v / self.v) ** (self.sdln_d / self.sdlnv) * sd * dbexp(0.5 * (self.sdlnv - self.sdln_d) * self.sdln_d)
            self.p = self.p / self.v * v * sd / self.d
        elif (m56 == 1 and self.mcorr == -1) or (m56 == 0 and self.modd <= -1):
            self.d = (v / self.v) ** (-self.sdln_d / self.sdlnv) * sd * dbexp(0.5 * (-self.sdlnv - self.sdln_d) * self.sdln_d)
            self.p = self.p / self.v * v * sd / self.d

        if self.mode in [3, 4]:
            m34 = 1
            self.sdlnk = self.sdlny
        else:
            m34 = 0

        if (m34 == 1 and self.mcorr == 1) or (m34 == 0 and self.modk >= 1):
            self.dk = (v / self.v) ** (self.sdlnk / self.sdlnv) * sdk * dbexp(0.5 * (self.sdlnv - self.sdlnk) * self.sdlnk)
            self.r = 1.0 + self.rhoth * self.dk
        elif (m34 == 1 and self.mcorr == -1) or (m34 == 0 and self.modk <= -1):
            self.dk = (v / self.v) ** (-self.sdlnk / self.sdlnv) * sdk * dbexp(0.5 * (-self.sdlnv - self.sdlnk) * self.sdlnk)
            self.r = 1.0 + self.rhoth * self.dk

        if self.mneq == 1:
            self.dmu2 = self.dmu2 / (sr - 1.0) * (self.r - 1.0)
            self.beta = 1.0 / self.r
            self.omega = self.omega / (sr - 1.0) * (self.r - 1.0)

        if self.mode in [3, 5]:
            self.dmu1 = self.dmu1 / sr * self.r

        if self.mode == 8 and self.mstoch == 3:
            if self.mcorr == 1:
                self.alpha = (v / self.v) ** (self.sdlny / self.sdlnv) * salpha * dbexp(0.5 * (self.sdlnv - self.sdlny) * self.sdlny)
            elif self.mcorr == -1:
                self.alpha = (v / self.v) ** (-self.sdlny / self.sdlnv) * salpha * dbexp(0.5 * (-self.sdlnv - self.sdlny) * self.sdlny)
            self.omega = somega / salpha * self.alpha

        self.omega = self.omega / v * self.v
        self.dmu1 = self.dmu1 / v * self.v
        self.dmu2 = self.dmu2 / v * self.v

        for i in range(self.npulse):
            self.tpulse[i] = self.tpulse[i] / self.v * v
        for i in range(self.npro1):
            self.gamma1[i] = self.gamma1[i] / v * self.v
        for i in range(self.npro2):
            self.gamma2[i] = self.gamma2[i] / v * self.v

        if self.modb == 1 and self.massst != 1:
            self.cpulse[0] = self.cpulse[0] / self.v * v
        if self.modb == 3 and self.massst == 1:
            self.tpulse[1] = stpulse[1]

        c1, c2 = self.DetCDE()

        if self.mcon == 1:
            c = c1
        elif self.mcon == 0:
            c = c1 * self.beta * self.r + c2 * (1.0 - self.beta) * self.r
        elif self.mcon == 2:
            if self.nredu == 0:
                c = c2 * self.dk
            else:
                c = c2

        if self.mstoch == 4 and self.mcorr != 0:
            if self.mode in [3, 4]:
                prob = self.blnprob(v, self.v, self.sdlnv, sdk, self.avey, self.sdlny, self.corr)
            elif self.mode in [5, 6]:
                prob = self.blnprob(v, self.v, self.sdlnv, sd, self.avey, self.sdlny, self.corr)
            elif self.mode == 8:
                prob = self.blnprob(v, self.v, self.sdlnv, salpha, self.avey, self.sdlny, self.corr)
        else:
            prob = self.xlnprob(v, self.v, self.sdlnv)

        if self.msd == 0:
            if self.modc == 2:
                conprov = prob * c * v / self.v
            else:
                conprov = prob * c
        else:
            if self.modc == 2:
                conprov = prob * c * c * v * v / self.v / self.v
            else:
                conprov = prob * c * c

        self.mode = mmode
        self.p = sp
        self.d = sd
        if self.mode == 8:
            self.alpha = salpha
        self.r = sr
        self.dk = sdk
        self.beta = sbeta
        self.betar = sbetr
        self.omega = somega
        self.dmu1 = sdmu1
        self.dmu2 = sdmu2

        return conprov
        
    def conproy(self, y):
        """Calculate argument of stochastic models for a log-normal distribution of another parameter (v, D, K, alpha)"""
        mmode = self.mode
        sp = self.p
        sdk = self.dk
        sdd = self.d
        salpha = self.alpha
        sr = self.r
        sbeta = self.beta
        sbetr = self.betar
        somega = self.omega
        sdmu1 = self.dmu1
        sdmu2 = self.dmu2

        if self.mode in [3, 4]:
            self.avey = self.dk
            self.dk = y
        if self.mode in [5, 6]:
            self.avey = self.d
            self.d = y
        if self.mode == 8:
            self.avey = self.alpha
            self.alpha = y

        if self.mode in [3, 4]:
            self.r = 1 + self.rhoth * self.dk
            if self.mneq == 1:
                self.dmu2 = self.dmu2 / (sr - 1.0) * (self.r - 1.0)
                self.beta = 1.0 / self.r
                self.omega = self.omega / (sr - 1.0) * (self.r - 1.0)
            if self.mode == 3:
                self.dmu1 = self.dmu1 / sr * self.r
        if self.mode in [5, 6]:
            self.p = self.p * sdd / self.d
        if self.mode == 8:
            self.omega = self.omega / salpha * self.alpha

        prob = self.xlnprob(y, self.avey, self.sdlny)

        if self.mstoch == 2:
            c1, c2 = self.detcde()
            if self.mcon == 1:
                c = c1
            elif self.mcon == 0:
                c = c1 * self.beta * self.r + c2 * (1.0 - self.beta) * self.r
            elif self.mcon == 2:
                if self.nredu == 0:
                    c = c2 * self.dk
                else:
                    c = c2

            if self.msd == 0:
                conproy = prob * c
            else:
                conproy = prob * c * c

        if self.mstoch == 4:
            c = self.romb(self.conprov, 0.0, self.vmin, self.vmax)
            if self.mcorr == 0:
                conproy = c * prob
            elif self.mcorr == 2:
                conproy = c

        self.mode = mmode
        self.p = sp
        self.dk = sdk
        self.d = sdd
        self.alpha = salpha
        self.r = sr
        self.beta = sbeta
        self.betar = self.beta * self.r
        self.omega = somega
        self.dmu1 = sdmu1
        self.dmu2 = sdmu2

        return conproy
    
    def limit(self, x, sdlnx):
        """
        Calculate integration limits for a log-normal distribution
        for stochastic models using Newton-Raphson method.
        """
        xlnm = np.log(x) - 0.5 * sdlnx * sdlnx
        xmod = dbexp(xlnm - sdlnx * sdlnx)
        xmin1 = max(1.0e-5, xmod - xmod * sdlnx)
        xmax1 = xmod + xmod * sdlnx
        cmin = 1.0e-7

        # Calculate XMIN
        for _ in range(self.level):
            clst = self.xlnprob(xmin1, x, sdlnx)
            if clst < cmin:
                break
            slope = -clst / (2.0 * sdlnx * sdlnx * xmin1) * (np.log(xmin1) - xlnm + sdlnx * sdlnx)
            xmin1 -= clst / slope
            if xmin1 <= 0.0:
                xmin1 = 1.0e-30
                break
        else:
            print("WARNING ! OUT OF RANGE (LOWER LIMIT)")

        # Calculate XMAX
        for _ in range(self.level):
            clst = self.xlnprob(xmax1, x, sdlnx)
            if clst < cmin:
                break
            slope = -clst / (2.0 * sdlnx * sdlnx * xmax1) * (np.log(xmax1) - xlnm + sdlnx * sdlnx)
            xmax1 -= clst / slope
        else:
            print("WARNING ! OUT OF RANGE FOR (UPPER LIMIT)")

        return xmin1, xmax1

    def xlnprob(self, x, avex, sdlnx):
        """
        Single log-normal distribution.
        """
        dpi = np.sqrt(2 * np.pi)
        xlnm = np.log(avex) - 0.5 * sdlnx * sdlnx
        xln = np.log(x)
        arg = (xln - xlnm) * (xln - xlnm) / (2.0 * sdlnx * sdlnx)
        xlnprob = dbexp(-arg) / (dpi * sdlnx * x)
        return xlnprob

    def blnprob(self, x, avex, sdlnx, y, avey, sdlny, rho):
        """
        Bivariate log-normal distribution.
        """
        xlnm = np.log(avex) - 0.5 * sdlnx * sdlnx
        ylnm = np.log(avey) - 0.5 * sdlny * sdlny
        x1 = (np.log(x) - xlnm) / sdlnx
        y1 = (np.log(y) - ylnm) / sdlny
        arg1 = (x1 * x1 - 2 * rho * x1 * y1 + y1 * y1) / (2.0 * (1.0 - rho * rho))
        arg2 = 2.0 * np.pi * sdlnx * sdlny * x * y * np.sqrt(1.0 - rho * rho)
        blnprob = dbexp(-arg1) / arg2
        return blnprob

    def init_stocde(self,simparms):
        """
        Initialize stochastic model parameters before computation
        """
        # Velocity for dimensionless variables, value may change for each iteration
        mods = self.mode % 2
        self.v = simparms['<V>'] # VELOCITY

        if self.modb == 3 and self.mass == 1:
            self.tpulse[1] = simparms['T2']

        if self.nredu in (0,1):
            # CHANGE pulse TO DIMENSIONLESS VARIABLES
            if self.modb in (3,4):
                for i in range(self.npulse):
                    self.tpulse[i] = self.tpulse[i] * self.v / self.zl
            if (self.modb == 5):
                self.tpulse[0] = self.tpulse[0] / self.v * self.zl

            # CHANGE gamma TO DIMENSIONLESS VARIABLES FOR MODE=1,3,5
            if mods == 1 and self.modp in (1,2) :
                for i in range(self.npro1) :
                    self.gamma1[i] = self.gamma1[i] * self.zl / self.v
                    # self.zpro1[i] = self.zpro1[i] / self.zl

        # PARAMETER FOR TOTAL MASS
        if self.mass == 1:
            if self.modb ==2:
                self.cpulse[0] = simparms['MASS']
            if self.modb in (1,3):
                self.cpulse[0] = simparms['Cin']

        self.d = simparms['<D>']# DISPERSIVITY
        self.p = self.v * self.zl / self.d
        self.dk = simparms['<Kd>'] # DISTRIBUTION COEFFICIENT
        self.r = 1.0 + self.rhoth * self.dk # RETARDATION FACTOR
        self.dmu1 = simparms['mu1']# DEGRADATION RATE
        self.beta = 1.0
        self.omega = 0.0
        self.dmu2 = 0.0
        if self.nredu <= 1:
            self.dmu1 *= self.zl / self.v
        self.sdlnv = simparms['SD.v']
        
        if self.mode == 3:
            self.y = self.dk
            self.sdlny = simparms['SD.Kd']
            self.betar = self.r
            if self.mk34 == 1 and self.mit >= 1:
                self.sdlny = self.sdlnv
                simparms['SD.Kd']= self.sdlny
            if self.modd == 2 and self.mit >= 1:
                self.sdln_d = self.sdlnv
                simparms['SD.D']= self.sdln_d
            else:
                self.sdln_d = simparms['SD.D']
            self.corr = simparms['RhovKd']
        elif self.mode == 4:
            self.y = self.dk
            self.sdlny = simparms['SD.Kd']
            self.beta = 1.0 / self.r
            self.betar = 1.0
            self.omega = simparms['omega']
            self.dmu2 = simparms['mu2']
            if self.mit >= 1:
                if self.mdeg == 1:
                    self.dmu2 = (self.r - 1.0) * self.dmu1
                    simparms['mu2']= self.dmu2
                elif self.mdeg == 2:
                    self.dmu2 = 0
                    simparms['mu2']= self.dmu2
                elif self.mdeg == 3:
                    self.dmu1 = 0.0
                    simparms['mu1']= self.dmu1
            if self.mk34 == 1 and self.mit >= 1:
                self.sdlny = self.sdlnv
                simparms['SD.Kd']= self.sdlny
            if self.modd == 2 and self.mit >= 1:
                self.sdln_d = self.sdlnv
                simparms['SD.D']= self.sdln_d
            else:
                self.sdln_d = simparms['SD.D']
            self.corr = simparms['RhovKd']
        elif self.mode == 5:
            self.y = self.d
            self.sdlny = simparms['SD.D']
            self.betar = self.r
            if self.md56 == 1 and self.mit >= 1:
                self.sdlny = self.sdlnv
                simparms['SD.D']= self.sdlny
            if self.modk == -2 and self.mit >= 1:
                self.sdlnk = self.sdlnv
                simparms['SD.Kd']= self.sdlnk
            else:
                self.sdlnk = simparms['SD.Kd']
            self.corr = simparms['RhovD']
        elif self.mode == 6:
            self.y = self.d
            self.sdlny = simparms['SD.D']
            self.beta = 1.0 / self.r
            self.betar = 1.0
            self.omega = simparms['omega']
            self.dmu2 = simparms['mu2']
            if self.mit >= 1:
                if self.mdeg == 1:
                    self.dmu2 = (self.r - 1.0) * self.dmu1
                    simparms['mu2']= self.dmu2
                elif self.mdeg == 2:
                    self.dmu2 = 0
                    simparms['mu2']= self.dmu2
                elif self.mdeg == 3:
                    self.dmu1 = 0.0
                    simparms['mu1']= self.dmu1

            if self.md56 == 1 and self.mit >= 1:
                self.sdlny = self.sdlnv
                simparms['SD.D']= self.sdlny

            if self.modd == -2 and self.mit >= 1:
                self.sdlnk = self.sdlnv
                simparms['SD.Kd']= self.sdlnk
            else:
                self.sdlnk = simparms['SD.Kd']

            self.corr = simparms['RhovD']
        elif self.mode == 8:
            self.alpha = simparms['alpha']
            self.y = self.alpha
            self.omega = self.alpha * (self.r - 1.0) * self.zl / self.v
            self.beta = 1.0 / self.r
            self.betar = 1.0
            self.dmu2 = simparms['mu2']
            if self.mit >= 1:
                if self.mdeg == 1:
                    self.dmu2 = (self.r - 1.0) * self.dmu1
                    simparms['mu2']= self.dmu2
                elif self.mdeg == 2:
                    self.dmu2 = 0
                    simparms['mu2']= self.dmu2
                elif self.mdeg == 3:
                    self.dmu1 = 0.0
                    simparms['mu1']= self.dmu1

            if self.mal8 == 1 and self.mit >= 1:
                self.sdlny = self.sdlnv
                simparms['SD.alp']= self.sdlny
            else:
                self.sdlny = simparms['SD.alp']
            
            if self.modd == -2 and self.mit >= 1:
                self.sdlnk = self.sdlnv
                simparms['SD.Kd']= self.sdlnk
            else:
                self.sdlnk = simparms['SD.Kd']

            self.corr = simparms['RhovAl']
        
        # MSTOCH;    Index for stochastic v and Y
        #   1: V; VARIABLE Y;CONSTANT
        #   2: V; CONSTANT Y;VARIABLE
        #   3: POSITIVE OR NEGATIVE CORRELATION BETWEEN  V & Y 
        #   4: V; VARIABLE Y;VARIABLE
        # ISKIP   Calculation control code for the 
        #         evaluation of the integral limits
        #   0: CALCULATE INTEGRAL LIMITS
        #   1: SKIP THIS PART.
        # MCORR=1 POSITIVE CORRELATION V&Y.
        # MCORR=-1 NEGATIVE CORRELATION V&Y.
        # MCORR=0 NO CORRELATION V&Y.
        # MCORR=2 BIVARIATE DISTRIBUTION V&Y.

        if self.iskip == 1:
            return
        elif self.sdlny < 1.0e-7:
            self.mstoch = 1
            self.corr = 0.0
        elif self.sdlnv < 1.0e-7:
            self.mstoch = 2
            self.corr = 0.0
        else:
            self.mstoch = 4

        self.mcorr = 0
        if self.mstoch == 4:
            if self.corr < -0.999999:
                self.mcorr = -1
                self.mstoch = 3
            elif self.corr > 0.999999:
                self.mcorr = 1
                self.mstoch = 3
            elif abs(self.corr) < 1.0e-20:
                self.mcorr = 0
            else:
                self.mcorr = 2

        self.const2()
        
        if self.mstoch != 2:
            x = self.v
            self.sdlnx = self.sdlnv
            self.vmin, self.vmax = self.limit(x, self.sdlnx)

        if self.mstoch % 2 == 0:
            x = self.y
            self.sdlnx = self.sdlny
            self.ymin, self.ymax = self.limit(x, self.sdlnx)

        return

    # Converted from CONST2 function in USER.FOR Fortran 
    def const2(self):
        self.stoper = 1.0e-9
        self.level = 12
        if (self.mode % 2) == 1:
            if self.p <= 1.0e+3:
                self.stoper = 5.0e-7
                self.level = 11
            elif self.p >= 1.0e+5:
                self.stoper = 1.0e-10
                self.level = 15
            elif (self.p >= 1.0e+3) and (self.p <= 5.0e+5):
                self.stoper = 5.0e-8
                self.level = 12
        if ((self.mode % 2) == 0) and (self.mstoch == 4):
            self.stoper = 5.0e-5
            self.level = 9
            self.icheb = 0
    
    def update_stomode(self):
        """
        Converted from DATA.FOR lines 935-1110.
        Updates and returns the stochastic model flags and
        modifies the passed arrays (b, index).
        """

        # Initialize parameters
        self.modd = self.modk = self.md56 = self.mk34 = self.mal8 = 0

        # ---- DISPERSION COEFFICIENT FOR STOCHASTIC MODEL FOR MODE=3,4,8 ----
        # MODD; Index for a stochastic dispersion coefficient 
        # 2  SDLND=SDLNV (FIELD SCALE DISPERSIVITY=CONSTANT)
        # 1 POSITIVE CORRELATION V&D.
        # 0  D=CONSTANT
        # -1 NEGATIVE CORRELATION V&D.

        if self.mode in (3,4,8) :
            if self.parms.loc['binit','SD.D'] < 1.0e-7:
                self.modd = 0
            elif abs(self.parms.loc['binit','SD.v']- self.parms.loc['binit','SD.D']) < 1.0e-5:
                self.modd= 2
            else:
                self.modd= 1

        # ---- DISPERSION COEFFICIENT FOR STOCHASTIC MODEL FOR MODE=5,6 ----
        # MD56; Index for a stochastic dispersion coefficient for MODE=5,6 
        #   MD56=1  SDLND=SDLNV 
        #   MD56=0  SDLND AND SALNV ARE INDEPENSENT.
        #  WHEN MODD=0, NO ESTIMATE FOR SDLNV (SIG.V)
        if self.mode in (5,6):
            if abs(self.parms.loc['binit','SD.v']- self.parms.loc['binit','SD.D']) < 1.0e-5:
                self.md56 = 1

        if self.mit >= 1:
            if self.mode in (3,4,8):
                if self.modd== 0:
                    self.parms.loc['bfit','SD.D'] = 0
                if (self.parms.loc['bfit','SD.D'] == self.parms.loc['bfit','SD.v']) and self.modd== 2:
                    self.parms.loc['bfit','SD.D'] = 0

            if self.mode in (5,6):
                if self.parms.loc['bfit','SD.D'] == self.parms.loc['bfit','SD.v'] and self.md56 == 1:
                    self.parms.loc['bfit','SD.D'] = 0
                elif self.parms.loc['bfit','SD.D'] * self.parms.loc['bfit','SD.v'] == 0:
                    self.md56 = 0

        # ---- DISTRIBUTION COEFFICIENT FOR STOCHASTIC MODEL MODE=5,6,8 ----
        #    Assume negative correlation V & Kd for self.mode 5,6,8
        #    MODK=1 positive correlation, MODK=0 constant, MODK=-1 negative,
        #    MODK=-2 means SDLNK=SDLNV
        if self.mode in (5,6,8):
            if self.parms.loc['binit','SD.Kd']< 1.0e-7:
                modk = 0
            elif abs(self.parms.loc['binit','SD.v']- self.parms.loc['binit','SD.Kd']) < 1.0e-5:
                modk = -2
            else:
                modk = -1

        # ---- DISTRIBUTION COEFFICIENT FOR MODE=3,4 => MK34 ----
        #    MK34=1 if abs(SDLNK - SDLNV)<1e-05, else 0
        if self.mode in (3,4):
            if abs(self.parms.loc['binit','SD.v']- self.parms.loc['binit','SD.Kd']) < 1.0e-5:
                mk34 = 1

        # ---- If MIT >=1, update INDEX array based on MK34 ----
        if self.mit >= 1:
            if self.mode in (5,6,8):
                if modk == 0:
                    self.parms.loc['bfit','SD.Kd'] = 0
                elif self.parms.loc['bfit','SD.v'] == self.parms.loc['bfit','SD.Kd'] and modk == -2:
                    self.parms.loc['bfit','SD.Kd'] = 0

            if self.mode in (3,4):
                if self.parms.loc['bfit','SD.v'] == self.parms.loc['bfit','SD.Kd'] and mk34 == 1:
                    self.parms.loc['bfit','SD.Kd'] = 0
                elif self.parms.loc['bfit','SD.v'] * self.parms.loc['bfit','SD.Kd'] == 0:
                    mk34 = 0

        # ---- ALPHA FOR MODE=8 => MAL8=1 if abs(b(nvar+7]-b(nvar+10)) < 1e-5 ----
        #  MAL8; Index for a stochastic ALPHA for MODE=8 
        #     MAL8=1  SDLNAL=SDLNV 
        #     MAL8=0  SDLNAL AND SALNV ARE INDEPENSENT.
        if self.mode == 8:
            if abs(self.parms.loc['binit','SD.v']- self.parms.loc['binit','SD.alp']) < 1.0e-5:
                mal8 = 1
            if self.mit >= 1:
                if self.parms.loc['bfit','SD.v'] == self.parms.loc['bfit','SD.alp'] and mal8 == 1:
                    self.parms.loc['bfit','SD.alp'] = 0
                elif self.parms.loc['bfit','SD.v'] * self.parms.loc['bfit','SD.alp'] == 0:
                    mal8 = 0
        return
