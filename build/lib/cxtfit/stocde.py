import numpy as np
from .detcde import DetCDE,dbexp
from scipy.integrate import quad

def xlnprob(x, avex, sdlnx):
    """
    Single log-normal distribution.
    """
    dpi = np.sqrt(2 * np.pi)
    xlnm = np.log(avex) - 0.5 * sdlnx * sdlnx
    xln = np.log(x)
    arg = (xln - xlnm) * (xln - xlnm) / (2.0 * sdlnx * sdlnx)
    return dbexp(-arg) / (dpi * sdlnx * x)

def blnprob(x, avex, sdlnx, y, avey, sdlny, rho):
    """
    Bivariate log-normal distribution.
    """
    xlnm = np.log(avex) - 0.5 * sdlnx * sdlnx
    ylnm = np.log(avey) - 0.5 * sdlny * sdlny
    x1 = (np.log(x) - xlnm) / sdlnx
    y1 = (np.log(y) - ylnm) / sdlny
    arg1 = (x1 * x1 - 2 * rho * x1 * y1 + y1 * y1) / (2.0 * (1.0 - rho * rho))
    arg2 = 2.0 * np.pi * sdlnx * sdlny * x * y * np.sqrt(1.0 - rho * rho)
    return dbexp(-arg1) / arg2

def limit(x, sdlnx, level=15, cmin = 1.0e-7):
    """
    Calculate integration limits for a log-normal distribution
    for stochastic models using Newton-Raphson method.
    """
    xlnm = np.log(x) - 0.5 * sdlnx * sdlnx
    xmod = dbexp(xlnm - sdlnx * sdlnx)
    xmin1 = max(1.0e-5, xmod - xmod * sdlnx)
    xmax1 = xmod + xmod * sdlnx

    # Calculate XMIN
    for i in range(level):
        clst = xlnprob(xmin1, x, sdlnx)
        if clst < cmin:
            break
        slope = -clst / (2.0 * sdlnx * sdlnx * xmin1) * (np.log(xmin1) - xlnm + sdlnx * sdlnx)
        xmin1 -= clst / slope
        if xmin1 <= 0.0:
            xmin1 = 1.0e-30
            break
            
    if i == level - 1:
        raise RuntimeError("WARNING ! OUT OF RANGE (LOWER LIMIT)")
    
    # Calculate XMAX
    for i in range(level):
        clst = xlnprob(xmax1, x, sdlnx)
        if clst < cmin:
            break
        slope = -clst / (2.0 * sdlnx * sdlnx * xmax1) * (np.log(xmax1) - xlnm + sdlnx * sdlnx)
        xmax1 -= clst / slope

    if i == level - 1:
        raise RuntimeError("WARNING ! OUT OF RANGE (LOWER LIMIT)")

    return xmin1, xmax1

# Converted from functions in STOCDE.FOR original Fortran code
class StoCDE(DetCDE):
    def StoCDE(self):
        """
        Stochastic models -- emsemble averages concentration
        """
        if self.mstoch in [1, 3]: # varaible velocity
            mcon = 1 # Phase 1 Concentration
            msd = 0 # Ensemble averages <C>
            c1, _ = quad(self.conprov, self.vmin, self.vmax, args=(mcon,msd))
            msd = 1 # Ensemble averages <C*C>
            c12, _ = quad(self.conprov, self.vmin, self.vmax, args=(mcon, msd))
            if self.modc in [4, 6] or self.mode in [3, 5]: # total resident concentration or equilibrium model
                c2 = 0.0
                c22 = 0.0
            else : # nonequilibrium model
                mcon  = 2
                msd = 0
                c2, _ = quad(self.conprov, self.vmin, self.vmax, args=(mcon,msd))
                msd = 1
                c22, _ = quad(self.conprov, self.vmin, self.vmax, args=(mcon, msd))
        elif self.mstoch in [2, 4]: # variable Y
            mcon = 1 # Phase 1 Concentration
            msd = 0 # Ensemble averages <C>
            c1, _ = quad(self.conproy, self.ymin, self.ymax, args=(mcon,msd))
            msd = 1
            c12, _ = quad(self.conproy, self.ymin, self.ymax, args=(mcon, msd))

            if self.modc in [4, 6] or self.mode in [3,5]: # total resident concentration or equilibrium model
                c2 = 0.0
                c22 = 0.0
            else :
                mcon = 2 # Phase 2 Concentration
                msd = 0 # Ensemble averages <C>
                c2, _ = quad(self.conproy, self.ymin, self.ymax, args=(mcon, msd))
                msd = 1 # Ensemble averages <C*C>
                c22, _ = quad(self.conproy, self.ymin, self.ymax, args=(mcon, msd))
        else:
            raise ValueError("Invalid value for MSTOCH")

        return c1, c2, c12-c1*c1, c22-c2*c2
    
    def conprov(self, vv, mcon = 1, msd = 0):
        """Calculate argument of stochastic models for a log-normal velocity distribution"""

        avev = self.v
        ttt = self.tt

        sd = self.d
        sp = self.p
        sr = self.r
        sdk = self.dk
        sbeta = self.beta
        salpha = self.alpha
        somega = self.omega
        sdmu1 = self.dmu1
        sdmu2 = self.dmu2
        spulse = self.cpulse[0]
        stpulse = [i for i in self.tpulse]
        sgamma1 = [i for i in self.gamma1]
        sgamma2 = [i for i in self.gamma2]

        if abs(vv) < 1.0e-30:
            vv = avev * 1.0e-30

        self.v = vv
        self.tt = ttt/avev * vv
        # ASSIGN NONDIMENTIONAL PARAMETERS TO EACH STREAM TUBE

        if self.mode in [5, 6]:
            m56 = 1
            self.sdln_d = self.sdlny
        else:
            m56 = 0

        if (m56 == 1 and self.mstoch != 3) or (m56 == 0 and self.modd == 0):
            self.p = self.p / avev * vv
        elif (m56 == 1 and self.mcorr == 1) or (m56 == 0 and self.modd >= 1):
            self.d = (vv / avev) ** (self.sdln_d / self.sdlnv) * sd * dbexp(0.5 * (self.sdlnv - self.sdln_d) * self.sdln_d)
            self.p = self.p / avev * vv * sd / self.d
        elif (m56 == 1 and self.mcorr == -1) or (m56 == 0 and self.modd <= -1):
            self.d = (vv / avev) ** (-self.sdln_d / self.sdlnv) * sd * dbexp(0.5 * (-self.sdlnv - self.sdln_d) * self.sdln_d)
            self.p = self.p / avev * vv * sd / self.d

        if self.mode in [3, 4]:
            m34 = 1
            self.sdlnk = self.sdlny
        else:
            m34 = 0

        if (m34 == 1 and self.mcorr == 1) or (m34 == 0 and self.modk >= 1):
            self.dk = (vv / avev) ** (self.sdlnk / self.sdlnv) * sdk * dbexp(0.5 * (self.sdlnv - self.sdlnk) * self.sdlnk)
            self.r = 1.0 + self.rhoth * self.dk
        elif (m34 == 1 and self.mcorr == -1) or (m34 == 0 and self.modk <= -1):
            self.dk = (vv / avev) ** (-self.sdlnk / self.sdlnv) * sdk * dbexp(0.5 * (-self.sdlnv - self.sdlnk) * self.sdlnk)
            self.r = 1.0 + self.rhoth * self.dk

        if self.mneq == 1:
            self.dmu2 = self.dmu2 / (sr - 1.0) * (self.r - 1.0)
            self.beta = 1.0 / self.r
            self.omega = self.omega / (sr - 1.0) * (self.r - 1.0)

        if self.mode in [3, 5]:
            self.dmu1 = self.dmu1 / sr * self.r

        if self.mode == 8 and self.mstoch == 3:
            if self.mcorr == 1:
                self.alpha = (vv / avev) ** (self.sdlny / self.sdlnv) * salpha * dbexp(0.5 * (self.sdlnv - self.sdlny) * self.sdlny)
            elif self.mcorr == -1:
                self.alpha = (vv / avev) ** (-self.sdlny / self.sdlnv) * salpha * dbexp(0.5 * (-self.sdlnv - self.sdlny) * self.sdlny)
            self.omega = somega / salpha * self.alpha

        self.omega = self.omega / vv * avev
        self.dmu1 = self.dmu1 / vv * avev
        self.dmu2 = self.dmu2 / vv * avev

        self.tpulse = [i/ avev * vv for i in stpulse]
        self.gamma1 = [i / vv * avev for i in sgamma1]
        self.gamma2 = [i / vv * avev for i in sgamma2]
              
        if self.modb == 1 and self.massst != 1:
            self.cpulse[0] = self.cpulse[0] / avev * vv
        if self.modb == 3 and self.massst == 1:
            self.tpulse[1] = stpulse[1]

        c1, c2 = self.DetCDE()

        if self.modc in [4, 6]: # total resident concentration
            c = c1 * self.beta * self.r + c2 * (1.0 - self.beta) * self.r
        elif mcon == 1:
            c = c1
        elif mcon == 2:
            if self.nredu == 0:
                c = c2 * self.dk
            else:
                c = c2
        else:
            raise ValueError("Invalid value for MCON")

        if self.mstoch == 4 and self.mcorr != 0:
            if self.mode in [3, 4]:
                prob = blnprob(vv, avev, self.sdlnv, sdk, self.avey, self.sdlny, self.corr)
            elif self.mode in [5, 6]:
                prob = blnprob(vv, avev, self.sdlnv, sd, self.avey, self.sdlny, self.corr)
            elif self.mode == 8:
                prob = blnprob(vv, avev, self.sdlnv, salpha, self.avey, self.sdlny, self.corr)
        else:
            prob = xlnprob(vv, avev, self.sdlnv)

        self.v = avev
        self.tt = ttt
        self.p = sp
        self.d = sd
        self.alpha = salpha
        self.r = sr
        self.dk = sdk
        self.beta = sbeta
        self.omega = somega
        self.dmu1 = sdmu1
        self.dmu2 = sdmu2

        self.cpulse[0] = spulse
        self.tpulse = [i for i in stpulse]
        self.gamma1 = [i for i in sgamma1]
        self.gamma2 = [i for i in sgamma2]

        if msd == 0: # <C>
            if self.modc == 2: # <C> field scale flux average concentration
                return prob * c * vv / avev
            else:
                return prob * c
        else: # <C*C>
            if self.modc == 2:
                return prob * c * c * vv * vv / avev / avev
            else:
                return prob * c * c
        
    def conproy(self, y, mcon = 1, msd = 0):
        """Calculate argument of stochastic models for a log-normal distribution of another parameter (v, D, K, alpha)"""
        mmode = self.mode
        sp = self.p
        sdk = self.dk
        sdd = self.d
        salpha = self.alpha
        sr = self.r
        sbeta = self.beta
        somega = self.omega
        sdmu1 = self.dmu1
        sdmu2 = self.dmu2

        if self.mode in [3, 4]:
            self.avey = self.dk
            self.dk = y
            self.r = 1 + self.rhoth * self.dk
            if self.mneq == 1:
                self.dmu2 = self.dmu2 / (sr - 1.0) * (self.r - 1.0)
                self.beta = 1.0 / self.r
                self.omega = self.omega / (sr - 1.0) * (self.r - 1.0)
            if self.mode == 3:
                self.dmu1 = self.dmu1 / sr * self.r
        elif self.mode in [5, 6]:
            self.avey = self.d
            self.d = y
            self.p = self.p * sdd / self.d
        elif self.mode == 8:
            self.avey = self.alpha
            self.alpha = y
            self.omega = self.omega / salpha * self.alpha

        prob = xlnprob(y, self.avey, self.sdlny)

        if self.mstoch == 2:
            c1, c2 = self.detcde()
            if self.modc in [4, 6]: # total resident concentration
                c = c1 * self.beta * self.r + c2 * (1.0 - self.beta) * self.r
            elif mcon == 1:
                c = c1
            elif mcon == 2:
                if self.nredu == 0:
                    c = c2 * self.dk
                else:
                    c = c2

            if msd == 0:
                conproy = prob * c
            else:
                conproy = prob * c * c

        if self.mstoch == 4:
            c, _ = quad(self.conprov, self.vmin, self.vmax, args=(mcon, msd))
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
        self.omega = somega
        self.dmu1 = sdmu1
        self.dmu2 = sdmu2

        return conproy