import numpy as np
from .detcde import DetCDE,dbexp
# from scipy.integrate import quad

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
    # print(f'X {x} SDLNX  {sdlnx}')
    xlnm = np.log(x) - 0.5 * sdlnx * sdlnx
    xmod = dbexp(xlnm - sdlnx * sdlnx)
    xmin1 = max(1.0e-5, xmod - xmod * sdlnx)
    xmax1 = xmod + xmod * sdlnx

    # Calculate XMIN
    for i in range(level):
        clst = xlnprob(xmin1, x, sdlnx)
        # print(f'I {i} CLST {clst} XMIN {xmin1} ')
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
        # print(f'I {i} CLST {clst} XMIN {xmax1} ')
        if clst < cmin:
            break
        slope = -clst / (2.0 * sdlnx * sdlnx * xmax1) * (np.log(xmax1) - xlnm + sdlnx * sdlnx)
        xmax1 -= clst / slope

    if i == level - 1:
        raise RuntimeError("WARNING ! OUT OF RANGE (UPPER LIMIT)")

    return xmin1, xmax1

def limit2(func, aa, bb, tfac = 1.2, ttol = 1.0e-3, miter = 100):
    t0 = aa
    t1 = bb

    tstep = 1.0
    cm0, _ = func(t0)
    if cm0 == 0: 
        for _ in range(miter): # exponential search
            t0 += tstep
            cm0, _ = func(t0)
            if (cm0 > 0): 
                tmin = t0 - tstep
                while t0 - tmin > ttol: # binary search
                    tmid = 0.5 * (t0 + tmin)
                    cm0, _ = func(tmid)
                    if cm0 > 0:
                        t0 = tmid
                    else:
                        tmin = tmid
                break
            tstep *= tfac
    
    tstep = 1.0
    cm0, _ = func(t1)
    if cm0 == 0:
        for _ in range(miter): # exponential search
            t1 -= tstep
            cm0, _ = func(t1)
            if cm0 > 0: 
                tmax = t1 + tstep
                while tmax - t1 > ttol: # binary search
                    tmid = 0.5 * (t1 + tmax)
                    cm0, _ = func(tmid)
                    if cm0 > 0:
                        t1 = tmid
                    else:
                        tmax = tmid
                break
            tstep *= tfac

    return t0, t1

def chebylog2(func, aa, bb, icheb=0, mm=100, stopch=1.0e-3, level=10, ctol=1.0e-10):
    """
    Perform integration of F(x) between log-transformed A and B 
    using M-point Gauss-Chebyshev quadrature formula.

    Args:
        func (callable): The function to integrate.
        aa (float): Lower limit of integration.
        bb (float): Upper limit of integration.
        icheb (int): If 0, use fixed number of integration points. 
                     If 1, increase number of points using stop criteria.
        mm (int): Number of integration points.
        stopch (float): The stopping criterion for the integration.
        level (int): The maximum number of iterations for adaptive integration.
        ctol (float): The tolerance for the integration.

    Returns:
        float: The approximate value of the integral.
    """
    a = np.log(aa)
    b = np.log(bb)
    summ1 = 0.0
    summ2 = 0.0
    m=mm
    if icheb != 1:
        for i in range(1, m + 1):
            z1 = np.cos((2 * (i - 1) + 1) * np.pi / (2 * m))
            x1 = (z1 * (b - a) + b + a) / 2.0
            dx1 = np.exp(x1)
            cm1, cm2 = func(dx1) 

            summ1 += dx1 * cm1 * np.sqrt(1.0 - z1 * z1)
            summ2 += dx1 * cm2 * np.sqrt(1.0 - z1 * z1)
        g = (b - a) * np.pi / (2 * m)
        return g * summ1, g * summ2
    
    area1 = 0.0    
    for k in range(level):
        summ1 = 0.0
        summ2 = 0.0
        for i in range(1, m + 1):
            z1 = np.cos((2 * (i - 1) + 1) * np.pi / (2 * m))
            x1 = (z1 * (b - a) + b + a) / 2.0
            dx1 = np.exp(x1)
            cm1, cm2 = func(dx1) 

            summ1 += dx1 * cm1 * np.sqrt(1.0 - z1 * z1)
            summ2 += dx1 * cm2 * np.sqrt(1.0 - z1 * z1)
        g = (b - a) * np.pi / (2 * m)
        area = g * summ1
        area2 = g * summ2
        # print(f'CHEBYLOG2: iteration = {k} m = {m} area = {area} area2 = {area2}')
        if abs(area) < ctol:
            return area, area2

        error = abs(area1 - area) / area
        if error < stopch:
            return area, area2
        else:
            area1 = area
            m *= 2
    raise ValueError("chebylog2 Failed to converge")

class StoCDE(DetCDE):
    def StoCDE(self):
        """
        Stochastic models -- emsemble averages concentration
        """
        self.mcon = 1 # Phase 1 Concentration
        if self.mstoch in [1, 3]: # varaible velocity
            c1, c12 = chebylog2(self.conprov,self.vmin,self.vmax)

            if self.modc in [4, 6] or self.mode in [3, 5]: # total resident concentration or equilibrium model
                c2 = 0.0
                c22 = 0.0
            else : # nonequilibrium model
                self.mcon  = 2
                c2, c22 = chebylog2(self.conprov, self.vmin, self.vmax)

        elif self.mstoch in [2, 4]: # variable Y
            c1, c12 = chebylog2(self.conproy, self.ymin, self.ymax)
            if self.modc in [4, 6] or self.mode in [3,5]: # total resident concentration or equilibrium model
                c2 = 0.0
                c22 = 0.0
            else :
                self.mcon = 2 # Phase 2 Concentration
                c2, c22 = chebylog2(self.conproy, self.ymin, self.ymax)
        else:
            raise ValueError("Invalid value for MSTOCH")

        return c1, c2, c12-c1*c1, c22-c2*c2
    
    def conprov(self, vv):
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
        stpulse = self.tpulse.copy()
        sgamma1 = self.gamma1.copy()
        sgamma2 = self.gamma2.copy()

        # ASSIGN NONDIMENTIONAL PARAMETERS TO EACH STREAM TUBE
        if abs(vv) < 1.0e-30:
            vv = avev * 1.0e-30

        self.tt = ttt/avev * vv

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

        if self.mode in [4, 6, 8] and self.mneq == 1:
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
            self.cpulse[0] = spulse * vv / avev
        if self.modb == 3 and self.massst == 1:
            self.tpulse[1] = stpulse[1]

        c1, c2 = self.DetCDE()

        if self.modc in [4, 6]: # total resident concentration
            c = c1 * self.beta * self.r + c2 * (1.0 - self.beta) * self.r
        elif self.mcon == 1:
            c = c1
        elif self.mcon == 2:
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

        # print(f'{self.tt} {self.zz} {self.v} {self.d} {self.p} {self.r}')
        # print(f'{self.tpulse}')
        # print(f'{self.cpulse}')
        # print(f'{vv} {c1} {c2} {prob}')
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
        self.tpulse = stpulse.copy()
        self.gamma1 = sgamma1.copy()
        self.gamma2 = sgamma2.copy()

        if self.modc == 2: # <C> field scale flux average concentration
            cmu1 = prob * c * vv / avev
        else:
            cmu1 = prob * c
        if self.modc == 2:
            cmu2 = prob * c * c * vv * vv / avev / avev
        else:
            cmu2 = prob * c * c

        return cmu1, cmu2
        
    def conproy(self, y):
        """Calculate argument of stochastic models for a log-normal distribution of another parameter (v, D, K, alpha)"""
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
            elif self.mcon == 1:
                c = c1
            elif self.mcon == 2:
                if self.nredu == 0:
                    c = c2 * self.dk
                else:
                    c = c2

            cmu1 = prob * c
            cmu2 = prob * c * c

        if self.mstoch == 4:
            cmu1, cmu2 = chebylog2(self.conprov, self.vmin, self.vmax)
            if self.mcorr == 0:
                cmu1 = cmu1 * prob
                cmu2 = cmu2 * prob

        self.p = sp
        self.dk = sdk
        self.d = sdd
        self.alpha = salpha
        self.r = sr
        self.beta = sbeta
        self.omega = somega
        self.dmu1 = sdmu1
        self.dmu2 = sdmu2

        return cmu1, cmu2