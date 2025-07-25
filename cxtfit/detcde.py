import numpy as np
# from scipy.special import erfc,i0,i1
from scipy.integrate import quad

def dbexp(x):
    if x < -100:
        return 0.0
    elif x > 700: # avoid overflow
        return np.exp(700.0)
    else:
        return np.exp(x)

def exf(a, b):
    """
    Calculate EXP(A) ERFC(B).

    Args:
        a (float): The value of A.
        b (float): The value of B.

    Returns:
        float: The calculated value.
    """
    if abs(a) > 170.0 and b <= 0.0:
        return 0.0

    c = a - b * b
    if abs(c) > 170.0 and b >= 0.0:
        return 0.0

    if c < -170.0:
        return 0.0

    x = abs(b)
    if x > 3.0:
        y = 0.5641896 / (x + 0.5 / (x + 1.0 / (x + 1.5 / (x + 2.0 / (x + 2.5 / x + 1.0)))))
    else:
        t = 1.0 / (1.0 + 0.3275911 * x)
        y = t * (0.2548296 - t * (0.2844967 - t * (1.421414 - t * (1.453152 - 1.061405 * t))))

    exf_value = y * np.exp(c)

    if b < 0.0:
        exf_value = 2.0 * np.exp(a) - exf_value

    return exf_value

def gold(x, y):
    # Converted from GOLD function in FUNC2.FOR original Fortran code
    """Calculate Goldstein's J-function J(X,Y)"""
    if abs(x) < 1e-10:
        x = 0.0
    gold = 0.0
    bf = 0.0
    e = 2.0 * np.sqrt(max(1.0e-35, x * y))
    if e < 1.0e-15:
        return dbexp(-x)
    z = x + y - e

    if z < 17.0:
        a = max(x, y)
        b = min(x, y)
        rt = 11.0 + 2.0 * b + 0.3 * a
        if rt > 25:
            da = np.sqrt(a)
            db = np.sqrt(b)
            p = 3.75 / e
            b0 = (0.3989423 + p * (0.01328592 + p * (0.00225319 - p * (0.00157565 - p * (0.00916281 - p * (0.02057706 - p * (0.02635537 - p * (0.01647633 - 0.00392377 * p)))))))) / np.sqrt(e)
            bf = b0 * dbexp(-z)
            p = 1.0 / (1.0 + 0.3275911 * (da - db))
            erf = p * (0.2548296 - p * (0.2844967 - p * (1.421414 - p * (1.453152 - p * 1.061405))))
            p = 0.25 / e
            c0 = 1.0 - 1.772454 * (da - db) * erf
            c1 = 0.5 - z * c0
            c2 = 0.75 - z * c1
            c3 = 1.875 - z * c2
            c4 = 6.5625 - z * c3
            summ = 0.1994711 * (a - b) * p * (c0 + 1.5 * p * (c1 + 1.666667 * p * (c2 + 1.75 * p * (c3 + p * (c4 * (1.8 - 3.3 * p * z) + 97.45313 * p)))))
            gold = 0.5 * bf + (0.3535534 * (da + db) * erf + summ) * bf / (b0 * np.sqrt(e))
        else:
            nt = int(rt)
            i = 0
            if x < y:
                i = 1
            gxy = 1.0 + i * (b - 1.0)
            gxyo = gxy
            gx = 1.0
            gy = gxy
            gz = 1.0
            for k in range(1, nt + 1):
                gx *= a / k
                gy *= b / (k + i)
                gz += gx
                gxy += gy * gz
                if abs((gxy - gxyo) / gxy) < 1.0e-08:
                    break
                gxyo = gxy
            gold = gxy * dbexp(-x - y)
    
    if x < y:
        gold = 1.0 + bf - gold
    
    return gold

def expbi0(x, z):
    """Returns EXP(Z)*I0(X) for any real X and Z"""
    p = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813]
    q = [0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281, -0.02057706, 0.02635537, -0.01647633, 0.00392377]
    if abs(x) < 3.75:
        y = (x / 3.75) ** 2
        expbi0 = np.exp(z) * (p[0] + y * (p[1] + y * (p[2] + y * (p[3] + y * (p[4] + y * (p[5] + y * p[6]))))))
    else:
        ax = abs(x)
        y = 3.75 / ax
        expbi0 = np.exp(ax + z) / np.sqrt(ax) * (q[0] + y * (q[1] + y * (q[2] + y * (q[3] + y * (q[4] + y * (q[5] + y * (q[6] + y * (q[7] + y * q[8]))))))))
    return expbi0

# Converted from EXPBI1 function in FUNC2.FOR original Fortran code
def expbi1(x, z):
    """Returns EXP(Z)*I1(X) for any real X and Z"""
    p = [0.5, 0.87890594, 0.51498869, 0.15084984, 0.02658733, 0.00301532, 0.00032411]
    q = [0.39894228, -0.03988024, -0.003662018, 0.00163801, -0.01031555, 0.02282967, -0.02895312, 0.01787654, -0.00420059]
    if abs(x) < 3.75:
        y = (x / 3.75) ** 2
        expbi1 = np.exp(z) * x * (p[0] + y * (p[1] + y * (p[2] + y * (p[3] + y * (p[4] + y * (p[5] + y * p[6]))))))
    else:
        ax = abs(x)
        y = 3.75 / ax
        expbi1 = np.exp(ax + z) / np.sqrt(ax) * (q[0] + y * (q[1] + y * (q[2] + y * (q[3] + y * (q[4] + y * (q[5] + y * (q[6] + y * (q[7] + y * q[8]))))))))
        if x < 0:
            expbi1 = -expbi1
    return expbi1

def chebycon(func, a, b, mc = 0, icheb=1, mm=8, stopch=1.0e-3, level=10, ctol=1.0e-10):
    """
    Perform integration of F(x) between A and B using M-point Gauss-Chebyshev quadrature formula.
    This function evaluates integrals for nonequilibrium CDE.

    Args:
        func (callable): The function to integrate.
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.
        mc (int): mc = 1 mobile concentration mc = 2 immobile concentration. Default is None.
        icheb (int): If 0, use fixed number of integration points. If 1, increase number of points using stop criteria.
        mm (int): Number of Gauss-Chebyshev nodes.
        stopch (float): The stopping criterion for the integration.
        level (int): The maximum number of iterations for adaptive integration.
        ctol (float): The tolerance for the integration.

    Returns:
        float: The approximate value of the integral.
    """
    m = mm
    if icheb != 1:
        summ = 0.0
        for i in range(1, m + 1):
            z1 = np.cos((2 * (i - 1) + 1) * np.pi / (2 * m))
            x1 = (z1 * (b - a) + b + a) / 2.0
            if mc:
                fx1 = func(x1, mc)
            else:
                fx1 = func(x1)
            summ += fx1 * np.sqrt(1.0 - z1 * z1)
        area = (b - a) * np.pi * summ / (2 * m)
        return area

    area1 = 0.0
    for _ in range(level):
        summ = 0.0
        for i in range(1, m + 1):
            z1 = np.cos((2 * (i - 1) + 1) * np.pi / (2 * m))
            x1 = (z1 * (b - a) + b + a) / 2.0
            if mc:
                fx1 = func(x1, mc)
            else:
                fx1 = func(x1)
            summ += fx1 * np.sqrt(1.0 - z1 * z1)
        area = (b - a) * np.pi * summ / (2 * m)

        if abs(area) < ctol:
            return area

        error = abs(area - area1) / area
        if error < stopch:
            # print(f'chebycon: m = {m} area = {area} area1 = {area1} error = {error}')
            return area
        else:
            area1 = area
            m *= 2
    raise ValueError("chebycon Failed to converge")
    
# Converted from functions in DETCDE.FOR original Fortran code
class DetCDE:
    """
    Calculate deterministic equilibrium & nonequilibrium CDE
    """
    def DetCDE(self):
        """Calculate deterministic equilibrium & nonequilibrium CDE"""      
        self.betr = self.beta * self.r
        omegamu2 = self.omega + self.dmu2

        if self.mode in (1,3,5) or omegamu2 < self.ctol: 
            self.a = 0.0
            self.da = 0.0
            self.cx = 0.0
        else: # nonequilibrium CDE
            self.a = self.omega * self.omega / omegamu2 / self.betr
            self.da = self.omega * self.dmu2 / omegamu2
            self.cx = self.omega / omegamu2

        if self.mode in (1,3,5) or (self.beta >= 0.9999999 and self.omega < self.ctol) or omegamu2 < self.ctol:
            self.b = 0.0
        else:
            self.b = omegamu2 / (self.r -self.betr)

        cbou1, cbou2 = self.bound()
        cbou1 = cbou1 if cbou1 > self.ctol else 0.0
        cbou2 = cbou2 if cbou2 > self.ctol else 0.0
        cint1, cint2 = self.initial()
        cint1 = cint1 if cint1 > self.ctol else 0.0
        cint2 = cint2 if cint2 > self.ctol else 0.0
        cpro1, cpro2 = self.produc()
        cpro1 = cpro1 if cpro1 > self.ctol else 0.0
        cpro2 = cpro2 if cpro2 > self.ctol else 0.0

        c1 = cbou1 + cint1 + cpro1
        if self.mode in (2,4,6,8): # nonequilibrium CDE
            c2 = cbou2 + cint2 + cpro2
        else: # equilibrium CDE
            c2 = 0.0

        return c1, c2
    
    def bound(self):
        """Boundary value problem (BVP) for equilibrium and nonequilibrium CDE"""

        if self.modb == 0 or self.tt <= self.dtmin: # Zero input or initial time
            return 0.0, 0.0
        elif self.modb == 1: # Dirac Delta input
            if self.nredu <= 1:
                bmass = self.cpulse[0] * self.v / self.zl
            else:
                bmass = self.cpulse[0]

            c1 = self.cc0(self.tt) * dbexp(-self.omega * self.tt / self.betr)
            # print(self.cpulse[0],self.v, self.zl,bmass)
            if self.mode in (1,3,5): # equilibrium CDE
                c1 *= bmass
                return c1, 0.0
            else :
                ap = np.sqrt(1.0 + self.p * self.zz / 30.0)
                tmax = min(self.tt, self.betr * (self.zz + 60.0 * (1.0 + ap) / self.p))
                tmin = max(0.0, self.betr * (self.zz + 60.0 * (1.0 - ap) / self.p))

                mc = 1
                a1 = chebycon(self.ctran, tmin, tmax, mc)
                # a1, _ = quad(self.ctran, tmin, tmax, args=(mc,))
                c1 = (c1 + a1) * bmass

                mc = 2
                c2 = chebycon(self.ctran, tmin, tmax, mc) * bmass
                # a2, _ = quad(self.ctran, tmin, tmax, args=(mc,))
                # c2 = a2 * bmass

                return c1, c2

        # Multiple pulse input (modb = 2,3,4), 
        # Step input (modb == 2) and pulse input (modb == 3) are special cases  of 
        # the multiple pulse input (modb == 4)
        elif self.modb in (2, 3, 4): 
            ttt = self.tt
            c1 = 0.0
            c2 = 0.0
            for i in range(self.npulse):
                ttt -= self.tpulse[i]
                if ttt <= 0:
                    return c1, c2

                if self.mode in (1,3,5):
                    a1 = self.cc4(ttt, self.da)
                    a2 = 0.0
                else:
                    if ttt <= 0 :
                        return c1, c2
                    mc = 1
                    a1, _ = quad(self.cbj, 0, ttt, args=(ttt,mc))
                    mc = 2
                    a2, _ = quad(self.cbj, 0, ttt, args=(ttt,mc))
                if i == 0:
                    c1 = self.cpulse[i] * a1
                    c2 = self.cpulse[i] * a2
                else:
                    c1 += (self.cpulse[i] - self.cpulse[i - 1]) * a1
                    c2 += (self.cpulse[i] - self.cpulse[i - 1]) * a2

            return c1, c2
        
        elif self.modb == 5: # Exponential input
            if self.mode in (1,3,5):
                a1 = self.cc4(self.tt, self.da)
                a2 = 0.0
            else:
                mc = 1
                a11 = chebycon(self.cbal, 0.0, self.tt, mc)
                # a11, _ = quad(self.cbal, 0.0, self.tt, args=(mc,))
                a1 = self.cc4(self.tt, self.da) * dbexp(-self.a * self.tt) + a11

                mc = 2
                a2 = chebycon(self.cbal, 0.0, self.tt, mc)
                # a2, _ = quad(self.cbal, 0.0, self.tt, args=(mc,))

            c1 = self.cpulse[0] * a1
            c2 = self.cpulse[0] * a2

            #EXPONENTIAL TERM  TPULSE(1)= lamda^b
            if abs(self.tpulse[0]) < self.dtmin:
                c1 += self.cpulse[1] * a1
                c2 += self.cpulse[1] * a2
                return c1, c2

            if self.mode in (1,3,5):
                dum = -self.r * self.tpulse[0]
                dum1 = 1.0 + 4.0 * (self.dmu1 + dum) / self.p
                mc = 1
                if dum1 < self.parmin:
                    a1 = chebycon(self.cbexp, 0.0, self.tt, mc)
                    # a1, _ = quad(self.cbexp, 0.0, self.tt, args=(mc,))
                else:
                    a1 = dbexp(-self.tpulse[0] * self.tt) * self.cc4(self.tt, dum)
            else:
                mc = 1
                a1 = chebycon(self.cbexp, 0.0, self.tt, mc)
                # a1, _ = quad(self.cbexp, 0.0, self.tt, args=(mc,))

                mc = 2
                a2 = chebycon(self.cbexp, 0.0, self.tct, mc)
                # a2, _ = quad(self.cbexp, 0.0, self.tt, args=(mc,))
                a2 = self.omega / (self.r - self.betr) * a2

            c1 += self.cpulse[1] * a1
            c2 += self.cpulse[1] * a2

            return c1, c2

        elif self.modb == 6: # Arbitrary input
            c1 = chebycon(self.cbin1, 0.0, self.tt)
            # c1, _ = quad(self.cbin1, 0.0, self.tt)
            if self.modc not in [3, 4] and self.zz < self.dzmin:
                c1 = self.cinput(self.tt)

            if self.mode in (1,3,5):
                return c1, 0.0
            else :
                c2 = chebycon(self.cbin2, 0.0, self.tt)
                # c2, _ = quad(self.cbin2, 0.0, self.tt)
                c2 = c2 * self.omega / (self.r - self.betr)

                return c1, c2
        else:
            raise ValueError("ERROR! MODB SHOULD BE 0,1,2,3,4,5,6")
      
    def initial(self):
        """Initial value problem (IVP) for equilibrium and nonequilibrium CDE"""
        if self.modi == 0: # Zero initial concentration
            return 0.0, 0.0

        mmodi = self.modi
        if mmodi == 4 and self.cini[0] > self.ctol:
            mmodi = 1
        
        # Initial concentration for small T
        if self.tt <= self.dtmin: 
            # stepwise initial distribution (modi = 0, 1, 2)
            # Zero initial distribution (modi = 0) and constant initial distribution (modi = 1) 
            # are special cases of stepwise initial distribution (modi = 2)
            if mmodi in (1, 2):
                i = np.argmin(abs(self.zini[i] - self.zz))
                c1 = self.cini[i]
                c2 = self.cini[i]
                return c1, c2
            elif mmodi == 3: # exponential initial distribution
                c1 = self.cini[0] + self.cini[1] * dbexp(-self.zini[0] * self.zz)
                return c1, c1
            elif mmodi == 4: # Dirac delta initial distribution
                return 0.0, 0.0
            else :
                raise ValueError("ERROR! MODI SHOULD BE 0,1,2,3,4")
            
        # prepare constants
        dg = dbexp(-(self.dmu1 + self.da) / self.betr * self.tt)
        if self.mode in (1,3,5):
            a1 = 0.0
        else:
            mc = 1
            # a1 = chebycon(self.civp, 0.0, self.tt, mc)
            a1, _ = quad(self.civp, 0.0, self.tt, args=(mc,))
            mc = 2
            # a2 = chebycon(self.civp, 0.0, self.tt, mc)
            a2, _ = quad(self.civp, 0.0, self.tt, args=(mc,))

        mcc0 = self.cc0(self.tt)
        mcc1 = self.cc1(self.tt)

        c1=0.0
        c2=0.0
        if mmodi in (1, 2): # stepwise initial distribution
            for i in range(self.nini):
                if i == 0:
                    g = -self.cini[i] * dg * (mcc1 - 1.0)
                else:
                    mcc2 = self.cc2(self.tt, self.zini[i])
                    g += (self.cini[i - 1] - self.cini[i]) * dg * (mcc2 - 1.0)
            c1 = g * dbexp(-self.a * self.tt) + a1
            if self.mode in (2, 4, 6, 8): # nonequilibrium CDE
                i = np.argmin(abs(self.zini[i] - self.zz))
                c2 = self.cini[i] * dbexp(-self.b * self.tt) + a2
            return c1, c2
        elif mmodi == 3: # exponential initial distribution
            U1 = self.cini[0]
            U2 = self.cini[1]
            lamda = self.zini[0]
            mcc3 = self.cc3(self.tt, lamda)
            c1 = (U1 * dg * (1.0 - mcc1) + U2  * dg * mcc3) * dbexp(-self.a * self.tt) + a1
            if self.mode in (2, 4, 6, 8):
                c2 = (U1 + U2 * dbexp(-lamda * self.zz)) * dbexp(-self.b * self.tt) + a2
            return c1, c2
        elif mmodi == 4: # Dirac delta initial distribution
            U1 = self.cini[0]
            dmass = self.cini[1]
            zini = self.zini[1]
            mcc5 = self.cc5(self.tt, zini)
            if self.nredu != 2:
                dmass /= self.zl
            if dmass < self.ctol:
                return c1, c2
            
            if self.modc <= 4 and abs(zini) < self.dzmin:
                g = dmass * dg * mcc0 * self.beta * self.r 
            else :
                g = dmass * dg * mcc5
            c1 += g * dbexp(-self.a * self.tt) + dmass * a1
            if self.mode in (2, 4, 6, 8) :
                c2 += dmass * a2
            return c1, c2
        else :
            raise ValueError("ERROR! MODI SHOULD BE 0,1,2,3,4")
        
    def produc(self):
        """Production value problem (PVP) for equilibrium and nonequilibrium CDE"""
       
        if self.modp == 0 or self.tt <= self.dtmin: # initial time or zero production
            return 0.0, 0.0

        # Equilibrium CDE with constant production term
        
        # if mods == 1 and (self.npro1 == 1 or self.modp == 1):
        #     if self.modp1 == 0: # Eq.(2.32) 
        #         if (self.omega + self.dmu1) < self.ctol:
        #             c1 = self.prod0(self.tt)
        #             return c1, c2
        #         else:
        #             c1 = self.gamma1[0] / self.dmu1 * (1 - dbexp(-(self.dmu1 * self.tt) / self.betr) * (1 - self.cc1(self.tt)) - self.cc4(self.tt, self.da))
        #             return c1, c2

        # Eq.(2.33) or (2.34) (current setting)
      
        c1 = chebycon(self.c1pro, 0.0, self.tt)
        # c1, _ = quad(self.c1pro, 0.0, self.tt)

        if self.mode in (1, 3, 5): # equilibrium CDE
            return c1, 0.0

        # Nonequilibrium CDE phase2 concentration
        omegamu2 = self.omega + self.dmu2
        tconv = self.tt / (self.r - self.betr)
        if omegamu2 < self.ctol:
            if self.modp in (1, 2):
                i = np.argmin(abs(self.zpro2[i] - self.zz))
                c2 = tconv * self.gamma2[i]
                return c1, c2
            elif self.modp == 3:
                c2 = tconv * (self.gamma2[0] + self.gamma2[1] * dbexp(-self.zpro2[0] * self.zz))
                return c1, c2
            else:
                raise ValueError("ERROR! MODP SHOULD BE 1,2,3")
        else:
            a2 = chebycon(self.c2pro, 0.0, self.tt)
            # a2, _ = quad(self.c2pro, 0.0, self.tt)
            a3 = omegamu2 * (1.0 - dbexp(-omegamu2 * tconv))

            if self.modp in (1, 2):
                i = np.argmin(abs(self.zpro2[i] - self.zz))
                c2 = self.gamma2[i] / a3 + a2
                return c1, c2
            elif self.modp == 3:
                c2 = (self.gamma2[0] + self.gamma2[1] * dbexp(-self.zpro2[0] * self.zz)) / a3 + a2
                return c1, c2
            else:
                raise ValueError("ERROR! MODP SHOULD BE 1,2,3")

    def ctran(self, tau, mc):
        """Calculate argument in integral for delta input (transfer function model)"""
        g = self.cc0(tau)
        if g < self.ctol:
            return 0.0

        aa = self.omega * tau / self.betr
        at = self.a * tau
        bt = self.b * (self.tt - tau)
        xii = 2.0 * np.sqrt(at * bt)
        self.beta = self.betr / self.r

        if mc == 1:
            cbi1 = np.sqrt(tau / ((1 - self.beta) * self.beta * (self.tt - tau))) * self.omega / self.r
            return g * expbi1(xii, -aa - bt) * cbi1
        else:
            return self.omega / (self.r - self.betr) * g * expbi0(xii, -aa - bt)

    def cbj(self,tau, ttt, mc):
        """
        PURPOSE: CALCULATE ARGUMENT IN INTEGRAL FOR STEP INPUT
        (SOLUTION USING GOLDSTEIN'S J-FUNCTION)
        """
        g = self.cc0(tau)
        if g < self.ctol:
            return 0.0

        at = self.a * tau
        bt = self.b * (ttt - tau)
        if mc == 1:
            return g * gold(at, bt) * dbexp(-at * self.dmu2 / self.omega)
        else:
            return self.cx * g * (1.0 - gold(bt, at)) * dbexp(-at * self.dmu2 / self.omega)

    def cbal(self, tau, mc):
        """Calculate argument in integral for step input"""
        g = self.cc4(tau, self.da)
        if g < self.ctol:
            return 0.0

        at = self.a * tau
        bt = self.b * (self.tt - tau)
        xii = 2.0 * np.sqrt(at * bt)

        if mc == 1:
            cbi1 = np.sqrt(tau / ((1 - self.beta) * self.beta * (self.tt - tau)))
            cbi0 = self.cx / self.beta
            cbi2 = expbi0(xii, -at - bt) * cbi0 + expbi1(xii, -at - bt) * cbi1
            return self.omega / self.r * g * cbi2
        else:
            cbi1 = np.sqrt((1 - self.beta) * (self.tt - tau) / tau / self.beta) * self.cx
            cbi2 = expbi0(xii, -at - bt) + expbi1(xii, -at - bt) * cbi1
            return self.omega / (self.r - self.betr) * g * cbi2

    def cbexp(self, tau, mc):
        """Calculate argument in integral for exponential input"""
        mods = self.mode % 2
        g = self.cc0(tau)
        if g < self.ctol:
            return 0.0

        tpulse = self.tpulse[0]
        if mods == 1:
            return g * dbexp(-tpulse * (self.tt - tau))

        c1 = -self.omega * tau / self.betr + self.b * tau
        c2 = -self.omega * tau / self.betr + tpulse * tau
        c3 = tpulse - self.b

        if mc == 1:
            mexp1 = dbexp(-tpulse * self.tt + c2 - self.a * self.b / c3 * tau)
            mexp2 = dbexp(-self.b * self.tt + c1)
            mphi1 = self.phi1(tau)
            return g * (mexp1 - mexp2 * mphi1)
        else:
            mexp1 = dbexp(-self.b * self.tt + c1 - self.a * self.b * tau / c3)
            mexp2 = dbexp(-tpulse * self.tt + c2 - self.a * self.b * tau / c3)
            mexp3 = dbexp(-self.b * self.tt + c1)
            mphi2 = self.phi2(tau)
            return g * (mexp1/ c3 - mexp2/ c3 - mexp3* mphi2)

    def cbin1(self, tau):
        """Calculate argument in integral for arbitrary input given in function CINPUT"""
        mods = self.mode % 2
        g = self.cc0(tau)
        if g < self.ctol and mods == 1:
            return 0.0

        c1 = g * dbexp(-self.omega * tau / self.betr)
        if mods != 1:
            mc = 1
            a1 = chebycon(self.ctran, 0.0, tau, mc)
            # a1, _ = quad(self.ctran, 0.0, tau, args=(mc,))
        return (c1 + a1) * self.cinput(self.tt - tau)

    def cbin2(self, tau):
        """Calculate argument in integral for arbitrary input given in function CINPUT for nonequilibrium phase"""

        a1 = self.cheby2(self.cbin1, 0.0, tau)
        # a1, _ = quad(self.cbin1, 0.0, tau)
        return a1 * dbexp(-self.b * (self.tt - tau))

    def civp(self, tau, mc):
        """Calculate argument in initial value problem"""
        self.beta = self.betr / self.r
        dg = dbexp(-(self.dmu1 + self.da) / self.betr * tau)

        if self.modi in (1,2):
            g = -self.cini[0] * dg * (self.cc1(tau) - 1.0)
            for i in range(self.nini-1):
                g += (self.cini[i] - self.cini[i+1]) * dg * (self.cc2(tau, self.zini[i+1]) - 1.0)
        elif self.modi == 3:
            g = self.cini[0]* dg * (1.0 - self.cc1(tau)) + self.cini[1] * dg * self.cc3(tau, self.zini[0])
        elif self.modi == 4:
            if self.modc <= 4 and abs(self.zini[1]) < self.dzmin:
                g = dg * self.cc0(tau) * self.beta * self.r
            else:
                g = dg * self.cc5(tau, self.zini[1])

        at = self.a * tau
        bt = self.b * (self.tt - tau)
        xii = 2.0 * np.sqrt(at * bt)

        if mc == 1:
            cbi1 = np.sqrt(tau / ((1 - self.beta) * self.beta * (self.tt - tau))) * self.omega / self.r
            civp = expbi0(xii, -at - bt) * self.omega / self.betr + expbi1(xii, -at - bt) * cbi1
            civp = g * civp
        else:
            cbi1 = np.sqrt((1 - self.beta) * (self.tt - tau) / self.beta / tau)
            civp = expbi0(xii, -at - bt) + expbi1(xii, -at - bt) * cbi1
            civp = self.omega / (self.r - self.betr) * g * civp

        return civp

    def c1pro(self, tau):
        """Calculate argument in production term for equilibrium concentration"""
        dg = dbexp(-(self.dmu1 + self.da) / self.betr * tau)
        gcc1 = self.cc1(tau)
        if self.modp in (1,2):
            if self.npro1 > 0:
                g = -self.gamma1[0] / self.betr * dg * (gcc1 - 1.0)
                for i in range(self.npro1-1):
                    gcc2 = self.cc2(tau, self.zpro1[i+1])
                    g += (self.gamma1[i] - self.gamma1[i+1]) / self.betr * dg * (gcc2 -1.0) 
            if self.mode in (2,4,6,8) and self.npro2 > 0:
                h = -self.cx * self.gamma2[0]/ self.betr * dg * (gcc1 - 1.0)
                for i in range(self.npro2-1):
                    gcc2 = self.cc2(tau, self.zpro2[i+1])
                    h += self.cx * (self.gamma2[i] - self.gamma2[i+1]) / self.betr * dg * (gcc2 - 1.0)
                g += h
        elif self.modp == 3:
            gcc3 = self.cc3(tau, self.zpro1[0])
            g = (self.gamma1[0] * (1.0 - gcc1) + self.gamma1[1] * gcc3) * dg / self.betr
            if self.mode in (2,4,6,8):
                gcc3 = self.cc3(tau, self.zpro2[0])
                h = (self.gamma2[0] * (1.0 - gcc1) + self.gamma2[1] * gcc3) * dg / self.betr * self.cx
                g += h
        else :
            raise ValueError("ERROR! MODP SHOULD BE 0,1,2,3")

        if self.mode in (1,3,5):
            return g
        else :
            at = self.a * tau
            bt = self.b * (self.tt - tau)
            xii = 2.0 * np.sqrt(at * bt)
            return g * gold(at, bt) - h * expbi0(xii, -at - bt)

    def c2pro(self, tau):
        """Calculate argument in production term for nonequilibrium concentration"""
        dg = dbexp(-(self.dmu1 + self.da) / self.betr * tau)
        gcc1 = self.cc1(tau)

        if self.modp == 1:
            if self.npro1 == 0:
                return 0.0
            g = -self.gamma1[0] / self.betr * dg * (gcc1 - 1.0)
            for i in range(self.npro1-1):
                g += (self.gamma1[i] - self.gamma1[i+1]) / self.betr * dg * (self.cc2(tau, self.zpro1[i+1]) - 1.0)
            if self.npro2 == 0:
                return 0.0
            h = -self.cx * self.gamma2[0] / self.betr * dg * (gcc1 - 1.0)
            for i in range(self.npro2-1):
                h += self.cx * (self.gamma2[i] - self.gamma2[i+1]) / self.betr * dg * (self.cc2(tau, self.zpro2[i+1]) - 1.0)
            g += h
        elif self.modp == 2:
            g = (self.gamma1[0] * (1.0 - gcc1) + self.gamma1[1] * self.cc3(tau, self.zpro1[0])) * dg / self.betr
            h = (self.gamma2[0] * (1.0 - gcc1) + self.gamma2[1] * self.cc3(tau, self.zpro2[0])) * dg / self.betr * self.cx
            g += h

        at = self.a * tau
        bt = self.b * (self.tt - tau)
        self.beta = self.betr / self.r
        xii = 2.0 * np.sqrt(at * bt)
        cbi1 = np.sqrt(self.beta / (1 - self.beta) * (self.tt - tau) / tau)

        return self.cx * g * (1.0 - gold(bt, at)) - cbi1 * h * expbi1(xii, -at - bt)

    def cc0(self, tau):
        """Calculate solutions for delta input travel time distribution for equilibrium CDE"""
        dg = dbexp(-self.dmu1 / self.betr * tau)
        g1 = dbexp(self.p * (self.betr * self.zz - tau) * (tau - self.betr * self.zz) / (4.0 * self.betr * tau))
        
        if self.modc in (3, 4):  # third-type concentration
            g2 = np.sqrt(self.p / (self.betr * tau))
            gexf = exf(self.p * self.zz, g2 / 2.0 * (self.betr * self.zz + tau))
            return dg * (0.56419 * g2 * g1 - self.p / (2.0 * self.betr) *  gexf)
        else:
            g2 = np.sqrt(self.p * self.betr / (4.0 * np.pi * tau))
            return dg * (self.zz / tau) * g2 * g1
        
    def cc1(self, tau):
        """Calculate equilibrium solutions for step input"""
        ba = self.p / (4.0 * self.betr * tau)
        rba = np.sqrt(ba)
        rbb = np.sqrt(self.p * tau / np.pi / self.betr)
        bb1 = rba * (self.betr * self.zz - tau)
        g1 = exf(0.0, bb1)
        aa2 = self.p * self.zz
        bb2= rba * (self.betr * self.zz + tau)
        g2 = exf(aa2, bb2) 

        if self.modc in (3, 4): # third-type concentration
            g3 = dbexp(-ba * (self.betr * self.zz - tau) * (self.betr * self.zz - tau))
            return g1 / 2.0 + rbb * g3 - (1.0 + self.p * self.zz + self.p * tau / self.betr) * g2 / 2.0
        else:
            return (g1 + g2) / 2.0

    def cc2(self, tau, z1):
        """Calculate argument for stepwise initial and production profiles"""
        ba = self.p / (4.0 * self.betr * tau)
        rba = np.sqrt(ba)
        rbb = np.sqrt(self.p * tau / np.pi / self.betr)
        g1 = exf(0.0, rba * (self.betr * (self.zz - z1) - tau))
        g2 = exf(self.p * self.zz, rba * (self.betr * (self.zz + z1) + tau))
        
        if self.modc in (3, 4):  # third-type concentration
            g3 = dbexp(self.p * self.zz - ba * (self.betr * (self.zz + z1) + tau) * (self.betr * (self.zz + z1) + tau))
            return g1 / 2.0 + rbb * g3 - (1.0 + self.p * (self.zz + z1) + self.p * tau / self.betr) * g2 / 2.0
        elif self.modc in (5, 6): # first-type concentration
            return (g1 + g2) / 2.0
        else:
            rbc = np.sqrt(self.betr / np.pi / self.p / tau) / 2.0
            a1 = rbc * dbexp(-ba * (self.betr * (self.zz - z1) - tau) * (self.betr * (self.zz - z1) - tau))
            g3 = dbexp(self.p * self.zz  - ba * (self.betr * (self.zz + z1) + tau) * (self.betr * (self.zz + z1) + tau))
            a2 = rbc * g3
            return (g1 + g2) / 2.0 + a1 - a2

    def cc3(self, tau, z1):
        """Calculate argument for exponential initial and production profiles"""
        ba = self.p / (4.0 * self.betr * tau)
        rba = np.sqrt(ba)
        a1 = dbexp(z1 * z1 * tau / self.betr / self.p + z1 * tau / self.betr - z1 * self.zz)
        g1 = exf(0.0, rba * (self.betr * self.zz - (1.0 + 2.0 * z1 / self.p) * tau))
        g2 = exf(self.p * self.zz + 2.0 * z1 * self.zz, rba * (self.betr * self.zz + (1.0 + 2.0 * z1 / self.p) * tau))
        
        if self.modc in (1, 2): # flux concentration
            return (1.0 + z1 / self.p) * a1 * (2.0 - g1 - g2) / 2.0
        elif self.modc in (3, 4):  # third-type concentration
            g3 = exf(self.p * self.zz, rba * (self.betr * self.zz + tau))
            return a1 * (1.0 - g1 / 2.0 + (1 + self.p / z1) * g2 / 2.0) - self.p / z1 / 2.0 * g3
        elif self.modc in (5, 6): # first-type concentration
            return a1 * (2.0 - g1 - g2) / 2.0
        else:
            raise ValueError("ERROR! MODC SHOULD BE 1,2,3,4,5,6")

    def cc4(self, tau, dum):
        """Argument for step input or constant production"""
        u = np.sqrt(1.0 + 4.0 * (self.dmu1 + dum) / self.p)
        ba = self.p / (4.0 * self.betr * tau)
        rba = np.sqrt(ba)
        rbb = np.sqrt(self.p * tau / np.pi / self.betr)
        aa1 = self.p * (1.0 - u) * self.zz / 2.0
        bb1 = rba * (self.betr * self.zz - u * tau)
        g1 = exf(aa1, bb1)
        aa2 = self.p * (1.0 + u) * self.zz / 2.0
        bb2 = rba * (self.betr * self.zz + u * tau)
        g2 = exf(aa2, bb2)

        if self.modc in (3, 4): # third-type concentration
            if abs(self.dmu1 + dum) < self.ctol:
                aa3 = -ba * (self.betr * self.zz - tau) * (self.betr * self.zz - tau)
                g3 = dbexp(aa3)
                rbc = (1.0 + self.p * self.zz + self.p * tau / self.betr) / 2.0
                return g1 / 2.0 + rbb * g3 -  rbc* g2
            else:
                aa3=self.p * self.zz - (self.dmu1 + dum) * tau / self.betr
                bb3 = rba * (self.betr * self.zz + tau)
                g3 = exf(aa3, bb3)
                rbc = self.p / 2.0 / (self.dmu1 + dum)
                return 1 / (1 + u) * g1 + 1 / (1 - u) * g2 +  rbc * g3
        else:
            return (g1 + g2) / 2.0

    def cc5(self, tau, z1):
        """Calculate argument for delta initial condition or general IVP & PVP"""
        ba = self.betr * self.p / (4.0 * np.pi * tau)
        rba = np.sqrt(ba)
        bb = self.p / 4.0 / self.betr / tau
        g1 = dbexp(-bb * (self.betr * (z1 - self.zz) + tau) * (self.betr * (z1 - self.zz) + tau))
        g2 = dbexp(self.p * self.zz - bb * (self.betr * (z1 + self.zz) + tau) * (self.betr * (z1 + self.zz) + tau))
        
        if self.modc in (1, 2): # flux concentration
            bc = 1.0 - (self.betr * (z1 - self.zz) * tau) / 2.0 / tau
            bd = 1.0 - (self.betr * (z1 + self.zz) * tau) / 2.0 / tau
            return (bc * g1 - bd * g2) * rba
        elif self.modc in (3, 4): # third-type concentration
            rbb = np.sqrt(bb)
            g3 = exf(self.p * self.zz, rbb * (self.betr * (z1 + self.zz) + tau))
            return (g1 + g2) * rba - self.p / 2.0 * g3
        elif self.modc in (5, 6): # first-type concentration
            return (g1 - g2) * rba
        else:
            raise ValueError("ERROR! MODC SHOULD BE 1,2,3,4,5,6")

    def prod0(self, tau):
        """Calculate analytical solutions of constant production term for equilibrium CDE in case of dmu1=0"""
        ba = self.p / (4.0 * self.betr * tau)
        rba = np.sqrt(ba)
        br1 = (self.betr * self.zz - tau) / 2.0
        br2 = (self.betr * self.zz + tau) / 2.0
        g1 = exf(0.0, rba * (self.betr * self.zz - tau))
        g2 = exf(self.p * self.zz, rba * (self.betr * self.zz + tau))
        
        if self.modc in (3, 4):  # third-type concentration
            g3 = dbexp(-ba * (self.betr * self.zz - tau) * (self.betr * self.zz - tau))
            rbb = np.sqrt(self.p * tau / np.pi / self.betr / 4.0)
            brp = self.betr / self.p
            return self.gamma1[0] / self.betr * (tau + (br1 + brp / 2.0) * g1 - rbb * 2.0 * (br2 + brp) * g3 + (tau / 2.0 - brp / 2.0 + br2 * br2 / brp) * g2)
        else:
            return self.gamma1[0] / self.betr * (tau + br1 * g1 - br2 * g2)

    def phi1(self, tau, level=11):
        """Calculate series in exponential BVP for equilibrium phase"""
        omom = self.a * self.b
        c = self.tpulse[0] - self.b

        phi1 = 0.0
        fn = 1.0
        for n in range(1, level):
            sumk = 0.0
            k = 1
            ck = 1.0
            fnk = fn
            fn = fn * n
            while True:
                ck = -ck
                sumk += ck * (self.tt - tau) ** (n - k) / (c ** k) / fnk
                if n == k:
                    break
                fnk = fnk / (n - k)
                k += 1
            phi1 += sumk * ((omom * tau) ** n) / fn

        return phi1

    def phi2(self, tau, level=11):
        """Calculate series in exponential BVP for nonequilibrium phase"""
        omom = self.a * self.b
        c = self.tpulse[0] - self.b

        phi2 = 0.0
        fn = 1.0
        for n in range(1, level):
            sumk = 0.0
            k = 1
            ck = 1.0
            fnk = fn
            fn = fn * n
            while True:
                ck = -ck
                sumk += ck * (self.tt - tau) ** (n - k + 1) / (c ** k) / fnk / (n - k + 1)
                if n == k:
                    break
                fnk = fnk / (n - k)
                k += 1
            phi2 += sumk * ((omom * tau) ** n) / fn

        return phi2

    def cinput(self, tau):
        """
        PURPOSE: ARBITRARY FUNCTION DEFINED BY USER
        """
        if self.nredu <= 1:
            t = tau / self.v * self.zl
        else:
            t = tau

        if t <= 9.1:
            return (-560.39 * t) + 16528
        else:
            return 0