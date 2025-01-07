import numpy as np
from scipy.special import erfc

# Converted from functions in DETCDE.FOR original Fortran code
class DetCDE:
    """
    Calculate deterministic equilibrium & nonequilibrium CDE
    """
    def DetCDE(self):
        betr = self.beta * self.r
        omegamu2 = self.omega + self.dmu2

        if self.mode % 2 == 1 or omegamu2 < self.ctol: 
            self.a = 0.0
            self.da = 0.0
            self.cx = 0.0
        else: # nonequilibrium CDE
            self.a = self.omega * self.omega / omegamu2 / betr
            self.da = self.omega * self.dmu2 / omegamu2
            self.cx = self.omega / omegamu2

        if self.mode % 2 == 1 or (self.beta >= 0.9999999 and self.omega < self.ctol) or omegamu2 < self.ctol:
            self.b = 0.0
        else:
            self.b = omegamu2 / (self.r - betr)

        if self.modb != 0:
            cbou1, cbou2 = self.bound()
        else:
            cbou1 = cbou1 = 0.0
            cbou2 = 0.0

        if self.modi != 0:
            cint1, cint2 = self.initial()
        else:
            cint1 = cint2 = 0.0

        if self.modp != 0:
            cpro1, cpro2 = self.produc()
        else:
            cpro1 = cpro2 = 0.0

        c1 = cbou1 + cint1 + cpro1
        if self.mode % 2 == 0: # nonequilibrium CDE
            c2 = cbou2 + cint2 + cpro2
        else: # equilibrium CDE
            c2 = 0.0

        return c1, c2
    
    def bound(self):
        """Boundary value problem (BVP) for equilibrium and nonequilibrium CDE"""
        if self.modb == 0 or self.tt <= self.dtmin : # intial time
            return 0.0, 0.0

        elif self.modb == 1: # Dirac Delta input
            if self.nredu <= 1:
                bmass = self.pulse[0]['conc'] * self.v / self.zl
            else:
                bmass = self.pulse[0]['conc']

            c1 = self.cc0(self.tt) * np.exp(-self.omega * self.tt / self.betar)
            if self.mode % 2 == 1: # equilibrium CDE
                c1 *= bmass
                return c1, 0.0
            else :
                ap = np.sqrt(1.0 + self.p * self.zz / 30.0)
                tmax = min(self.tt, self.betar * (self.zz + 60.0 * (1.0 + ap) / self.p))
                tmin = max(0.0, self.betar * (self.zz + 60.0 * (1.0 - ap) / self.p))

                mc = 1
                a1 = self.chebycon(self.cbj, tmin, tmax, mc)
                c1 = (c1 + a1) * bmass

                mc = 2
                c2 = self.chebycon(self.cbj, tmin, tmax, mc) * bmass

                return c1, c2

        # Multiple pulse input (modb = 2,3,4), 
        # Step input (modb == 2) and pulse input (modb == 3) are special cases  of 
        # the multiple pulse input (modb == 4)
        elif self.modb in (2, 3, 4): 
            ttt = self.tt
            c1 = c2 = 0.0
            for i in range(self.npulse):
                ttt -= self.pulse[i]['time']
                if ttt <= self.dtmin:
                    return c1, c2

                if self.mode % 2 == 1:
                    a1 = self.cc4(ttt, self.da)
                else:
                    if self.modjh == 1: # Eq.(3.21) or (3.22) based on Goldstein's J-function 
                        ap = np.sqrt(1.0 + self.p * self.zz / 30.0)
                        tmax = min(ttt, self.betar * (self.zz + 60.0 * (1.0 + ap) / self.p))
                        tmin = min(0.0, self.betar * (self.zz + 60.0 * (1.0 - ap) / self.p))

                        if self.modc not in [3, 4] and self.zz < self.ctol and ttt > 0.0:
                            a1 = 1.0
                        else :
                            mc = 1
                            a1 = self.chebycon(self.cbj, tmin, tmax, mc)
                        mc = 2
                        a2 = self.chebycon(self.cbj, tmin, tmax, mc)
                    else: # evaluate Eq.(3.23) or (3.24)
                        mc = 1
                        a11 = self.chebycon(self.cbal, 0.0, ttt, mc)
                        a1 = self.cc4(ttt, self.da) * np.exp(-self.a * ttt) + a11

                        mc = 2
                        a2 = self.chebycon(self.cbal, 0.0, ttt, mc)

                if i == 0:
                    c1 = self.pulse[i]['conc'] * a1
                    c2 = self.pulse[i]['conc'] * a2
                else:
                    c1 += (self.pulse[i]['conc'] - self.pulse[i - 1]['conc']) * a1
                    c2 += (self.pulse[i]['conc'] - self.pulse[i - 1]['conc']) * a2

            return c1, c2
        
        elif self.modb == 5: # Exponential input
            if self.mode % 2 == 1:
                a1 = self.cc4(self.tt, self.da)
                a2 = 0.0
            else:
                mc = 1
                a11 = self.chebycon(self.cbal, 0.0, self.tt, mc)
                a1 = self.cc4(self.tt, self.da) * np.exp(-self.a * self.tt) + a11

                mc = 2
                a2 = self.chebycon(self.cbal, 0.0, self.tt, mc)

            c1 = self.pulse[0]['conc'] * a1
            c2 = self.pulse[0]['conc'] * a2

            #EXPONENTIAL TERM  TPULSE(1)= lamda^b
            if abs(self.pulse[0]['time']) < self.dtmin:
                c1 += self.pulse[1]['conc'] * a1
                c2 += self.pulse[1]['conc'] * a2
                return c1, c2

            if self.mode % 2 == 1:
                dum = -self.r * self.pulse[0]['time']
                dum1 = 1.0 + 4.0 * (self.dmu1 + dum) / self.p
                mc = 1
                if dum1 < self.pmin:
                    a1 = self.chebycon(self.cbexp, 0.0, self.tt, mc)
                else:
                    a1 = np.exp(-self.pulse[0]['time'] * self.tt) * self.cc4(self.tt, dum)
            else:
                mc = 1
                a1 = self.chebycon(self.cbexp, 0.0, self.tt, mc)

                mc = 2
                a2 = self.chebycon(self.cbexp, 0.0, self.tt, mc)
                a2 = self.omega / (self.r - self.betar) * a2

            c1 += self.pulse[1]['conc'] * a1
            c2 += self.pulse[1]['conc'] * a2

            return c1, c2

        elif self.modb == 6: # Arbitrary input
            c1 = self.cheby(self.cbin1, 0.0, self.tt)
            if self.modc not in [3, 4] and self.zz < self.dzmin:
                c1 = self.cinput(self.tt)

            if self.mode % 2 == 1:
                return c1, 0.0
            else :
                c2 = self.cheby(self.cbin2, 0.0, self.tt)
                c2 = c2 * self.omega / (self.r - self.betar)

                return c1, c2
        else:
            raise ValueError("ERROR! MODB SHOULD BE 0,1,2,3,4,5,6")
      
    def initial(self):
        """Initial value problem (IVP) for equilibrium and nonequilibrium CDE"""
        mmodi = self.modi
        if mmodi == 4 and self.cini[0]['conc'] > self.ctol:
            mmodi = 1
        
        # Initial concentration for small T
        if self.tt <= self.dtmin: 
            # stepwise initial distribution (modi = 0, 1, 2)
            # Zero initial distribution (modi = 0) and constant initial distribution (modi = 1) 
            # are special cases of stepwise initial distribution (modi = 2)
            if mmodi in (0, 1, 2):
                ## match model length zz with cloest zini
                cini = min(self.cini, key=lambda cini: abs(cini['z'] - self.zz))
                c1 = cini['conc']
                c2 = cini['conc']
            elif mmodi == 3: # exponential initial distribution
                c1 = self.cini[0]['conc'] + self.cini[1]['conc'] * np.exp(-self.cini[0]['z'] * self.zz)
                c2 = c1
            elif mmodi == 4: # Dirac delta initial distribution
                c1 = c2 = 0.0
            else :
                raise ValueError("ERROR! MODI SHOULD BE 0,1,2,3,4")
            return c1, c2
            
        # prepare constants
        dg = np.exp(-(self.dmu1 + self.da) / self.betar * self.tt)
        if self.mode % 2 == 1:
            a1 = 0.0
        else:
            mc = 1
            a1 = self.chebycon(self.civp, 0.0, self.tt, mc)
            mc = 2
            a2 = self.chebycon(self.civp, 0.0, self.tt, mc)

        mcc0 = self.cc0(self.tt)
        mcc1 = self.cc1(self.tt)

        if mmodi in (0, 1, 2):
            iconc0 = self.cini[0]['conc']
            g = -iconc0 * dg * (mcc1 - 1.0)
            for i in range(1, self.nini):
                iconc0 = self.cini[i - 1]['conc']
                iconc1 = self.cini[i]['conc']
                zini = self.cini[i]['z']
                mcc2 = self.cc2(self.tt, zini)
                g += (iconc0 - iconc1) * dg * (mcc2 - 1.0)
            c1 = g * np.exp(-self.a * self.tt) + a1
            if self.mode % 2 == 0: # nonequilibrium CDE
                cini = min(self.cini, key=lambda cini: abs(cini['z'] - self.zz))
                c2 = cini['conc'] * np.exp(-self.b * self.tt) + a2
        elif mmodi == 3: # exponential initial distribution
            U1 = self.cini[0]['conc']
            U2 = self.cini[1]['conc']
            lamda = self.cini[0]['z']
            mcc3 = self.cc3(self.tt, lamda)
            c1 = (U1 * dg * (1.0 - mcc1) + U2  * dg * mcc3) * np.exp(-self.a * self.tt) + a1
            if self.mode % 2 == 0:
                c2 = (U1 + U2 * np.exp(-lamda * self.zz)) * np.exp(-self.b * self.tt) + a2
        elif mmodi == 4: # Dirac delta initial distribution
            U1 = self.cini[0]['conc']
            dmass = self.cini[1]['conc']
            zini = self.cini[1]['z']
            mcc5 = self.cc5(self.tt, zini)
            if self.nredu != 2:
                dmass /= self.zl
            if dmass < self.ctol:
                return c1, c2
            
            if self.modc <= 4 and abs(zini) < self.dzmin:
                g = dmass * dg * mcc0 * self.beta * self.r 
            else :
                g = dmass * dg * mcc5
            c1 += g * np.exp(-self.a * self.tt) + dmass * a1
            if self.mode % 2 != 1 and self.mcon != 1:
                c2 += dmass * a2
        else :
            raise ValueError("ERROR! MODI SHOULD BE 0,1,2,3,4")

        return c1, c2
        
    def produc(self):
        """Production value problem (PVP) for equilibrium and nonequilibrium CDE"""
       
        c1 = 0.0
        c2 = 0.0
        if self.tt <= self.dtmin: # initial time
            return c1, c2

        # Equilibrium CDE with constant production term
        
        # if self.mode % 2 == 1 and (self.npro1 == 1 or self.modp == 1):
        #     if self.modp1 == 0: # Eq.(2.32) 
        #         if (self.omega + self.dmu1) < self.ctol:
        #             c1 = self.prod0(self.tt)
        #             return c1, c2
        #         else:
        #             c1 = self.prodval1[0]['gamma'] / self.dmu1 * (1 - np.exp(-(self.dmu1 * self.tt) / self.betar) * (1 - self.cc1(self.tt)) - self.cc4(self.tt, self.da))
        #             return c1, c2

        # Eq.(2.33) or (2.34) (current setting)
        c1 = self.cheby(self.c1pro, 0.0, self.tt)

        if self.mode % 2 == 1:
            return c1, c2

        # Nonequilibrium CDE phase2 concentration
        omegamu2 = self.omega + self.dmu2
        tconv = self.tt / (self.r - self.betar)
        if omegamu2 < self.ctol:
            if self.modp in (0, 1, 2):
                if self.npro2 == 1:
                    gamma2 = self.prodval2[0]['gamma']
                    c2 = gamma2 * tconv
                    return c1, c2
                else:
                    prod = min(self.prodval2, key=lambda prod: abs(prod['zpro'] - self.zz))
                    gamma2 = prod['gamma']
                    c2 = gamma2 * tconv
                    return c1, c2
            elif self.modp == 3:
                gamma21 = self.prodval2[0]['gamma']
                gamma22 = self.prodval2[1]['gamma']
                zpro2 = self.prodval2[0]['zpro']
                item1 = (gamma21 + gamma22 * np.exp(-zpro2 * self.zz))
                item2 = tconv
                c2 = item1 * item2
                return c1, c2
            else:
                raise ValueError("ERROR! MODP SHOULD BE 0,1,2,3")
        else:
            a2 = self.cheby(self.c2pro, 0.0, self.tt)
            a3 = omegamu2 * (1.0 - np.exp(-omegamu2 * tconv))

            if self.modp in (0, 1, 2):
                if self.npro2 == 1:
                    gamma2 = self.prodval2[0]['gamma']
                    c2 = gamma2 / a3 + a2
                    return c1, c2
                else:
                    prod = min(self.prodval2, key=lambda prod: abs(prod['zpro'] - self.zz))
                    gamma2 = prod['gamma']
                    c2 = gamma2 / a3 + a2
                    return c1, c2
            elif self.modp == 3:
                gamma21 = self.prodval2[0]['gamma']
                gamma22 = self.prodval2[1]['gamma']
                zpro2 = self.prodval2[0]['zpro']
                gamma = gamma21 + gamma22 * np.exp(-zpro2 * self.zz)
                c2 = gamma / a3 + a2
                return c1, c2
            else:
                raise ValueError("ERROR! MODP SHOULD BE 0,1,2,3")

    def cheby(self, func, a, b):
        """Perform integration of f(x) between a and b using M-point Gauss-Chebyshev quadrature formula"""
        mmm = self.mm
        if self.icheb != 1:
            sum_val = 0.0
            for i in range(1, mmm + 1):
                z1 = np.cos((2 * (i - 1) + 1) * np.pi / (2 * mmm))
                x1 = (z1 * (b - a) + b + a) / 2.0
                sum_val += func(x1) * np.sqrt(1.0 - z1 * z1)
            area = (b - a) * np.pi * sum_val / (2 * mmm)
            return area

        area1 = 0.0
        for _ in range(1, self.cheby_level):
            sum_val = 0.0
            for i in range(1, mmm + 1):
                z1 = np.cos((2 * (i - 1) + 1) * np.pi / (2 * mmm))
                x1 = (z1 * (b - a) + b + a) / 2.0
                sum_val += func(x1) * np.sqrt(1.0 - z1 * z1)
            area = (b - a) * np.pi * sum_val / (2 * mmm)
            error = abs(area - area1) / area
            if abs(area) < self.ctol or error < self.stopch or mmm>= self.mmax:
                return area
            area1 = area
            mmm*= 2

        return area

    def cheby2(self, func, a, b):
        """Perform integration of f(x) between a and b using M-point Gauss-Chebyshev quadrature formula"""

        mmm = self.mm
        if self.icheb != 1:
            sum_val = 0.0
            for i in range(1, mmm + 1):
                z1 = np.cos((2 * (i - 1) + 1) * np.pi / (2 * mmm))
                x1 = (z1 * (b - a) + b + a) / 2.0
                sum_val += func(x1) * np.sqrt(1.0 - z1 * z1)
            area = (b - a) * np.pi * sum_val / (2 * mmm)
            return area

        area1 = 0.0
        for j in range(1, self.cheby_level):
            sum_val = 0.0
            for i in range(1, mmm + 1):
                z1 = np.cos((2 * (i - 1) + 1) * np.pi / (2 * mmm))
                x1 = (z1 * (b - a) + b + a) / 2.0
                sum_val += func(x1) * np.sqrt(1.0 - z1 * z1)
            area = (b - a) * np.pi * sum_val / (2 * mmm)
            error = abs(area - area1) / area
            if abs(area) < self.ctol or error < self.stopch or mmm >= self.mmax:
                return area
            area1 = area
            mmm *= 2

        return area

    def chebycon(self, func, area, a, b, mc):
        """Perform integration of f(x, mc) between a and b using M-point Gauss-Chebyshev quadrature formula"""
        mmm = self.mm
        if self.icheb != 1:
            sum_val = 0.0
            for i in range(1, mmm + 1):
                z1 = np.cos((2 * (i - 1) + 1) * np.pi / (2 * mmm))
                x1 = (z1 * (b - a) + b + a) / 2.0
                sum_val += func(x1, mc) * np.sqrt(1.0 - z1 * z1)
            area = (b - a) * np.pi * sum_val / (2 * mmm)
            return area

        area1 = 0.0
        for j in range(1, 11):
            sum_val = 0.0
            for i in range(1, mmm + 1):
                z1 = np.cos((2 * (i - 1) + 1) * np.pi / (2 * mmm))
                x1 = (z1 * (b - a) + b + a) / 2.0
                sum_val += func(x1, mc) * np.sqrt(1.0 - z1 * z1)
            area = (b - a) * np.pi * sum_val / (2 * mmm)
            error = abs(area - area1) / area
            if abs(area) < self.ctol or error < self.stopch or mmm >= self.mmax:
                return area
            area1 = area
            mmm *= 2

        return area

    def ctran(self, tau, mc):
        """Calculate argument in integral for delta input (transfer function model)"""
        g = self.cc0(tau)
        if g < self.ctol:
            return 0.0

        aa = self.omega * tau / self.betar
        at = self.a * tau
        bt = self.b * (self.tt - tau)
        xii = 2.0 * np.sqrt(at * bt)
        self.beta = self.betar / self.r

        if mc == 1:
            cbi1 = np.sqrt(tau / ((1 - self.beta) * self.beta * (self.tt - tau))) * self.omega / self.r
            return g * self.expbi1(xii, -aa - bt) * cbi1
        else:
            return self.omega / (self.r - self.betar) * g * self.expbi0(xii, -aa - bt)

    def cbj(self, tau, mc):
        """Calculate argument in integral for step input (solution using Goldstein's J-function)"""
        g = self.cc0(tau)
        if g < self.ctol:
            return 0.0

        at = self.a * tau
        bt = self.b * (self.tt - tau)
        if mc == 1:
            return g * self.gold(at, bt) * np.exp(-at * self.dmu2 / self.omega)
        else:
            if self.modc not in [3, 4]:
                if self.zz < self.dzmin and self.tt > 0.0:
                    return self.omega / (self.r - self.betar) * np.exp(-bt)
            return self.cx * g * (1.0 - self.gold(bt, at)) * np.exp(-at * self.dmu2 / self.omega)

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
            cbi2 = self.expbi0(xii, -at - bt) * cbi0 + self.expbi1(xii, -at - bt) * cbi1
            return self.omega / self.r * g * cbi2
        else:
            cbi1 = np.sqrt((1 - self.beta) * (self.tt - tau) / tau / self.beta) * self.cx
            cbi2 = self.expbi0(xii, -at - bt) + self.expbi1(xii, -at - bt) * cbi1
            return self.omega / (self.r - self.betar) * g * cbi2

    def cbexp(self, tau, mc):
        """Calculate argument in integral for exponential input"""
        g = self.cc0(tau)
        if g < self.ctol:
            return 0.0

        tpulse = self.pulse[0]['time']
        if self.mode % 2 == 1:
            return g * np.exp(-tpulse * (self.tt - tau))

        c1 = -self.omega * tau / self.betar + self.b * tau
        c2 = -self.omega * tau / self.betar + tpulse * tau
        c3 = tpulse - self.b

        if mc == 1:
            mexp1 = np.exp(-tpulse * self.tt + c2 - self.a * self.b / c3 * tau)
            mexp2 = np.exp(-self.b * self.tt + c1)
            mphi1 = self.phi1(tau)
            return g * (mexp1 - mexp2 * mphi1)
        else:
            mexp1 = np.exp(-self.b * self.tt + c1 - self.a * self.b * tau / c3)
            mexp2 = np.exp(-tpulse * self.tt + c2 - self.a * self.b * tau / c3)
            mexp3 = np.exp(-self.b * self.tt + c1)
            mphi2 = self.phi2(tau)
            return g * (mexp1/ c3 - mexp2/ c3 - mexp3* mphi2)

    def cbin1(self, tau):
        """Calculate argument in integral for arbitrary input given in function CINPUT"""
        g = self.cc0(tau)
        if g < self.ctol and self.mode % 2 == 1:
            return 0.0

        c1 = g * np.exp(-self.omega * tau / self.betar)
        if self.mode % 2 != 1:
            mc = 1
            a1 = self.chebycon(self.ctran, 0.0, tau, mc)
        return (c1 + a1) * self.cinput(self.tt - tau)

    def cbin2(self, tau):
        """Calculate argument in integral for arbitrary input given in function CINPUT for nonequilibrium phase"""

        a1 = self.cheby2(self.cbin1, 0.0, tau)
        return a1 * np.exp(-self.b * (self.tt - tau))

    def civp(self, tau, mc):
        """Calculate argument in initial value problem"""
        self.beta = self.betar / self.r
        dg = np.exp(-(self.dmu1 + self.da) / self.betar * tau)

        if self.initi['modi'] == 1:
            for i in range(self.initi['nini']):
                if i == 0:
                    g = -self.initi['cini'][i] * dg * (self.cc1(tau) - 1.0)
                else:
                    g += (self.initi['cini'][i - 1] - self.initi['cini'][i]) * dg * (self.cc2(tau, self.initi['zini'][i]) - 1.0)
        elif self.initi['modi'] == 2:
            g = self.initi['cini'][0] * dg * (1.0 - self.cc1(tau)) + self.initi['cini'][1] * dg * self.cc3(tau, self.initi['zini'][0])
        elif self.initi['modi'] == 3:
            if self.modc <= 4 and abs(self.initi['zini'][1]) < self.dzmin:
                g = dg * self.cc0(tau) * self.beta * self.r
            else:
                g = dg * self.cc5(tau, self.initi['zini'][1])

        at = self.a * tau
        bt = self.b * (self.tt - tau)
        xii = 2.0 * np.sqrt(at * bt)

        if mc == 1:
            cbi1 = np.sqrt(tau / ((1 - self.beta) * self.beta * (self.tt - tau))) * self.omega / self.r
            civp = self.expbi0(xii, -at - bt) * self.omega / self.betar + self.expbi1(xii, -at - bt) * cbi1
            civp = g * civp
        else:
            cbi1 = np.sqrt((1 - self.beta) * (self.tt - tau) / self.beta / tau)
            civp = self.expbi0(xii, -at - bt) + self.expbi1(xii, -at - bt) * cbi1
            civp = self.omega / (self.r - self.betar) * g * civp

        return civp

    def c1pro(self, tau):
        """Calculate argument in production term for equilibrium concentration"""
        c1pro = 0.0
        g = 0.0
        h = 0.0
        dg = np.exp(-(self.dmu1 + self.da) / self.betar * tau)

        if self.modp == 1:
            if self.npro1 == 0:
                return c1pro
            for i in range(self.npro1):
                if i == 0:
                    g = -self.prodval1[0]['gamma'] / self.betar * dg * (self.cc1(tau) - 1.0)
                else:
                    g += (self.prodval1[i - 1]['gamma'] - self.prodval1[i]['gamma']) / self.betar * dg * (self.cc2(tau, self.prodval1[i]['zpro']) - 1.0)
            if self.mode % 2 != 1:
                if self.npro2 == 0:
                    return c1pro
                for i in range(self.npro2):
                    if i == 0:
                        h = -self.cx * self.prodval2[0]['gamma'] / self.betar * dg * (self.cc1(tau) - 1.0)
                    else:
                        h += self.cx * (self.prodval2[i - 1]['gamma'] - self.prodval2[i]['gamma']) / self.betar * dg * (self.cc2(tau, self.prodval2[i]['zpro']) - 1.0)
                g += h
        elif self.modp == 2:
            g = (self.prodval1[0]['gamma'] * (1.0 - self.cc1(tau)) + self.prodval1[1]['gamma'] * self.cc3(tau, self.prodval1[0]['zpro'])) * dg / self.betar
            if self.mode % 2 != 1:
                h = (self.prodval2[0]['gamma'] * (1.0 - self.cc1(tau)) + self.prodval2[1]['gamma'] * self.cc3(tau, self.prodval2[0]['zpro'])) * dg / self.betar * self.cx
                g += h

        if g < self.ctol:
            return c1pro

        if self.mode % 2 == 1:
            return g

        at = self.a * tau
        bt = self.b * (self.tt - tau)
        xii = 2.0 * np.sqrt(at * bt)
        c1pro = g * self.gold(at, bt) - h * self.expbi0(xii, -at - bt)

        return c1pro

    def c2pro(self, tau):
        """Calculate argument in production term for nonequilibrium concentration"""
        c2pro = 0.0
        g = 0.0
        h = 0.0
        dg = np.exp(-(self.dmu1 + self.da) / self.betar * tau)

        if self.modp == 1:
            if self.npro1 == 0:
                return c2pro
            for i in range(self.npro1):
                if i == 0:
                    g = -self.prodval1[0]['gamma'] / self.betar * dg * (self.cc1(tau) - 1.0)
                else:
                    g += (self.prodval1[i - 1]['gamma'] - self.prodval1[i]['gamma']) / self.betar * dg * (self.cc2(tau, self.prodval1[i]['zpro']) - 1.0)
            if self.npro2 == 0:
                return c2pro
            for i in range(self.npro2):
                if i == 0:
                    h = -self.cx * self.prodval2[0] / self.betar * dg * (self.cc1(tau) - 1.0)
                else:
                    h += self.cx * (self.prodval2[i - 1] - self.prodval2[i]) / self.betar * dg * (self.cc2(tau, self.prodval2[i]['zpro']) - 1.0)
            g += h
        elif self.modp == 2:
            g = (self.prodval1[0]['gamma'] * (1.0 - self.cc1(tau)) + self.prodval1[1]['gamma'] * self.cc3(tau, self.prodval1[0]['zpro'])) * dg / self.betar
            h = (self.prodval2[0]['gamma'] * (1.0 - self.cc1(tau)) + self.prodval2[1]['gamma'] * self.cc3(tau, self.prodval2[0]['zpro'])) * dg / self.betar * self.cx
            g += h

        if g < self.ctol:
            return c2pro

        at = self.a * tau
        bt = self.b * (self.tt - tau)
        self.beta = self.betar / self.r
        xii = 2.0 * np.sqrt(at * bt)
        cbi1 = np.sqrt(self.beta / (1 - self.beta) * (self.tt - tau) / tau)
        c2pro = self.cx * g * (1.0 - self.gold(bt, at)) - cbi1 * h * self.expbi1(xii, -at - bt)

        return c2pro

    def cc0(self, tau):
        """Calculate solutions for delta input travel time distribution for equilibrium CDE"""
        dg = np.exp(-self.dmu1 / self.betar * tau)
        g1 = np.exp(self.p * (self.betar * self.zz - tau) * (tau - self.betar * self.zz) / (4.0 * self.betar * tau))
        
        if self.modc not in [3, 4]:
            g2 = np.sqrt(self.p / (self.betar * tau))
            return dg * (0.56419 * g2 * g1 - self.p / (2.0 * self.betar) * self.exf(self.p * self.zz, g2 / 2.0 * (self.betar * self.zz + tau)))
        else:
            return dg * (self.zz / tau) * np.sqrt(self.p * self.betar / (12.5664 * tau)) * g1
        
    def cc1(self, tau):
        """Calculate equilibrium solutions for step input"""
        ba = self.p / (4.0 * self.betar * tau)
        rba = np.sqrt(ba)
        rbb = np.sqrt(self.p * tau / np.pi / self.betar)
        g1 = self.exf(0.0, rba * (self.betar * self.zz - tau))
        g2 = self.exf(self.p * self.zz, rba * (self.betar * self.zz + tau))
        
        if self.modc not in [3, 4]:
            g3 = np.exp(-ba * (self.betar * self.zz - tau) * (self.betar * self.zz - tau))
            return g1 / 2.0 + rbb * g3 - (1.0 + self.p * self.zz + self.p * tau / self.betar) * g2 / 2.0
        else:
            return (g1 + g2) / 2.0

    def cc2(self, tau, z1):
        """Calculate argument for stepwise initial and production profiles"""
        ba = self.p / (4.0 * self.betar * tau)
        rba = np.sqrt(ba)
        rbb = np.sqrt(self.p * tau / np.pi / self.betar)
        g1 = self.exf(0.0, rba * (self.betar * (self.zz - z1) - tau))
        g2 = self.exf(self.p * self.zz, rba * (self.betar * (self.zz + z1) + tau))
        
        if self.modc in [3, 4]:
            g3 = np.exp(self.p * self.zz - ba * (self.betar * (self.zz + z1) + tau) * (self.betar * (self.zz + z1) + tau))
            return g1 / 2.0 + rbb * g3 - (1.0 + self.p * (self.zz + z1) + self.p * tau / self.betar) * g2 / 2.0
        elif self.modc in [5, 6]:
            return (g1 + g2) / 2.0
        else:
            rbc = np.sqrt(self.betar / np.pi / self.p / tau) / 2.0
            a1 = rbc * np.exp(-ba * (self.betar * (self.zz - z1) - tau) * (self.betar * (self.zz - z1) - tau))
            g3 = np.exp(self.p * self.zz  - ba * (self.betar * (self.zz + z1) + tau) * (self.betar * (self.zz + z1) + tau))
            a2 = rbc * g3
            return (g1 + g2) / 2.0 + a1 - a2

    def cc3(self, tau, z1):
        """Calculate argument for exponential initial and production profiles"""
        ba = self.p / (4.0 * self.betar * tau)
        rba = np.sqrt(ba)
        rbb = np.sqrt(self.p * tau / np.pi / self.betar)
        a1 = np.exp(z1 * z1 * tau / self.betar / self.p + z1 * tau / self.betar - z1 * self.zz)
        g1 = self.exf(0.0, rba * (self.betar * self.zz - (1.0 + 2.0 * z1 / self.p) * tau))
        g2 = self.exf(self.p * self.zz + 2.0 * z1 * self.zz, rba * (self.betar * self.zz + (1.0 + 2.0 * z1 / self.p) * tau))
        
        if self.modc in [3, 4]:
            g3 = self.exf(self.p * self.zz, rba * (self.betar * self.zz + tau))
            return a1 * (1.0 - g1 / 2.0 + (1 + self.p / z1) * g2 / 2.0) - self.p / z1 / 2.0 * g3
        elif self.modc in [1, 2]:
            return (1.0 + z1 / self.p) * a1 * (2.0 - g1 - g2) / 2.0
        elif self.modc in [5, 6]:
            return a1 * (2.0 - g1 - g2) / 2.0

    def cc4(self, tau, dum):
        """Argument for step input or constant production"""
        u = np.sqrt(1.0 + 4.0 * (self.dmu1 + dum) / self.p)
        ba = self.p / (4.0 * self.betar * tau)
        rba = np.sqrt(ba)
        rbb = np.sqrt(self.p * tau / np.pi / self.betar)
        g1 = self.exf(self.p * (1.0 - u) * self.zz / 2.0, rba * (self.betar * self.zz - u * tau))
        g2 = self.exf(self.p * (1.0 + u) * self.zz / 2.0, rba * (self.betar * self.zz + u * tau))
        
        if self.modc not in [3, 4]:
            if abs(self.dmu1 + dum) < self.ctol:
                g3 = np.exp(-ba * (self.betar * self.zz - tau) * (self.betar * self.zz - tau))
                return g1 / 2.0 + rbb * g3 - (1.0 + self.p * self.zz + self.p * tau / self.betar) * g2 / 2.0
            else:
                g3 = self.exf(self.p * self.zz - (self.dmu1 + dum) * tau / self.betar, rba * (self.betar * self.zz + tau))
                return 1 / (1 + u) * g1 + 1 / (1 - u) * g2 + self.p / 2.0 / (self.dmu1 + dum) * g3
        else:
            return (g1 + g2) / 2.0

    def cc5(self, tau, z1):
        """Calculate argument for delta initial condition or general IVP & PVP"""
        ba = self.betar * self.p / (4.0 * np.pi * tau)
        rba = np.sqrt(ba)
        bb = self.p / 4.0 / self.betar / tau
        g1 = np.exp(-bb * (self.betar * (z1 - self.zz) + tau) * (self.betar * (z1 - self.zz) + tau))
        g2 = np.exp(self.p * self.zz - bb * (self.betar * (z1 + self.zz) + tau) * (self.betar * (z1 + self.zz) + tau))
        
        if self.modc in [3, 4]:
            rbb = np.sqrt(bb)
            g3 = self.exf(self.p * self.zz, rbb * (self.betar * (z1 + self.zz) + tau))
            cc5 = (g1 + g2) * rba - self.p / 2.0 * g3
        elif self.modc in [5, 6]:
            cc5 = (g1 - g2) * rba
        else:
            bc = 1.0 - (self.betar * (z1 - self.zz) * tau) / 2.0 / tau
            bd = 1.0 - (self.betar * (z1 + self.zz) * tau) / 2.0 / tau
            cc5 = (bc * g1 - bd * g2) * rba
        
        return cc5

    def prod0(self, tau):
        """Calculate analytical solutions of constant production term for equilibrium CDE in case of dmu1=0"""
        ba = self.p / (4.0 * self.betar * tau)
        rba = np.sqrt(ba)
        br1 = (self.betar * self.zz - tau) / 2.0
        br2 = (self.betar * self.zz + tau) / 2.0
        g1 = self.exf(0.0, rba * (self.betar * self.zz - tau))
        g2 = self.exf(self.p * self.zz, rba * (self.betar * self.zz + tau))
        
        if self.modc not in [3, 4]:
            g3 = np.exp(-ba * (self.betar * self.zz - tau) * (self.betar * self.zz - tau))
            rbb = np.sqrt(self.p * tau / np.pi / self.betar / 4.0)
            brp = self.betar / self.p
            return self.prodval1[0]['gamma'] / self.betar * (tau + (br1 + brp / 2.0) * g1 - rbb * 2.0 * (br2 + brp) * g3 + (tau / 2.0 - brp / 2.0 + br2 * br2 / brp) * g2)
        else:
            return self.prodval1[0]['gamma'] / self.betar * (tau + br1 * g1 - br2 * g2)

    def phi1(self, tau):
        """Calculate series in exponential BVP for equilibrium phase"""
        omom = self.a * self.b
        c = self.pulse[0]['time'] - self.b

        phi1 = 0.0
        fn = 1.0
        for n in range(1, self.phi_level):
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

    def phi2(self, tau):
        """Calculate series in exponential BVP for nonequilibrium phase"""
        omom = self.a * self.b
        c = self.pulse[0]['time'] - self.b

        phi2 = 0.0
        fn = 1.0
        for n in range(1, self.phi_level):
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
    
    # Converted from EXF function in FUNC2.FOR original Fortran code
    @staticmethod
    def exf(a, b):
        """Calculate EXP(A) ERFC(B)"""
        if abs(a) > 170 and b <= 0:
            return 0.0
        c = a - b * b
        if abs(c) > 170 and b >= 0:
            return 0.0
        if c < -170:
            return 0.0
        
        # Modified Use SciPy's erfc function and numpy's exp
        exf = np.exp(a) * erfc(b)
        
        # If b is negative, adjust the result
        if b < 0.0:
            exf = 2.0 * np.exp(a) - exf
        
        return exf

    # Converted from EXPBI0 function in FUNC2.FOR original Fortran code
    @staticmethod
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
    @staticmethod
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

    @staticmethod
    # Converted from GOLD function in FUNC2.FOR original Fortran code
    def gold(x, y):
        """Calculate Goldstein's J-function J(X,Y)"""
        bf=0.0
        e = 2.0 * np.sqrt(max(1.0e-35, x * y))
        z = x + y - e
        if z > 17.0:
            return 0.0
        if e > 1.0e-15:
            a = max(x, y)
            b = min(x, y)
            rt = 11.0 + 2.0 * b + 0.3 * a
            if rt > 25:
                da = np.sqrt(a)
                db = np.sqrt(b)
                p = 3.75 / e
                b0 = (0.3989423 + p * (0.01328592 + p * (0.00225319 - p * (0.00157565 - p * (0.00916281 - p * (0.02057706 - p * (0.02635537 - p * (0.01647633 - 0.00392377 * p)))))))) / np.sqrt(e)
                bf = b0 * DetCDE.exf(-z, 0.0)
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
                gold = gxy * DetCDE.exf(-x - y, 0.0)
        else:
            gold = np.exp(-x)
        
        if x < y:
            gold = 1.0 + bf - gold
        return gold

    def init_detcde(self):
        """
        Initialize deterministic model parameters before computation
        """

        self.d = self.parms.loc[self.curp,'D'] # DISPERSIVITY
        self.r = self.parms.loc[self.curp,'R'] # RETARDATION FACTOR
        self.dmu1 = self.parms.loc[self.curp,'mu1'] # DEGRADATION RATE
        if self.nredu <= 1:
            self.dmu1 *= self.zl / self.v
        
        if self.mode % 2 == 2:
            self.beta = self.parms.loc[self.curp,'beta'] # nonequilibrium partitioning coefficient 
            self.omega = self.parms.loc[self.curp,'omega'] # nonequilibrium mass transfer coefficient
            self.dmu2 = self.parms.loc[self.curp,'mu2'] # DEGRADATION RATE IN IMMOBILE PHASE
            #ONE-SITE CHEMICAL NONEUILIBRIUM MODEL
            if self.mneq == 1 and self.mit >= 1:
                self.beta = 1.0 / self.r
                self.parms.loc[self.curp,'beta'] = self.beta
            if self.mit >= 1 and self.mdeg >= 1:
                if self.mdeg == 1:
                    self.dmu2 = (1.0 - self.beta) / self.beta * self.dmu1
                    self.parms.loc[self.curp,'mu2'] = self.dmu2
                elif self.mdeg == 2 and (self.mneq == 0 or self.mneq == 3):
                    self.dmu2 = self.phiim / self.phim * self.dmu1
                    self.parms.loc[self.curp,'mu2'] = self.dmu2
                elif self.mdeg == 3 and (self.mneq == 0 or self.mneq == 3):
                    self.dmu2 = self.dmu1 / (self.beta * self.r - self.phim) * ((1.0 - self.beta) * self.r - self.phiim)
                    self.parms.loc[self.curp,'mu2'] = self.dmu2
                elif self.mdeg == 3 and self.mneq == 2:
                    self.dmu2 = self.dmu1 / (self.beta * self.r - 1.0) * (1.0 - self.beta) * self.r
                    self.parms.loc[self.curp,'mu2'] = self.dmu2
                elif self.mdeg == 3 and self.mneq == 1:
                    self.dmu1 = 0.0
                    self.parms.loc[self.curp,'mu1'] = self.dmu1
                elif self.mdeg == 2 and (self.mneq == 2 or self.mneq == 1):
                    self.dmu2 = 0.0
                    self.parms.loc[self.curp,'mu2'] = self.dmu2
            if self.inverse == 1 and (self.modc != 4 and self.modc != 6):
                self.mcon = 1
        else:
            self.beta = 1.0
            self.omega = 0.0
            self.dmu2 = 0.0
            
        self.p = self.v * self.zl / self.d # PECLET NUMBER
        self.betar = self.r 
        if self.nredu <= 1:
            self.dmu1 *= self.zl / self.v

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
