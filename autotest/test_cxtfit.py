from cxtfit import CXTfit
from cxtfit import CXTsim
import pandas as pd
import matplotlib.pyplot as plt

def test_fig4_5(input_path):
    sims = CXTfit.load(f'{input_path}/FIG4-5.IN',verbose = False)
    sims.run(verbose=False)

    sims.simcases[0].plot_profile()
    sims.simcases[1].plot_profile()

    bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
    binit = [50.,  20.,   0.0,   0.0,    0.5,    0.0,    0.5,    0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] =[0] * len(bname)

    nz = 101
    dz = 2.0
    obsdata = pd.DataFrame([{'t': 1.0, 'z': z*dz} for z in range(nz)])

    pulse = [{'conc': 1.0,'time': 0.0}]
    bvp = CXTsim(mode=3, modc=3, parms=parms, modb=1, rhoth = 4.0, pulse=pulse, obsdata=obsdata)
    bvp.run()
    bvp.plot_profile()

    cini = [{'conc':0.0,'z':0.0}, {'conc':50.0,'z':0.0}]
    ivp = CXTsim(mode=3, modc=3, parms=parms, rhoth = 4.0, modi = 4, cini=cini, obsdata=obsdata)
    ivp.run()
    ivp.plot_profile()


def test_fig4_7(input_path):
    sims = CXTfit.load(f'{input_path}/FIG4-7.IN',verbose = False)
    sims.run(verbose=False)

    sims.simcases[0].plot_profile()
    sims.simcases[1].plot_profile()

    bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
    binit = [50.,  20.,   0.0,   0.0,    0.5,    0.0,    0.5,    0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] =[0] * len(bname)

    nz = 101
    dz = 4.0
    obsdata = pd.DataFrame([{'t': 3.0, 'z': z*dz} for z in range(nz)])

    pulse =[{'conc':1.0,'time':0.0},
            {'conc':0.0, 'time':1.0}]
    
    Tconst = CXTsim(mode=3, modc=3, parms=parms, modb=3, pulse=pulse, 
                    obsdata=obsdata, rhoth = 4.0)
    Tconst.run()
    Tconst.plot_profile()

    Tvari = CXTsim(mode=3, modc=4, parms=parms, modb=3, pulse=pulse, massst =1, 
                   obsdata=obsdata, rhoth = 4.0)
    Tvari.run()
    Tvari.plot_profile()

def test_fig7_1(input_path):
    sims = CXTfit.load(f'{input_path}/FIG7-1.IN',verbose = False)
    sims.run(verbose=False)

    for simcase in sims.simcases:
        simcase.plot_profile()

    bname = ['V', 'D', 'R', 'mu1']
    binit = [25.0,37.5,3.0,0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] =[0] * len(bname)

    nz = 101
    dz = 1.0
    obsdata = pd.DataFrame([{'t': 7.5, 'z': z*dz} for z in range(nz)])

    pulse =[{'conc':1.0,'time':0.0},
            {'conc':0.0, 'time':5.0}]
    
    prodval1=[{'gamma':0.5,'zpro':0.0}]

    Mu0 = CXTsim(mode=1, modc=3, parms=parms, modb=3, pulse=pulse, 
                 modp=1, prodval1=prodval1,
                 obsdata=obsdata)
    Mu0.run()
    Mu0.plot_profile()

    parms.loc['binit','mu1'] = 0.25
    Mu025 = CXTsim(mode=1, modc=3, parms=parms, modb=3, pulse=pulse, 
                 modp=1, prodval1=prodval1,
                 obsdata=obsdata)
    Mu025.run()
    Mu025.plot_profile()

    parms.loc['binit','mu1'] = 0.5
    Mu05 = CXTsim(mode=1, modc=3, parms=parms, modb=3, pulse=pulse, 
                 modp=1, prodval1=prodval1,
                 obsdata=obsdata)
    Mu05.run()
    Mu05.plot_profile()

    parms.loc['binit','mu1'] = 1.0
    Mu1 = CXTsim(mode=1, modc=3, parms=parms, modb=3, pulse=pulse, 
                 modp=1, prodval1=prodval1,
                 obsdata=obsdata)
    Mu1.run()
    Mu1.plot_profile()

def test_fig7_2(input_path):
    sims = CXTfit.load(f'{input_path}/FIG7-2A.IN',verbose = False)
    sims.run(verbose=False)

    fig,ax = plt.subplots()

    for simcase in sims.simcases:
        simcase.plot_profile(ax=ax)

    sims = CXTfit.load(f'{input_path}/FIG7-2B.IN',verbose = False)
    sims.run(verbose=False)

    for simcase in sims.simcases:
        simcase.plot_profile()

    bname = ['V', 'D', 'R', 'mu1']
    binit = [50.0,1250.0,1.0,0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] =[0] * len(bname)

    nz = 101
    dz = 0.025
    obsdata = pd.DataFrame([{'t': 0.05, 'z': z*dz} for z in range(nz)])

    cini = [{'conc':0.0,'z':0.0}, {'conc':1.0,'z':0.5}, {'conc':0.0,'z':1.0}]

    Modc1 = CXTsim(mode=1, nredu =2, modc=1, parms=parms, zl=50.0,
                 modi=2, cini=cini,
                 obsdata=obsdata)
    Modc1.run()
    Modc1.plot_profile()

    Modc3 = CXTsim(mode=1, nredu =2, modc=3, parms=parms, zl=50.0,
                 modi=2, cini=cini,
                 obsdata=obsdata)
    Modc3.run()
    Modc3.plot_profile()

    parms.loc['binit','D'] = 250.0
    Modc1 = CXTsim(mode=1, nredu =2, modc=1, parms=parms, zl=50.0,
                 modi=2, cini=cini,
                 obsdata=obsdata)
    Modc1.run()
    Modc1.plot_profile()    

    Modc3 = CXTsim(mode=1, nredu =2, modc=3, parms=parms, zl=50.0,
                 modi=2, cini=cini,
                 obsdata=obsdata)
    Modc3.run()
    Modc3.plot_profile()

def test_fig7_3(input_path):
    sims = CXTfit.load(f'{input_path}/FIG7-3A.IN',verbose = False)
    sims.run(verbose=False)

    fig,ax = plt.subplots()
    for simcase in sims.simcases:
        simcase.plot_profile(ax=ax)

    sims = CXTfit.load(f'{input_path}/FIG7-3B.IN',verbose = False)
    sims.run(verbose=False)

    for simcase in sims.simcases:
        simcase.plot_profile()

    bname = ['V', 'D', 'R', 'mu1']
    binit = [3.0,1.0,1.0,0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] =[1,1,0,0]
    parms.loc['bmin'] =[0.01,0.01,999,999]
    parms.loc['bmax'] =[100.0,100.0,999,999]

    pulse =[{'conc':1.0,'time':0.0}]

    obsdata = pd.DataFrame([
        (2.52, 0.0000), (2.68, 0.0002), (2.93, 0.0005), (3.18, 0.0013), (3.35, 0.0027),
        (3.43, 0.0040), (3.60, 0.0141), (3.68, 0.0258), (3.77, 0.0375), (3.85, 0.0641),
        (3.93, 0.0956), (4.02, 0.1418), (4.10, 0.1920), (4.18, 0.2560), (4.27, 0.3258),
        (4.35, 0.3930), (4.43, 0.4620), (4.52, 0.5306), (4.60, 0.5986), (4.68, 0.6626),
        (4.77, 0.7165), (4.85, 0.7689), (4.93, 0.8108), (5.02, 0.8497), (5.10, 0.8785),
        (5.18, 0.9034), (5.27, 0.9241), (5.35, 0.9404), (5.43, 0.9554), (5.60, 0.9674),
        (5.77, 0.9821), (5.93, 0.9885), (6.10, 0.9938), (6.43, 0.9961), (6.77, 1.0000)], 
        columns=['t', 'cobs'])
    obsdata['z'] = 11.0

    fig7_3a1 = CXTsim(inverse=1, mode=1, modc=3, parms=parms, modb=2, pulse=pulse,
                      mit=150, ilmt=1, obsdata=obsdata)
    fig7_3a1.run()
    fig7_3a1.plot_btc()

    obsdata =pd.DataFrame([
        (5.10, 0.0001), (5.43, 0.0011), (5.68, 0.0059), (5.77, 0.0141), (5.93, 0.0351),
        (6.02, 0.0541), (6.10, 0.0783), (6.18, 0.1123), (6.27, 0.1521), (6.35, 0.1990),
        (6.43, 0.2542), (6.52, 0.3170), (6.60, 0.3792), (6.68, 0.4455), (6.77, 0.5114),
        (6.85, 0.5740), (6.93, 0.6368), (7.02, 0.6917), (7.10, 0.7396), (7.18, 0.7853),
        (7.27, 0.8226), (7.35, 0.8562), (7.43, 0.8807), (7.52, 0.9012), (7.60, 0.9203),
        (7.77, 0.9470), (7.93, 0.9612), (8.10, 0.9748), (8.43, 0.9867), (8.68, 0.9916),
        (9.18, 0.9989), (9.43, 0.9992), (9.52, 0.9995), (9.77, 0.9998), (9.93, 1.0000)],
        columns=['t', 'cobs'])
    obsdata['z'] = 17.0
    fig7_3a2 = CXTsim(inverse=1, mode=1, modc=3, parms=parms, modb=2, pulse=pulse,
                      mit=150, ilmt=1, obsdata=obsdata)
    fig7_3a2.run()
    fig7_3a2.plot_btc()

    obsdata = pd.DataFrame([
        (7.10, 0.0001), (7.35, 0.0024), (7.52, 0.0049), (7.77, 0.0061), (8.02, 0.0129),
        (8.10, 0.0202), (8.18, 0.0292), (8.27, 0.0396), (8.35, 0.0600), (8.43, 0.0770),
        (8.52, 0.1062), (8.60, 0.1393), (8.68, 0.1794), (8.77, 0.2250), (8.85, 0.2788),
        (8.93, 0.3331), (9.02, 0.3902), (9.10, 0.4545), (9.18, 0.5147), (9.27, 0.5704),
        (9.35, 0.6281), (9.43, 0.6796), (9.52, 0.7249), (9.60, 0.7679), (9.68, 0.8035),
        (9.77, 0.8381), (9.85, 0.8674), (9.93, 0.8907), (10.02, 0.9127), (10.18, 0.9371),
        (10.35, 0.9632), (10.68, 0.9853), (11.02, 0.9949), (11.35, 0.9992), (11.60, 1.0000)
    ], columns=['t', 'cobs'])
    obsdata['z'] = 23.0
    fig7_3a3 = CXTsim(inverse=1, mode=1, modc=3, parms=parms, modb=2, pulse=pulse,
                    mit=150, ilmt=1, obsdata=obsdata)
    fig7_3a3.run()
    fig7_3a3.plot_btc()

    bname = ['V', 'D', 'R', 'mu1']
    binit = [0.3,0.3,1.0,0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] =[1,1,0,0]
    parms.loc['bmin'] =[0.01,0.001,999,999]
    parms.loc['bmax'] =[100.0,100.0,999,999]

    cini = [{'conc':1.0,'z':0.0}]

    obsdata = pd.DataFrame([
        (27.37, 1.0000), (29.37, 0.9974), (30.87, 0.9703), (32.37, 0.9518), (32.87, 0.9481),
        (33.87, 0.9401), (34.37, 0.9349), (34.87, 0.9176), (35.37, 0.8774), (35.87, 0.8709),
        (36.37, 0.8638), (36.87, 0.8345), (37.37, 0.8048), (37.87, 0.7968), (38.37, 0.7460),
        (38.87, 0.7276), (39.37, 0.6889), (39.87, 0.6714), (40.37, 0.6337), (40.87, 0.6071),
        (41.37, 0.5705), (41.87, 0.5345), (42.37, 0.4999), (42.87, 0.4652), (43.37, 0.4515),
        (43.87, 0.4180), (44.37, 0.3853), (44.87, 0.3631), (45.37, 0.3314), (45.87, 0.3200),
        (46.37, 0.2890), (46.87, 0.2684), (47.37, 0.2479), (47.87, 0.2278), (48.37, 0.2076),
        (48.87, 0.1876), (49.87, 0.1678), (50.87, 0.1384), (51.87, 0.1192), (52.87, 0.1004),
        (53.87, 0.0913), (54.87, 0.0829), (55.87, 0.0647), (56.87, 0.0564), (57.87, 0.0480),
        (58.87, 0.0398), (60.37, 0.0321), (60.87, 0.0327), (62.37, 0.0246), (63.87, 0.0264),
        (64.87, 0.0171), (66.37, 0.0181)
    ], columns=['t', 'cobs'])
    obsdata['z'] = 11.0
    
    fig7_3b1 = CXTsim(inverse=1, mode=1, modc=3, parms=parms, modi=1, cini=cini,
                      mit=150, ilmt=1, obsdata=obsdata)
    fig7_3b1.run()
    fig7_3b1.plot_btc()

    obsdata = pd.DataFrame([
        (37.87, 0.9971), (39.37, 0.9877), (40.87, 0.9867), (42.37, 0.9862), (44.37, 0.9857),
        (45.87, 0.9751), (47.37, 0.9844), (48.87, 0.9741), (50.37, 0.9738), (51.87, 0.9541),
        (53.37, 0.9531), (54.87, 0.9276), (55.87, 0.9181), (56.87, 0.8986), (58.37, 0.8530),
        (58.87, 0.8435), (59.87, 0.8068), (60.87, 0.7694), (61.37, 0.7508), (61.87, 0.7320),
        (62.37, 0.6950), (63.37, 0.6570), (64.37, 0.6193), (65.37, 0.5711), (66.37, 0.5135),
        (67.37, 0.4647), (68.37, 0.4162), (69.37, 0.3678), (70.37, 0.3283), (71.37, 0.2891),
        (72.37, 0.2778), (73.37, 0.2478), (74.37, 0.2086), (75.37, 0.1792), (76.37, 0.1684),
        (77.87, 0.1296), (79.37, 0.1280), (80.87, 0.0905), (82.37, 0.0804), (83.87, 0.0616),
        (85.37, 0.0519), (86.87, 0.0514), (88.37, 0.0419), (90.37, 0.0413), (91.87, 0.0318),
        (93.87, 0.0223), (95.87, 0.0128)
    ], columns=['t', 'cobs'])
    obsdata['z'] = 17.0

    fig7_3b2 = CXTsim(inverse=1, mode=1, modc=3, parms=parms, modi=1, cini=cini,
                      mit=150, ilmt=1, obsdata=obsdata)
    fig7_3b2.run()
    fig7_3b2.plot_btc()

    obsdata = pd.DataFrame([
        (55.87, 0.9901), (57.37, 0.9715), (58.87, 0.9715), (60.37, 0.9715), (61.87, 0.9714),
        (63.37, 0.9710), (64.87, 0.9708), (66.37, 0.9712), (67.87, 0.9620), (69.37, 0.9627),
        (70.87, 0.9634), (72.37, 0.9652), (73.87, 0.9585), (74.87, 0.9517), (75.87, 0.9359),
        (76.87, 0.9392), (77.87, 0.9245), (78.87, 0.9091), (79.87, 0.8847), (80.87, 0.8708),
        (81.87, 0.8552), (82.87, 0.8293), (83.87, 0.7927), (84.87, 0.7742), (85.87, 0.7362),
        (86.87, 0.6967), (87.87, 0.6569), (88.87, 0.6157), (89.87, 0.5735), (90.87, 0.5407),
        (91.37, 0.5288), (92.87, 0.4647), (93.87, 0.4217), (94.87, 0.3884), (95.87, 0.3654),
        (96.87, 0.3326), (97.87, 0.2911), (99.37, 0.2479), (100.87, 0.2242), (102.37, 0.2015),
        (103.87, 0.1626), (105.37, 0.1594), (108.37, 0.1214), (109.87, 0.0869), (112.87, 0.0695),
        (114.37, 0.0529), (117.37, 0.0537), (118.87, 0.0458), (120.37, 0.0380), (124.37, 0.0310),
        (128.37, 0.0241), (130.37, 0.0245), (131.87, 0.0162)
    ], columns=['t', 'cobs'])
    obsdata['z'] = 23.0

    fig7_3b3 = CXTsim(inverse=1, mode=1, modc=3, parms=parms, modi=1, cini=cini,
                      mit=150, ilmt=1, obsdata=obsdata)
    fig7_3b3.run()
    fig7_3b3.plot_btc()

def test_fig7_4(input_path):
    sims = CXTfit.load(f'{input_path}/FIG7-4.IN',verbose = False)
    sims.run(verbose=False)
    fig,ax = plt.subplots()
    for simcase in sims.simcases:
        simcase.plot_profile(ax=ax)

    obsdata = pd.DataFrame([
    (0.0000, 0.016666), (0.0000, 0.083333), (0.0002, 0.166666), (0.0032, 0.25),
    (0.0059, 0.333333), (0.0100, 0.416666), (0.0129, 0.5), (0.0179, 0.583333),
    (0.0233, 0.666666), (0.0286, 0.75), (0.0336, 0.833333), (0.0399, 0.916666),
    (0.0457, 1.0), (0.0490, 1.083333), (0.0552, 1.166666), (0.0601, 1.25),
    (0.0664, 1.333333), (0.0713, 1.416666), (0.0764, 1.5), (0.0837, 1.583333),
    (0.0890, 1.666666), (0.0929, 1.75), (0.0960, 1.833333), (0.0996, 1.916666),
    (0.1028, 2.0), (0.1081, 2.166666), (0.1136, 2.333333), (0.1136, 2.5),
    (0.1169, 2.666666), (0.1166, 2.833333), (0.1167, 3.0), (0.1147, 3.166666),
    (0.1125, 3.333333), (0.1105, 3.5), (0.1065, 3.666666), (0.1043, 3.833333),
    (0.1007, 4.0), (0.0985, 4.166666), (0.0932, 4.333333), (0.0918, 4.5),
    (0.0874, 4.666666), (0.0822, 4.833333), (0.0791, 5.0), (0.0771, 5.166666),
    (0.0738, 5.333333), (0.0712, 5.5), (0.0674, 5.666666), (0.0652, 5.833333),
    (0.0625, 6.0), (0.0588, 6.333333), (0.0543, 6.666666), (0.0489, 7.0),
    (0.0449, 7.333333), (0.0435, 7.666666), (0.0401, 8.0), (0.0376, 8.333333),
    (0.0350, 8.666666), (0.0322, 9.0), (0.0295, 9.5), (0.0248, 10.0),
    (0.0249, 10.5), (0.0224, 11.0), (0.0199, 11.5), (0.0184, 12.0),
    (0.0172, 12.5), (0.0170, 13.0), (0.0150, 13.5), (0.0144, 14.0),
    (0.0129, 14.5), (0.0118, 15.0), (0.0051, 21.0), (0.0032, 22.41666),
    (0.0032, 22.83333), (0.0035, 23.16666), (0.0030, 23.66666), (0.0020, 24.0),
    (0.0031, 24.5), (0.0018, 25.0), (0.0028, 26.0), (0.0011, 27.0),
    (0.0017, 28.16666), (0.0015, 29.16666), (0.0010, 31.16666), (0.0006, 32.5),
    (0.0001, 35.0), (0.0000, 35.88333)
    ], columns=['cobs', 't'])
    obsdata['z'] = 10.0

    bname = ['V', 'D', 'R', 'mu1', 'Cin',   'T2']
    binit = [3.00,1.00,1.0,0.0,1.0,0.5]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] =[1,1,0,0,0,1]
    parms.loc['bmin'] =[0.01,0.001,999,999,999,0.1]
    parms.loc['bmax'] =[100.0,100.0,999,999,999,2.0]

    pulse = [{'conc':1.0,'time':0.0},{'conc':0.0, 'time':0.5}]

    fig7_4 = CXTsim(inverse=1, mode=1, modc=3, parms=parms, modb=3, pulse=pulse,
                      mit=150, ilmt=1, mass=1, obsdata=obsdata)
    fig7_4.run()
    fig,ax = plt.subplots()
    fig7_4.plot_btc(ax=ax)

def test_fig7_5(input_path):
    sims = CXTfit.load(f'{input_path}/FIG7-5.IN',verbose = False)
    sims.run(verbose=False)
    fig,ax = plt.subplots()
    for simcase in sims.simcases:
        simcase.plot_btc(ax=ax)

    bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
    binit = [20., 10., 5.0,  0.2,    0.8,     0.0,   0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] = [0] * len(bname)

    pulse = [{'conc':1.0,'time':0.0}]

    nt = 101
    dt = 0.2
    obsdata = pd.DataFrame([{'t': t*dt, 'z': 50.0} for t in range(nt)])

    Case1 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case1.run()
    fig,ax = plt.subplots()
    Case1.plot_btc(ax=ax)

    parms.loc['binit','omega'] = 2.0
    Case2 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case2.run()
    fig,ax = plt.subplots()
    Case2.plot_btc(ax=ax)

    parms.loc['binit','omega'] = 10.0
    Case3 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case3.run()
    fig,ax = plt.subplots()
    Case3.plot_btc(ax=ax)

    bname = ['V', 'D', 'R', 'mu1']
    binit = [20.0,10.0,5.0,0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] =[0] * len(bname)
    Case4 = CXTsim(mode=1, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case4.run()
    fig,ax = plt.subplots()
    Case4.plot_btc(ax=ax)

def test_fig7_6(input_path):
    sims = CXTfit.load(f'{input_path}/FIG7-6A.IN',verbose = False)
    sims.run(verbose=False)
    fig,ax = plt.subplots()
    for simcase in sims.simcases:
        simcase.plot_btc(ax=ax)

    sims = CXTfit.load(f'{input_path}/FIG7-6B.IN',verbose = False)
    sims.run(verbose=False)
    fig,ax = plt.subplots()
    for simcase in sims.simcases:
        simcase.plot_btc(ax=ax)
    
    pulse = [{'conc':1.0,'time':0.0}]

    nt = 101
    dt = 0.2
    obsdata = pd.DataFrame([{'t': t*dt, 'z': 50.0} for t in range(nt)])

    bname = ['V', 'D', 'R', 'mu1']
    binit = [20.0,10.0,5.0,0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] =[0] * len(bname)
    Case0 = CXTsim(mode=1, modc=1, parms=parms, modb=1, pulse=pulse,
                   obsdata=obsdata)
    Case0.run()
    fig,ax = plt.subplots()
    Case0.plot_btc(ax=ax)

    bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
    binit = [20.,10.,5.0,0.2,0.8,0.0,0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] = [0] * len(bname)

    Case1 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case1.run()
    fig,ax = plt.subplots()
    Case1.plot_btc(ax=ax)

    parms.loc['binit','beta']  = 0.44
    parms.loc['binit','omega'] = 0.56
    Case2 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case2.run()
    fig,ax = plt.subplots()
    Case2.plot_btc(ax=ax)

    parms.loc['binit','beta']  = 0.76
    parms.loc['binit','omega'] = 0.24
    Case3 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case3.run()
    fig,ax = plt.subplots()
    Case3.plot_btc(ax=ax)

    parms.loc['binit','beta']  = 0.2
    parms.loc['binit','omega'] = 2.0
    Case5 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case5.run()
    fig,ax = plt.subplots()
    Case5.plot_btc(ax=ax)

    parms.loc['binit','beta']  = 0.44
    parms.loc['binit','omega'] = 1.4
    Case6 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case6.run()
    fig,ax = plt.subplots()
    Case6.plot_btc(ax=ax)

    parms.loc['binit','beta']  = 0.76
    parms.loc['binit','omega'] = 0.6
    Case7 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case7.run()
    fig,ax = plt.subplots()
    Case7.plot_btc(ax=ax)

def test_fig7_7(input_path):
    sims = CXTfit.load(f'{input_path}/FIG7-7A.IN',verbose = False)
    sims.run(verbose=False)
    fig,ax = plt.subplots()
    for simcase in sims.simcases:
        simcase.plot_btc(ax=ax)

    sims = CXTfit.load(f'{input_path}/FIG7-7B.IN',verbose = False)
    sims.run(verbose=False)
    fig,ax = plt.subplots()
    for simcase in sims.simcases:
        simcase.plot_btc(ax=ax)

    pulse = [{'conc':1.0,'time':0.0}]

    nt = 101
    dt = 0.2
    obsdata = pd.DataFrame([{'t': t*dt, 'z': 50.0} for t in range(nt)])

    bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
    binit = [20.,10.,5.0,0.44,0.56,0.0,0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] = [0] * len(bname)

    Case1 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case1.run()
    fig,ax = plt.subplots()
    Case1.plot_btc(ax=ax)

    pulse = [{'conc':1.0,'time':0.0},{'conc':0.0,'time':5.0}]
    Case3 = CXTsim(mode=2, modc=1, parms=parms, modb=3, pulse=pulse,
                   zl=50.0, obsdata=obsdata)
    Case3.run()
    fig,ax = plt.subplots()
    Case3.plot_btc(ax=ax)

    parms.loc['binit','R']  = 10.0
    parms.loc['binit','beta']  = 0.22
    parms.loc['binit','omega'] = 0.55
    Case2 = CXTsim(mode=2, modc=1, parms=parms, modb=1, pulse=pulse,
                    zl=50.0, obsdata=obsdata)
    Case2.run()
    fig,ax = plt.subplots()
    Case2.plot_btc(ax=ax)

    pulse = [{'conc':1.0,'time':0.0},{'conc':0.0,'time':5.0}]
    Case4 = CXTsim(mode=2, modc=1, parms=parms, modb=3, pulse=pulse,
                   zl=50.0, obsdata=obsdata)
    Case4.run()
    fig,ax = plt.subplots()
    Case4.plot_btc(ax=ax)

def test_fig7_8(input_path):
    sims = CXTfit.load(f'{input_path}/FIG7-8.IN',verbose = False)
    sims.run(verbose=False)
    fig,ax = plt.subplots()
    for simcase in sims.simcases:
        simcase.plot_btc(ax=ax)
