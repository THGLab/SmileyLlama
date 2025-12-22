'''
Author: Jie Li
Date created: Mar 30, 2021

Features are : 
    * fraction of sp3 carbon
    * number of heavy atoms
    * number of hbond donors
    * number of hbond acceptors
    * number of aliphatic rings
    * number of aromatic rings
    * number of rotatable bonds
    * QED (quantitative estimate of drug-likelihood) value (not used any moree)
    * molecular weight
    * alogP (approximate log partition coefficient between octanol and water)
    * PSA (polarizable surface area)
    * alerts (number of structural alerts)
    * fraction of hetero atoms in all heavy atoms
    * maximum ring size

Distributions generated from 10000 random samples of CHEMBL molecules
'''

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Lipinski, Descriptors, Crippen
from collections import namedtuple

OUT_OF_RANGE = -10

def log_prob(vector, bin_range):
    vector=np.array(vector)
    counts=[np.sum(vector<threshold) for threshold in bin_range]+[len(vector)]
    diffs = np.array(counts[1:])-np.array(counts[:-1])
    prob = diffs/len(vector)
    return np.log(prob)

def make_onehot(num, bin_range):
    bool_indicator = np.concatenate([num < bin_range, [True]])
    onehot_indicator = bool_indicator[1:].astype(int) - bool_indicator[:-1].astype(int)
    if num < bin_range[0]:
        onehot_indicator[0]=1
    return onehot_indicator
    
def get_max_ring_size(mol):
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    if len(ring_sizes) == 0:
        return 0
    return max(ring_sizes)

def calc_props(input):
    if type(input) is str:
        mol = Chem.MolFromSmiles(input)
    else:
        mol = input
    fraction_csp3 = Lipinski.FractionCSP3(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    hbond_donor = Lipinski.NumHDonors(mol)
    hbond_acceptor = Lipinski.NumHAcceptors(mol)
    num_ring_aliphatic = Lipinski.NumAliphaticRings(mol)
    num_ring_aromatic = Lipinski.NumAromaticRings(mol)
    num_rotatable_bond = Lipinski.NumRotatableBonds(mol)
    qed_prop = QED.properties(mol)
    qed_default = QED.default(mol)
    hetero_prop = Lipinski.NumHeteroatoms(mol)/heavy_atoms
    max_ring_size = get_max_ring_size(mol)
    return [fraction_csp3, heavy_atoms, hbond_donor, hbond_acceptor, num_ring_aliphatic, num_ring_aromatic, num_rotatable_bond, \
            qed_default, qed_prop.MW, qed_prop.ALOGP, qed_prop.PSA, qed_prop.ALERTS, hetero_prop, max_ring_size]

class DrugLikeliness():
    def __init__(self, relative_weights="inverse_entropy"):
        n_total_properties = 13
        self.frac_csp3_LP = np.array([-3.1844743981508685, -5.991464547107982, -3.5684333010380693, -3.6045383056801854, -3.7422802308410517, -3.4265151896464454, -3.387294476493164, -3.372609924810243, -3.3214624136433017, -3.579128590154817, -3.158251203051766, -3.1629681929299047, -3.3755296349135775, -3.165335057940171, -3.3524072174927233, -3.0159349808715104, -3.177254150177381, -3.3991993789994823, -3.280751228586288, -3.6045383056801854, -3.223888366691745, -3.547379891840237, -3.513246885470778, -3.684887432844399, -4.9062752787720125, -3.4641371814360293, -3.6769508832486624, -4.29768548624013, -4.166915255056936, -4.688551794927142, -4.226733750267846, -4.537511537514277, -4.7914997641795845, -4.879607031689852, -4.7914997641795845, -5.360192770266124, -5.259096653394756, -5.2785147392518565, -5.403677882205863, -5.952243833954701, -5.339139361068292, -5.683979847360021, -5.744604469176456, -5.952243833954701, -6.119297918617867, -5.914503505971854, -5.914503505971854, -6.437751649736401, -7.1308988302963465, -8.517193191416238, -5.1672891041416324]) #np.linspace(0,1,51) 
        self.heavy_atom_LP = np.array([-6.437751649736401, -6.119297918617867, -5.44914025628262, -5.149897361429764, -4.919880930827792, -4.474141923581687, -4.283086686818978, -3.958066943929553, -3.713172146682981, -3.649658740960655, -3.2649197633696074, -3.1419147837320724, -3.103317484233928, -2.9700645268054133, -2.9187712324178627, -2.761450977829325, -2.9058915695542016, -3.028255465259551, -2.935578350734244, -2.9603651297166995, -3.001750345879554, -3.2114038100294997, -3.151217176394386, -3.291446517703036, -3.372609924810243, -3.543913683863751, -3.816712825623821, -3.872802292274865, -4.110473944151984, -4.268697949366879, -4.6356293934728, -4.721704002244043, -4.8283137373023015, -5.149897361429764, -5.06720564558465, -5.318520073865556, -5.360192770266124, -5.654992310486769, -5.991464547107982, -6.032286541628237, -6.214608098422191, -5.914503505971854, -5.914503505971854, -6.502290170873972, -6.645391014514646, -6.502290170873972, -7.264430222920869, -6.907755278982137, -7.600902459542082, -8.517193191416238, -7.418580902748128]) #np.arange(10,61)
        self.hbond_donor_LP = np.array([-1.5436501718960973, -0.9834994815676051, -1.4250351894363211, -2.2936253516225737, -3.2213789550863194, -4.305065593537753, -4.0398563769380305]) # np.arange(7)
        self.hbond_acceptor_LP = np.array([-6.3771270279199666, -4.213128098212068, -2.7181005369557116, -2.011409131288009, -1.6174700841313647, -1.6533897999632832, -1.8344582237611702, -2.2433732333621994, -2.791975436039971, -3.4451492691913383, -4.086376392572924, -4.011843340710357]) # np.arange(12)
        self.n_ring_aliphatic_LP = np.array([-0.9068355731033999, -1.0051219455807707, -1.8865098057738656, -3.0812901619156374, -4.028556821684098, -4.2063940660307235]) # np.arange(6)
        self.n_ring_aromatic_LP = np.array([-3.0470255679415414, -1.9653988256391757, -1.1391218320063197, -1.1854782216897717, -1.9337839692574728, -3.167707538293801]) # np.arange(6)
        self.n_rot_bond_LP = np.array([-4.268697949366879, -3.1122660898099426, -2.475748712003235, -2.102914897865478, -1.8438952236485835, -1.9146052992269007, -2.0651442369790116, -2.371935171128839, -2.667868411469378, -3.158251203051766, -3.5613661338149765, -3.912023005428146, -4.406319327242926, -4.7914997641795845, -5.083205986931091, -5.683979847360021, -5.572754212249797, -5.626821433520073, -5.952243833954701, -6.319968614080018, -6.571283042360924, -5.115995809754082]) # np.arange(22)
        # self.qed_LP = np.array([-8.517193191416238, -6.319968614080018, -5.683979847360021, -5.521460917862246, -5.403677882205863, -4.9062752787720125, -4.866534950122499, -4.733003557497976, -4.677740878822926, -4.474141923581687, -4.350527967614511, -4.261580481598014, -4.045554398052668, -4.0398563769380305, -3.907035463917107, -3.83970234384852, -3.789805372703897, -3.789805372703897, -3.653512310276645, -3.611918412977808, -3.6082215510964817, -3.653512310276645, -3.575550768806933, -3.575550768806933, -3.499913354601313, -3.451438598098902, -3.586322865788844, -3.5899395062590327, -3.387294476493164, -3.547379891840237, -3.4673371841667002, -3.540459448995663, -3.4265151896464454, -3.4295968561838532, -3.3813947543659757, -3.6306105459899607, -3.4326880487535263, -3.4673371841667002, -3.470547459796949, -3.4295968561838532, -3.653512310276645, -3.6306105459899607, -3.6343912688298667, -3.816712825623821, -4.240527072400182, -4.186459851129906, -5.0206856299497575, -6.645391014514646]) #np.linspace(0,0.94,48)
        self.mw_LP = np.array([-6.907755278982137, -6.645391014514646, -6.437751649736401, -5.914503505971854, -5.546778725846536, -5.035953102080546, -4.815891217303744, -4.283086686818978, -3.9322257127456655, -3.789805372703897, -3.417326763592039, -3.216378944669614, -3.0597376035299035, -2.875286120478124, -2.722656353491572, -2.6971102610638757, -2.877060743836492, -2.820099704910833, -2.752002088631393, -2.926206210905381, -2.8913722582297483, -2.993734270891318, -3.1190304898984844, -3.2467610283577364, -3.402197881995739, -3.6573807870545654, -3.7009520353482057, -4.000854219134761, -4.254513314374922, -4.382026634673881, -4.55638002181866, -4.767689115485866, -5.099466507802871, -5.149897361429764, -5.403677882205863, -5.599422459331958, -5.7763531674910364, -5.809142990314028, -5.991464547107982, -6.3771270279199666, -5.8781358618009785, -6.3771270279199666, -6.437751649736401, -7.418580902748128, -7.013115794639964, -7.824046010856292, -7.824046010856292, -10, -8.517193191416238, -10, -7.824046010856292]) #np.linspace(100,900,51)
        self.alogp_LP = np.array([-9.210340371976182, -8.517193191416238, -7.418580902748128, -8.517193191416238, -7.264430222920869, -7.600902459542082, -7.264430222920869, -7.418580902748128, -6.437751649736401, -6.645391014514646, -5.991464547107982, -5.683979847360021, -5.626821433520073, -5.099466507802871, -4.892852258439873, -4.456750180869818, -4.006333684899388, -3.8490482062667577, -3.387294476493164, -3.228926160721702, -2.9565115604007097, -2.779009290042704, -2.6283152330833564, -2.5083800059736423, -2.5257286443082556, -2.482908647125327, -2.517016703706234, -2.682382454353632, -2.7822350992915865, -2.9798589243977007, -3.2213789550863194, -3.5267606046375013, -3.7550192566184815, -4.213128098212068, -4.290359446148058, -4.755993075722675, -4.947660494934867, -5.240048458424061, -5.952243833954701, -6.032286541628237, -6.571283042360924, -7.013115794639964, -7.824046010856292, -7.418580902748128, -7.824046010856292, -8.517193191416238, -8.517193191416238, -8.517193191416238, -9.210340371976182, -9.210340371976182, -7.824046010856292]) #np.linspace(-5,12,51)
        self.psa_LP = np.array([-5.572754212249797, -5.809142990314028, -4.474141923581687, -4.275866438845491, -3.70500883604382, -3.6082215510964817, -3.074775480894444, -3.0449225177447627, -2.875286120478124, -2.6091102532473065, -2.6736487743848776, -2.5232317641096684, -2.5902671654458267, -2.6255489795904663, -2.652142569163914, -2.9040650850281673, -3.036554268074246, -3.1349943408874985, -3.341043458842408, -3.600868576791223, -3.763603000309873, -4.154094566627875, -4.319991243754429, -4.56594947283481, -4.733003557497976, -4.840892519509161, -5.1672891041416324, -5.426150738057921, -5.626821433520073, -5.713832810509703, -6.074846156047033, -6.3771270279199666, -6.502290170873972, -6.319968614080018, -6.119297918617867, -6.571283042360924, -7.1308988302963465, -7.013115794639964, -7.1308988302963465, -7.418580902748128, -7.418580902748128, -7.013115794639964, -8.111728083308073, -7.264430222920869, -8.517193191416238, -8.111728083308073, -8.111728083308073, -8.111728083308073, -8.111728083308073, -7.824046010856292, -6.3771270279199666]) #np.linspace(0,300,51)
        self.alerts_LP = np.array([-0.764428382564909, -1.2986498512678444, -1.8263509139976741, -2.647896278282463, -3.7465085669505727, -5.099466507802871]) #np.arange(6)
        self.hetero_prop_LP = np.array([-7.600902459542082, -9, -9.210340371976182, -6.907755278982137, -6.645391014514646, -5.8781358618009785, -5.472670753692815, -5.149897361429764, -5.2030071867437115, -4.853631545286591, -4.154094566627875, -4.06284589516273, -3.9738984091462335, -3.543913683863751, -3.4357888264317746, -3.2570970376883985, -3.0159349808715104, -3.0159349808715104, -2.8788385220824915, -2.8577109756566164, -2.474560357733856, -3.028255465259551, -2.7045563118479543, -2.866459937849852, -3.085656981081978, -2.91139112512024, -2.928073625080176, -3.0512449834842497, -3.8076629901039034, -3.937340813412436, -3.825845309187094, -4.312500572025272, -4.193060535161258, -4.509860006183766, -4.6356293934728, -4.9062752787720125, -5.240048458424061, -5.035953102080546, -5.7763531674910364, -5.744604469176456, -7.013115794639964, -5.496768305271875, -8.111728083308073, -6.437751649736401, -6.725433722188183, -6.907755278982137, -8.111728083308073, -7.418580902748128, -8.517193191416238, -9.210340371976182, -6.074846156047033])  #np.linspace(0,0.6,51)
        self.max_ring_size_LP = np.array([ -4.64599114, -18.42068074, -18.42068074,  -7.41856424,        -8.51714319,  -3.84904774,  -0.08675685,  -3.27809492,-6.16581317,  -7.26441594,  -7.13088633,  -7.82402101,        -6.5712759 ,  -7.26441594,  -4.87960572]) # np.arange(15)
        
        # AutodockVina invalid atom types:
        self.invalid_atom_types = ["B", "Si", "Te"]
        
        if relative_weights == "inverse_entropy":
            weight_vectors = [-1 / x.dot(np.exp(x)) for x in [self.frac_csp3_LP, self.heavy_atom_LP, self.hbond_donor_LP, self.hbond_acceptor_LP, self.n_ring_aliphatic_LP, self.n_ring_aromatic_LP, self.n_rot_bond_LP, self.mw_LP, self.alogp_LP, self.psa_LP, self.alerts_LP, self.hetero_prop_LP, self.max_ring_size_LP]]
            self.relative_weights = np.array(weight_vectors) / np.sum(weight_vectors)
        elif isinstance(relative_weights, np.ndarray):
            self.relative_weights = relative_weights
        else:
            self.relative_weights = np.array([1/n_total_properties] * n_total_properties)
        
    
    def check_valid_atomtypes(self, input):
        if type(input) is not str:
            mol = Chem.SmilesFromMol(input)
        else:
            mol = input
        for element in self.invalid_atom_types:
            if element == "B":
                items = mol.split("B")
                if len(items) == 1:
                    continue
                for i in items[1:]:
                    if not i.startswith("r"):
                        return False
            else:
                if element in mol:
                    return False
        return True
        
    def calc_score(self, input, offset=5):
        try:
            props = calc_props(input)
        except:
            return 0
        #if not self.check_valid_atomtypes(input):
        #    return 0
        log_prob = np.array([
            make_onehot(props[0], np.linspace(0,1,51)).dot(self.frac_csp3_LP),
            make_onehot(props[1], np.arange(10,61)).dot(self.heavy_atom_LP),
            make_onehot(props[2], np.arange(7)).dot(self.hbond_donor_LP),
            make_onehot(props[3], np.arange(12)).dot(self.hbond_acceptor_LP),
            make_onehot(props[4], np.arange(6)).dot(self.n_ring_aliphatic_LP),
            make_onehot(props[5], np.arange(6)).dot(self.n_ring_aromatic_LP),
            make_onehot(props[6], np.arange(22)).dot(self.n_rot_bond_LP),
            # make_onehot(props[7], np.linspace(0,0.94,48)).dot(self.qed_LP),
            make_onehot(props[8], np.linspace(100,900,51)).dot(self.mw_LP),
            make_onehot(props[9], np.linspace(-5,12,51)).dot(self.alogp_LP),
            make_onehot(props[10], np.linspace(0,300,51)).dot(self.psa_LP),
            make_onehot(props[11], np.arange(6)).dot(self.alerts_LP),
            make_onehot(props[12], np.linspace(0,0.6,51)).dot(self.hetero_prop_LP),
            make_onehot(props[13], np.arange(15)).dot(self.max_ring_size_LP)
        ])
        return log_prob.dot(self.relative_weights) + offset
    

class LeadLikeliness():
    def __init__(self, relative_weights="inverse_entropy"):
        n_total_properties = 13
        self.frac_csp3_LP = np.array([-3.63389309, -3.23493419, -2.92147749, -2.8932536 , -2.80964939,
       -2.76134691, -2.7436188 , -2.76951022, -2.72463089, -2.81478199,
       -2.82686151, -2.73883792, -2.81135733, -2.89140004, -2.99060669,
       -3.10762314, -3.59198219, -3.83111104, -4.12037414, -4.32315677,
       -4.58784933, -4.82626035, -4.99331443, -5.23172546, -5.68646161,
       -6.13844674]) #np.linspace(0,1,26) 
        self.heavy_atom_LP = np.array([OUT_OF_RANGE, -3.85823066, -3.06930481, -2.56931046, -2.25707437, -2.0528191 ,
       -1.932403  , -1.90133541, -1.96913245, -2.213346  , -2.52894312, OUT_OF_RANGE]) #np.arange(15, 26)
        self.hbond_donor_LP = np.array([-1.44330595, -0.90783928, -1.40799594, -2.46073778, -3.86473172,
       -4.96334401, -5.99296342]) # np.arange(7)
        self.hbond_acceptor_LP = np.array([OUT_OF_RANGE, -5.65649119, -2.87501352, -1.75307656, -1.33330505,
       -1.37191989, -1.88208956, -2.75037107, -3.77375994, -4.76918799,
       -6.21610697, OUT_OF_RANGE]) # np.arange(12)
        self.n_ring_aliphatic_LP = np.array([-0.93799232, -0.84896342, -1.98200047, -3.42801407, -4.82981261,
       OUT_OF_RANGE]) # np.arange(6)
        self.n_ring_aromatic_LP = np.array([-3.15975008, -1.2798361 , -0.81455635, -1.5597685 , -3.69037833,
       OUT_OF_RANGE]) # np.arange(6)
        self.n_rot_bond_LP = np.array([-4.23510551, -2.75821425, -1.90195776, -1.4976081 , -1.51790192,
       -1.77640123, -2.22712293, -2.93896224, OUT_OF_RANGE]) # np.arange(9)
        # self.qed_LP = np.array([-8.517193191416238, -6.319968614080018, -5.683979847360021, -5.521460917862246, -5.403677882205863, -4.9062752787720125, -4.866534950122499, -4.733003557497976, -4.677740878822926, -4.474141923581687, -4.350527967614511, -4.261580481598014, -4.045554398052668, -4.0398563769380305, -3.907035463917107, -3.83970234384852, -3.789805372703897, -3.789805372703897, -3.653512310276645, -3.611918412977808, -3.6082215510964817, -3.653512310276645, -3.575550768806933, -3.575550768806933, -3.499913354601313, -3.451438598098902, -3.586322865788844, -3.5899395062590327, -3.387294476493164, -3.547379891840237, -3.4673371841667002, -3.540459448995663, -3.4265151896464454, -3.4295968561838532, -3.3813947543659757, -3.6306105459899607, -3.4326880487535263, -3.4673371841667002, -3.470547459796949, -3.4295968561838532, -3.653512310276645, -3.6306105459899607, -3.6343912688298667, -3.816712825623821, -4.240527072400182, -4.186459851129906, -5.0206856299497575, -6.645391014514646]) #np.linspace(0,0.94,48)
        self.mw_LP = np.array([OUT_OF_RANGE, -3.72427988, -3.31651856, -3.24142811, -3.25427625,
       -3.12506452, -3.11376497, -2.98070185, -2.99723115, -2.98728082,
       -2.99058661, -2.99390336, -2.94844099, -2.98070185, -2.93269263,
       -2.91411624, -2.86620289, -2.82889583, -2.80661079, -2.78211977,
       -3.19975541, OUT_OF_RANGE]) #np.arange(245,355,5)
        self.alogp_LP = np.array([OUT_OF_RANGE, -5.81064187, -4.65796236, -3.75225373, -2.35537726,
       -1.42031643, -0.85598921, -1.59851427, OUT_OF_RANGE]) #np.arange(-4,5)
        self.psa_LP = np.array([-5.20300719, -2.70306266, -1.25702203, -1.08618977, -1.56542103,
       -2.7488722 , -3.81671283, -4.82831374, -5.99146455, OUT_OF_RANGE]) #np.arange(0,200,20)
        self.alerts_LP = np.array([-0.52321694, -1.55266788, -2.10933989, -2.9295725 , -3.86473172, OUT_OF_RANGE]) #np.arange(6)
        self.hetero_prop_LP = np.array([OUT_OF_RANGE, -5.08742949, -4.04438705, -3.6294432 , -2.88720641, -2.56062035,
       -2.06240084, -2.03905348, -1.91163045, -2.15443555, -2.28488844,
       -2.72199649, -3.03538692, -3.49434071, -3.88616305, -4.49365478,
       -4.82714639, -5.56285318, -5.60730495, -6.50112282, -6.90658793, OUT_OF_RANGE])  #np.arange(0.075,0.625,0.025)
        self.max_ring_size_LP = np.array([-5.65349118, OUT_OF_RANGE, OUT_OF_RANGE, OUT_OF_RANGE, -7.59940133,
       -3.01443385, -0.10775586, -3.06680184, -5.98996342, OUT_OF_RANGE]) # np.arange(10)
        if relative_weights == "inverse_entropy":
            weight_vectors = [-1 / x.dot(np.exp(x)) for x in [self.frac_csp3_LP, self.heavy_atom_LP, self.hbond_donor_LP, self.hbond_acceptor_LP, self.n_ring_aliphatic_LP, self.n_ring_aromatic_LP, self.n_rot_bond_LP, self.mw_LP, self.alogp_LP, self.psa_LP, self.alerts_LP, self.hetero_prop_LP, self.max_ring_size_LP]]
            self.relative_weights = np.array(weight_vectors) / np.sum(weight_vectors)
        elif isinstance(relative_weights, np.ndarray):
            self.relative_weights = relative_weights
        else:
            self.relative_weights = np.array([1/n_total_properties] * n_total_properties)

    def calc_score(self, input, offset=5):
        try:
            props = calc_props(input)
        except:
            return 0
        log_prob = np.array([
            make_onehot(props[0], np.linspace(0,1,26)).dot(self.frac_csp3_LP),
            make_onehot(props[1], np.arange(15, 27)).dot(self.heavy_atom_LP),
            make_onehot(props[2], np.arange(7)).dot(self.hbond_donor_LP),
            make_onehot(props[3], np.arange(12)).dot(self.hbond_acceptor_LP),
            make_onehot(props[4], np.arange(6)).dot(self.n_ring_aliphatic_LP),
            make_onehot(props[5], np.arange(6)).dot(self.n_ring_aromatic_LP),
            make_onehot(props[6], np.arange(9)).dot(self.n_rot_bond_LP),
            # make_onehot(props[7], np.linspace(0,0.94,48)).dot(self.qed_LP),
            make_onehot(props[8], np.arange(245,355,5)).dot(self.mw_LP),
            make_onehot(props[9], np.arange(-4,5)).dot(self.alogp_LP),
            make_onehot(props[10], np.arange(0,200,20)).dot(self.psa_LP),
            make_onehot(props[11], np.arange(6)).dot(self.alerts_LP),
            make_onehot(props[12], np.arange(0.075,0.625,0.025)).dot(self.hetero_prop_LP),
            make_onehot(props[13], np.arange(10)).dot(self.max_ring_size_LP)
        ])
        return log_prob.dot(self.relative_weights) + offset


# ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure 
# John S. Delaney, J. Chem. Inf. Comput. Sci., 2004, 44, 1000 - 1005
# https://pubs.acs.org/doi/abs/10.1021/ci034243x 
# Adapted from https://github.com/PatWalters/solubility.git

class ESOLCalculator():
    def __init__(self):
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.Descriptor = namedtuple("Descriptor", "mw logp rotors ap".split())

    def calc_ap(self, mol):
        """
        Calculate aromatic proportion #aromatic atoms/#atoms total
        :param mol: input molecule
        :return: aromatic proportion
        """
        matches = mol.GetSubstructMatches(self.aromatic_query)
        return len(matches) / mol.GetNumAtoms()

    def calc_esol_descriptors(self, mol):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input molecule
        :return: named tuple with descriptor values
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        ap = self.calc_ap(mol)
        return self.Descriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)

    def calc_score(self, input):
        """
        Calculate ESOL based on descriptors in the Delaney paper, coefficients refit for the RDKit using the
        routine refit_esol below
        :param input: input molecule (smiles string)
        :return: predicted solubility
        """
        # original coef from delaney
        #intercept = 0.16
        #coef = {"logp": -0.63, "mw": -0.0062, "rotors": 0.066, "ap": -0.74}
        intercept = 0.26121066137801696
        coef = {'mw': -0.0066138847738667125, 'logp': -0.7416739523408995, 'rotors': 0.003451545565957996, 'ap': -0.42624840441316975}
        if type(input) is str:
            mol = Chem.MolFromSmiles(input)
        else:
            mol = input
        desc = self.calc_esol_descriptors(mol)
        esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return esol
    
    
if __name__ == '__main__':
    drug = DrugLikeliness()
    smiles = ["COCCO", "O=C(NC=CC=C(F)C=CF)C=CC=CCN[C@@H1][Si]/OI", "Br[C@@]=C", "O=CN=C(B)B=[C@][N+1][NH1]CBr"]
    for s in smiles:
        print(drug.calc_score(s))
