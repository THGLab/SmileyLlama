__all__ = ['iMinerDrugLikeliness']

from typing import Dict, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Lipinski

from .base import Score, accept_smiles
from .registry import register


def _log_prob(vector, bin_range):
    """Compute log probabilities for values in bins.
    
    Calculates the log probability of values falling into bins defined
    by bin_range thresholds.
    
    Parameters
    ----------
    vector : array-like
        Input values to bin.
    bin_range : array-like
        Threshold values defining bin boundaries.
    
    Returns
    -------
    numpy.ndarray
        Log probabilities for each bin.
    """
    vector=np.array(vector)
    counts=[np.sum(vector<threshold) for threshold in bin_range]+[len(vector)]
    diffs = np.array(counts[1:])-np.array(counts[:-1])
    prob = diffs/len(vector)
    return np.log(prob)


def _make_onehot(num, bin_range):
    """Create one-hot encoding for a value based on bin ranges.
    
    Creates a one-hot vector indicating which bin the value falls into.
    If the value is below the minimum bin range, the first bin is set to 1.
    
    Parameters
    ----------
    num : float or int
        Value to encode.
    bin_range : array-like
        Threshold values defining bin boundaries.
    
    Returns
    -------
    numpy.ndarray
        One-hot encoded vector with 1 in the appropriate bin, 0 elsewhere.
    """
    bool_indicator = np.concatenate([num < bin_range, [True]])
    onehot_indicator = bool_indicator[1:].astype(int) - bool_indicator[:-1].astype(int)
    if num < bin_range[0]:
        onehot_indicator[0]=1
    return onehot_indicator


def _get_max_ring_size(mol):
    """Get the maximum ring size in a molecule.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    
    Returns
    -------
    int
        Maximum ring size, or 0 if molecule has no rings.
    """
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    if len(ring_sizes) == 0:
        return 0
    return max(ring_sizes)


@register("rdkit_scores")
class iMinerDrugLikeliness(Score):
    '''
    Drug likeliness score described in `10.1021/acs.jcim.4c00634 <https://doi.org/10.1021/acs.jcim.4c00634>`_ equation 1-3
    '''

    _LOG_PROPS = {
        'frac_csp3': [
            -3.1844743981508685, -5.991464547107982, -3.5684333010380693, -3.6045383056801854, -3.7422802308410517, -3.4265151896464454, -3.387294476493164, -3.372609924810243, 
            -3.3214624136433017, -3.579128590154817, -3.158251203051766, -3.1629681929299047, -3.3755296349135775, -3.165335057940171, -3.3524072174927233, -3.0159349808715104, 
            -3.177254150177381, -3.3991993789994823, -3.280751228586288, -3.6045383056801854, -3.223888366691745, -3.547379891840237, -3.513246885470778, -3.684887432844399, -4.9062752787720125, 
            -3.4641371814360293, -3.6769508832486624, -4.29768548624013, -4.166915255056936, -4.688551794927142, -4.226733750267846, -4.537511537514277, -4.7914997641795845, -4.879607031689852, 
            -4.7914997641795845, -5.360192770266124, -5.259096653394756, -5.2785147392518565, -5.403677882205863, -5.952243833954701, -5.339139361068292, -5.683979847360021, -5.744604469176456, 
            -5.952243833954701, -6.119297918617867, -5.914503505971854, -5.914503505971854, -6.437751649736401, -7.1308988302963465, -8.517193191416238, -5.1672891041416324
        ],
        "heavy_atom": [
            -6.437751649736401, -6.119297918617867, -5.44914025628262, -5.149897361429764, -4.919880930827792, -4.474141923581687, -4.283086686818978, -3.958066943929553, -3.713172146682981, 
            -3.649658740960655, -3.2649197633696074, -3.1419147837320724, -3.103317484233928, -2.9700645268054133, -2.9187712324178627, -2.761450977829325, -2.9058915695542016, -3.028255465259551, 
            -2.935578350734244, -2.9603651297166995, -3.001750345879554, -3.2114038100294997, -3.151217176394386, -3.291446517703036, -3.372609924810243, -3.543913683863751, -3.816712825623821, 
            -3.872802292274865, -4.110473944151984, -4.268697949366879, -4.6356293934728, -4.721704002244043, -4.8283137373023015, -5.149897361429764, -5.06720564558465, -5.318520073865556, -5.360192770266124,
             -5.654992310486769, -5.991464547107982, -6.032286541628237, -6.214608098422191, -5.914503505971854, -5.914503505971854, -6.502290170873972, -6.645391014514646, -6.502290170873972, -7.264430222920869, 
             -6.907755278982137, -7.600902459542082, -8.517193191416238, -7.418580902748128
        ],
        "hbond_donor": [
            -1.5436501718960973, -0.9834994815676051, -1.4250351894363211, -2.2936253516225737, -3.2213789550863194, -4.305065593537753, -4.0398563769380305
        ],
        "hbond_acceptor": [
            -6.3771270279199666, -4.213128098212068, -2.7181005369557116, -2.011409131288009, -1.6174700841313647, -1.6533897999632832, 
            -1.8344582237611702, -2.2433732333621994, -2.791975436039971, -3.4451492691913383, -4.086376392572924, -4.011843340710357
        ],
        "num_ring_aliphatic": [
            -0.9068355731033999, -1.0051219455807707, -1.8865098057738656, -3.0812901619156374, -4.028556821684098, -4.2063940660307235
        ],
        "num_ring_aromatic": [
            -3.0470255679415414, -1.9653988256391757, -1.1391218320063197, -1.1854782216897717, -1.9337839692574728, -3.167707538293801
        ],
        "num_rot_bonds": [
            -4.268697949366879, -3.1122660898099426, -2.475748712003235, -2.102914897865478, -1.8438952236485835, -1.9146052992269007, -2.0651442369790116, -2.371935171128839,
             -2.667868411469378, -3.158251203051766, -3.5613661338149765, -3.912023005428146, -4.406319327242926, -4.7914997641795845, -5.083205986931091, -5.683979847360021, 
             -5.572754212249797, -5.626821433520073, -5.952243833954701, -6.319968614080018, -6.571283042360924, -5.115995809754082
        ],
        "molwt": [
            -6.907755278982137, -6.645391014514646, -6.437751649736401, -5.914503505971854, -5.546778725846536, -5.035953102080546, -4.815891217303744, 
            -4.283086686818978, -3.9322257127456655, -3.789805372703897, -3.417326763592039, -3.216378944669614, -3.0597376035299035, -2.875286120478124, 
            -2.722656353491572, -2.6971102610638757, -2.877060743836492, -2.820099704910833, -2.752002088631393, -2.926206210905381, -2.8913722582297483, 
            -2.993734270891318, -3.1190304898984844, -3.2467610283577364, -3.402197881995739, -3.6573807870545654, -3.7009520353482057, -4.000854219134761, 
            -4.254513314374922, -4.382026634673881, -4.55638002181866, -4.767689115485866, -5.099466507802871, -5.149897361429764, -5.403677882205863, -5.599422459331958,
             -5.7763531674910364, -5.809142990314028, -5.991464547107982, -6.3771270279199666, -5.8781358618009785, -6.3771270279199666, -6.437751649736401, -7.418580902748128,
              -7.013115794639964, -7.824046010856292, -7.824046010856292, -10, -8.517193191416238, -10, -7.824046010856292
        ],
        "alogp": [
            -9.210340371976182, -8.517193191416238, -7.418580902748128, -8.517193191416238, -7.264430222920869, -7.600902459542082, -7.264430222920869, -7.418580902748128, 
            -6.437751649736401, -6.645391014514646, -5.991464547107982, -5.683979847360021, -5.626821433520073, -5.099466507802871, -4.892852258439873, -4.456750180869818, 
            -4.006333684899388, -3.8490482062667577, -3.387294476493164, -3.228926160721702, -2.9565115604007097, -2.779009290042704, -2.6283152330833564, -2.5083800059736423, 
            -2.5257286443082556, -2.482908647125327, -2.517016703706234, -2.682382454353632, -2.7822350992915865, -2.9798589243977007, -3.2213789550863194, -3.5267606046375013, 
            -3.7550192566184815, -4.213128098212068, -4.290359446148058, -4.755993075722675, -4.947660494934867, -5.240048458424061, -5.952243833954701, -6.032286541628237, -6.571283042360924, 
            -7.013115794639964, -7.824046010856292, -7.418580902748128, -7.824046010856292, -8.517193191416238, -8.517193191416238, -8.517193191416238, -9.210340371976182, -9.210340371976182, -7.824046010856292
        ],
        "psa": [
            -5.572754212249797, -5.809142990314028, -4.474141923581687, -4.275866438845491, -3.70500883604382, -3.6082215510964817, -3.074775480894444, -3.0449225177447627, -2.875286120478124, 
            -2.6091102532473065, -2.6736487743848776, -2.5232317641096684, -2.5902671654458267, -2.6255489795904663, -2.652142569163914, -2.9040650850281673, -3.036554268074246, 
            -3.1349943408874985, -3.341043458842408, -3.600868576791223, -3.763603000309873, -4.154094566627875, -4.319991243754429, -4.56594947283481, -4.733003557497976, 
            -4.840892519509161, -5.1672891041416324, -5.426150738057921, -5.626821433520073, -5.713832810509703, -6.074846156047033, -6.3771270279199666, -6.502290170873972, 
            -6.319968614080018, -6.119297918617867, -6.571283042360924, -7.1308988302963465, -7.013115794639964, -7.1308988302963465, -7.418580902748128, -7.418580902748128, 
            -7.013115794639964, -8.111728083308073, -7.264430222920869, -8.517193191416238, -8.111728083308073, -8.111728083308073, -8.111728083308073, -8.111728083308073, -7.824046010856292, -6.3771270279199666
        ],
        "alerts": [
            -0.764428382564909, -1.2986498512678444, -1.8263509139976741, -2.647896278282463, -3.7465085669505727, -5.099466507802871
        ],
        "hetero_prop": [
            -7.600902459542082, -9, -9.210340371976182, -6.907755278982137, -6.645391014514646, -5.8781358618009785, -5.472670753692815, -5.149897361429764, -5.2030071867437115, 
            -4.853631545286591, -4.154094566627875, -4.06284589516273, -3.9738984091462335, -3.543913683863751, -3.4357888264317746, -3.2570970376883985, -3.0159349808715104, 
            -3.0159349808715104, -2.8788385220824915, -2.8577109756566164, -2.474560357733856, -3.028255465259551, -2.7045563118479543, -2.866459937849852, -3.085656981081978, 
            -2.91139112512024, -2.928073625080176, -3.0512449834842497, -3.8076629901039034, -3.937340813412436, -3.825845309187094, -4.312500572025272, -4.193060535161258, -4.509860006183766, 
            -4.6356293934728, -4.9062752787720125, -5.240048458424061, -5.035953102080546, -5.7763531674910364, -5.744604469176456, -7.013115794639964, -5.496768305271875, -8.111728083308073, 
            -6.437751649736401, -6.725433722188183, -6.907755278982137, -8.111728083308073, -7.418580902748128, -8.517193191416238, -9.210340371976182, -6.074846156047033
        ],
        "max_ring_size": [
             -4.64599114, -18.42068074, -18.42068074, -7.41856424, -8.51714319, -3.84904774, -0.08675685, 
             -3.27809492, -6.16581317, -7.26441594, -7.13088633, -7.82402101, -6.5712759, -7.26441594,  -4.87960572
        ]
    }

    _PROP_RANGES = {
        'frac_csp3': np.linspace(0,1,51), 'heavy_atom': np.arange(10,61),
        'hbond_donor': np.arange(7), 'hbond_acceptor': np.arange(12),
        'num_ring_aliphatic': np.arange(6), 'num_ring_aromatic': np.arange(6),
        'num_rot_bonds': np.arange(22), 'molwt': np.linspace(100,900,51),
        'alogp': np.linspace(-5,12,51), 'psa': np.linspace(0,300,51),
        'alerts': np.arange(6), 'hetero_prop': np.linspace(0,0.6,51), 'max_ring_size': np.arange(15)
    }

    def __init__(self, relative_weights="inverse_entropy", offset=5.0):
        """Initialize iMinerDrugLikeliness scorer.
        
        Parameters
        ----------
        relative_weights : str or numpy.ndarray, optional
            Method for computing property weights. If "inverse_entropy",
            weights are computed as inverse entropy. If numpy array,
            uses provided weights. Otherwise, uses uniform weights.
            Default is "inverse_entropy".
        offset : float, optional
            Offset value added to the final score. Default is 5.0.
        """
        super().__init__()
        n_total_properties = len(self._LOG_PROPS)
        if relative_weights == "inverse_entropy":
            weight_vectors = []
            for p in self._LOG_PROPS:
                x = np.array(self._LOG_PROPS[p])
                weight_vectors.append(-1 / x.dot(np.exp(x)))
            self.relative_weights = np.array(weight_vectors) / np.sum(weight_vectors)
        elif isinstance(relative_weights, np.ndarray):
            self.relative_weights = relative_weights
        else:
            self.relative_weights = np.array([1/n_total_properties] * n_total_properties)
        
        self.offset = offset
    
    def compute_from_props(self, props: Dict[str, Union[int, float]]):
        """Compute drug likeliness score from molecular properties.
        
        Parameters
        ----------
        props : dict
            Dictionary mapping 13 property names to their values. See the :meth:`compute_props` for more details
        
        Returns
        -------
        float
            Drug likeliness score.
        """
        log_props = np.array([
            _make_onehot(v, self._PROP_RANGES[k]).dot(self._LOG_PROPS[k]) for k, v in props.items()
        ])
        return log_props.dot(self.relative_weights) + self.offset
    
    @classmethod
    def compute_props(cls, mol: Chem.Mol):
        """Compute molecular properties needed for drug likeliness score.
        
        Parameters
        ----------
        mol : rdkit.Chem.Mol
            RDKit molecule object.
        
        Returns
        -------
        dict
            Dictionary of molecular properties including:
            - frac_csp3: Fraction of sp3 carbons
            - heavy_atom: Number of heavy atoms
            - hbond_donor: Number of hydrogen bond donors
            - hbond_acceptor: Number of hydrogen bond acceptors
            - num_ring_aliphatic: Number of aliphatic rings
            - num_ring_aromatic: Number of aromatic rings
            - num_rot_bonds: Number of rotatable bonds
            - molwt: Molecular weight
            - alogp: ALogP value
            - psa: Polar surface area
            - alerts: Number of QED structure alerts
            - hetero_prop: Proportion of heteroatoms
            - max_ring_size: Maximum ring size
        """
        qed_prop = QED.properties(mol)
        props = {
            'frac_csp3': Lipinski.FractionCSP3(mol),
            'heavy_atom': Lipinski.HeavyAtomCount(mol),
            'hbond_donor': Lipinski.NumHDonors(mol),
            'hbond_acceptor': Lipinski.NumHAcceptors(mol),
            'num_ring_aliphatic': Lipinski.NumAliphaticRings(mol),
            'num_ring_aromatic': Lipinski.NumAromaticRings(mol),
            'num_rot_bonds': Lipinski.NumRotatableBonds(mol),
            'molwt': qed_prop.MW,
            'alogp': qed_prop.ALOGP,
            'psa': qed_prop.PSA,
            'alerts': qed_prop.ALERTS
        }
        props['hetero_prop'] = Lipinski.NumHeteroatoms(mol) / props['heavy_atom']
        props['max_ring_size'] = _get_max_ring_size(mol)
        return props

    @accept_smiles
    def compute(self, mol: Chem.Mol):
        """Compute drug likeliness score for a molecule.
        
        Parameters
        ----------
        mol : rdkit.Chem.Mol or str
            Molecule to score, either as RDKit molecule or SMILES string.
        
        Returns
        -------
        float
            Drug likeliness score.
        """
        props = self.compute_props(mol)
        score = self.compute_from_props(props)
        return score