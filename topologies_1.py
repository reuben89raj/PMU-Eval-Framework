# File to store different topologies

def bus118():

	pmuNetwork = {'P2': ['S1'], 
        'P5': ['S1'], 
        'P9': ['S1'], 
        'P11': ['S1'], 
        'P12': ['S1'],
        'P110': ['S1'], 
        'P17': ['S2'], 
        'P21': ['S2'], 
        'P24': ['S2'],
        'P25': ['S2'],
        'P28': ['S2'],
        'P114': ['S2'],
        'P34': ['S3'],
        'P37': ['S3'],
        'P40': ['S3'],
        'P45': ['S3'],
        'P49': ['S3'],
        'P52': ['S4'],
        'P56': ['S4'],
        'P62': ['S4'],
        'P63': ['S4'],
        'P68': ['S4'],
        'P73': ['S5'],
        'P75': ['S5'],
        'P77': ['S5'],
        'P80': ['S5'],
        'P85': ['S5'],
        'P86': ['S6'],
        'P90': ['S6'],
        'P94': ['S6'],
        'P101': ['S6'],
        'P105': ['S6'],
        'S1': ['PDC1', 'P2', 'P5', 'P9', 'P11', 'P12', 'P110', 'S2', 'S6'],
        'S2': ['PDC2', 'P17', 'P21', 'P24', 'P25', 'P28', 'P114', 'S1', 'S3'],
        'S3': ['PDC3', 'P34', 'P37', 'P40', 'P45', 'P49', 'S2', 'S4'],
        'S4': ['PDC4', 'P52', 'P56', 'P62', 'P63', 'P68', 'S3', 'S5'],
        'S5': ['PDC5', 'P73', 'P75', 'P77', 'P80', 'P85', 'S4', 'S6'],
        'S6': ['PDC6', 'P86', 'P90', 'P94', 'P101', 'P105', 'S5', 'S1'],
        'PDC1' : ['S1'],
        'PDC2': ['S2'],
        'PDC3': ['S3'],
        'PDC4': ['S4'],
        'PDC5': ['S5'],
        'PDC6': ['S6']
        }


#List of all basic paths from each PMU to a PDC
	masterPathList = { 'P2': ['S1', 'PDC1'], 'P5': ['S1', 'PDC1'], 'P9': ['S1', 'PDC1'], 'P11': ['S1', 'PDC1'], 'P12': ['S1', 'PDC1'], 'P110': ['S1', 'PDC1'], 'P17': ['S2', 'PDC2'], 'P21': ['S2', 'PDC2'], 'P24': ['S2', 'PDC2'], 'P25': ['S2', 'PDC2'], 'P28': ['S2', 'PDC2'], 'P114': ['S2', 'PDC2'], 'P34': ['S3', 'PDC3'], 'P37': ['S3', 'PDC3'], 'P40': ['S3', 'PDC3'], 'P45': ['S3', 'PDC3'], 'P49': ['S3', 'PDC3'], 'P52': ['S4', 'PDC4'], 'P56': ['S4', 'PDC4'], 'P62': ['S4', 'PDC4'], 'P63': ['S4', 'PDC4'], 'P68': ['S4', 'PDC4'], 'P73': ['S5', 'PDC5'], 'P75': ['S5', 'PDC5'], 'P77': ['S5', 'PDC5'], 'P80': ['S5', 'PDC5'], 'P85': ['S5', 'PDC5'], 'P86': ['S6', 'PDC6'], 'P90': ['S6', 'PDC6'], 'P94': ['S6', 'PDC6'], 'P101': ['S6', 'PDC6'], 'P105': ['S6', 'PDC6']}

	numBuses = 118

	pmuBusMapping = { 'P2': ['B2', 'B1', 'B12'],
        'P5':   ['B5', 'B4', 'B3', 'B11', 'B6', 'B8'],
        'P9':   ['B8', 'B10', 'B9'],
        'P11':  ['B11', 'B4', 'B5', 'B12', 'B13', 'B11'],
        'P12':  ['B2', 'B3', 'B11', 'B7', 'B117', 'B14', 'B12', 'B16'],
        'P17':  ['B16', 'B15', 'B18', 'B30', 'B31', 'B17', 'B113'],
        'P21':  ['B21', 'B20', 'B22'],
        'P24':  ['B24', 'B23', 'B72', 'B70'],
        'P25':  ['B25', 'B26', 'B23'],
        'P28':  ['B28', 'B27', 'B29'],
        'P34':  ['B34', 'B19', 'B36', 'B37', 'B43'],
        'P37':  ['B37', 'B33', 'B34', 'B35', 'B38', 'B39', 'B40'],
        'P40':  ['B40', 'B39', 'B37', 'B41', 'B42'],
        'P45':  ['B45', 'B44', 'B46', 'B49'],
        'P49':  ['B49', 'B42', 'B54', 'B50', 'B51', 'B66', 'B69', 'B47', 'B48'],
        'P52':  ['B52', 'B53', 'B51'],
        'P56':  ['B56', 'B54', 'B55', 'B57', 'B58', 'B59'],
        'P62':  ['B62', 'B60', 'B61', 'B66', 'B67'],
        'P63':  ['B63', 'B59', 'B64'],
        'P68':  ['B68', 'B65', 'B69', 'B116', 'B81'],
        'P73':  ['B73', 'B71'],
        'P75':  ['B75', 'B74', 'B71', 'B69', 'B118', 'B77'],
        'P77':  ['B77', 'B76', 'B69', 'B78', 'B80', 'B82'],
        'P80':  ['B80', 'B79', 'B81', 'B99', 'B98', 'B96', 'B97', 'B77'],
        'P85':  ['B85', 'B84', 'B83', 'B88', 'B89', 'B86'],
        'P86':  ['B86', 'B87'],
        'P90':  ['B90', 'B89', 'B91'],
        'P94':  ['B94', 'B95', 'B96', 'B100', 'B93', 'B92'],
        'P101': ['B101', 'B100', 'B102'],
        'P105': ['B105', 'B104', 'B106', 'B107', 'B108', 'B103'],
        'P110': ['B110', 'B103', 'B109', 'B111', 'B112'],
        'P114': ['B114', 'B32', 'B115']
	}
	return pmuNetwork, masterPathList, pmuBusMapping, numBuses
