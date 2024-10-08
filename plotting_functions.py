import numpy as np
import pandas as pd
import pyDEW_functions as pDF
import pyDEW
from thermoengine import model


def condenseMinerals(mineralDF, system, cutoff=0.1, ordered=True, cumulative=True,
                     retainTime=False):
    """
    Function to reduce solid solution endmember volumes into solid solution series mineral volumes,
    e.g., FORSTERITE and FAYALITE are reduced to OLIVINE(SS).

    INPUTS
    ------
    mineralDF : pandas.DataFrame
        mineral volume output returned by decompression relay
    system : pyDEW.System object
        pyDEW.System for calculations
    cutoff : float
        minimum threshold value to accept in output DataFrame. Default is 0.1
    ordered : boolean
        order the minerals in ascending order of proportion based on first row, with serpentines
        sent to the back. Default is True
    cumulative : boolean
        return cumulative sum of minerals from left to right in DataFrame. Default is True
    retainTime : boolean
        if there is a 'Time' column, this is retained in the final condensed mineral DataFrame.
        Default is True

    OUTPUTS
    -------
    outputDF : pandas.DataFrame
        mineral volume output with reduced mineral series
    """

    systemMinerals = [i.props['phase_name'] for i in system.minerals]
    systemSolidSolutions = list(system.solid_solutions.keys())
    seriesKeys = mineralDF.columns

    # PART 1: FIND THE SOLID SOLUTION SERIES
    keyMinerals = []
    for i in mineralDF.columns:
        if '_volPER' in i:
            minString = i.replace('_volPER', '')
        else:
            continue
        if '(CPX)' in minString:
            keyMineral = 'CLINOPYROXENE(SS)'
        elif '(KSP)' in minString:
            keyMineral = 'KFELDSPAR(SS)'
        elif minString not in systemMinerals:
            # This allows us to collect all minerals regardless of the DataFrame columns
            continue
        else:
            if any(minString in j for j in system.solid_solutions.values()):
                keyMineral = [k for k, v in system.solid_solutions.items() if minString in v][0]
            else:
                keyMineral = minString
        if keyMineral not in keyMinerals:
            keyMinerals.append(keyMineral)

    # PART 2: SUM THE VOLUMES
    volumesList = []
    for i in keyMinerals:
        if i in systemSolidSolutions:
            endmembers = system.solid_solutions[i]
            if i == 'CLINOPYROXENE(SS)':
                endmembers = [j + '(CPX)' for j in endmembers]
            elif i == 'KFELDSPAR(SS)':
                endmembers = [j + '(KSP)' for j in endmembers]
            endmembers = [j + '_volPER' for j in endmembers]
            endmembers = [j for j in endmembers if j in seriesKeys]
            volumes = [mineralDF[j].fillna(0).tolist() for j in endmembers]
            summedVolumes = [np.sum(j) for j in zip(*volumes)]
        else:
            summedVolumes = mineralDF[i + '_volPER'].fillna(0).tolist()
        volumesList.append(summedVolumes)
    outputDF = pd.DataFrame(volumesList).T
    outputDF.columns = keyMinerals
    outputDF = outputDF[outputDF.columns[outputDF.max() > 0.1]]
    if ordered:
        outputDF = outputDF[outputDF.iloc[0].sort_values(ascending=False).index]
        if 'ANTIGORITE' in keyMinerals:
            outputDF['ANTIGORITE'] = outputDF.pop('ANTIGORITE')
        if 'CHRYSOTILE' in keyMinerals:
            outputDF['CHRYSOTILE'] = outputDF.pop('CHRYSOTILE')
    if cumulative:
        outputDF = outputDF.cumsum(axis=1)
    if retainTime:
        retainedTimes = mineralDF['Time'].tolist()
        outputDF['Time'] = retainedTimes
    return outputDF
