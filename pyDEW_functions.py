# Additional functions for volume percent calculations: 31/05/24
# Additional functions for element conservation and fluid mass calculations: 22/10/24

import numpy as np
import pyDEW
from thermoengine import model
import pandas as pd
from copy import deepcopy
import linecache


def findReactantMass(reactant, system):
    """
    Find the total mass in g of a list of dictionaries forming a reactant.

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    system : pyDEW.System object
        pyDEW.System for calculations

    RETURNS
    -------
    mass : float
        mass of solid in grams

    """
    
    # Atomic masses
    atomicMasses = {
        "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012,
        "B": 10.811, "C": 12.011, "N": 14.007, "O": 15.999,
        "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
        "Al": 26.982, "Si": 28.086, "P": 30.974, "S": 32.065,
        "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
        "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
        "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
        "Cu": 63.546, "Zn": 65.390, "Ga": 69.723, "Ge": 72.640,
        "As": 74.922, "Se": 78.960, "Br": 79.904, "Kr": 83.798,
        "Rb": 85.468, "Sr": 87.620, "Y": 88.906, "Zr": 91.224,
        "Nb": 92.906, "Mo": 95.950, "Tc": 98.000, "Ru": 101.070,
        "Rh": 102.906, "Pd": 106.420, "Ag": 107.868, "Cd": 112.411,
        "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.600,
        "I": 126.904, "Xe": 131.293, "Cs": 132.906, "Ba": 137.327,
        "La": 138.906, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242,
        "Pm": 145.000, "Sm": 150.360, "Eu": 151.964, "Gd": 157.250,
        "Tb": 158.925, "Dy": 162.500, "Ho": 164.930, "Er": 167.259,
        "Tm": 168.934, "Yb": 173.055, "Lu": 174.967, "Hf": 178.490,
        "Ta": 180.948, "W": 183.840, "Re": 186.207, "Os": 190.230,
        "Ir": 192.217, "Pt": 195.084, "Au": 196.967, "Hg": 200.590,
        "Tl": 204.383, "Pb": 207.200, "Bi": 208.980, "Po": 209.000,
        "At": 210.000, "Rn": 222.000, "Fr": 223.000, "Ra": 226.000,
        "Ac": 227.000, "Th": 232.038, "Pa": 231.036, "U": 238.029,
        "Np": 237.000, "Pu": 244.000, "Am": 243.000, "Cm": 247.000,
        "Bk": 247.000, "Cf": 251.000, "Es": 252.000, "Fm": 257.000,
        "Md": 258.000, "No": 259.000, "Lr": 262.000, "Rf": 267.000,
        "Db": 270.000, "Sg": 271.000, "Bh": 270.000, "Hs": 277.000,
        "Mt": 276.000, "Ds": 281.000, "Rg": 280.000, "Cn": 285.000,
        "Nh": 284.000, "Fl": 289.000, "Mc": 288.000, "Lv": 293.000,
        "Ts": 294.000, "Og": 294.000}


    # Find all minerals and molecular masses in system
    systemMineralNames = [i.phase_name for i in system.minerals]
    systemSolidSolutions = list(system.solid_solutions.keys())

    masses = []
    for i in range(len(reactant)):
        mineral = reactant[i]
        mineralName = mineral['reactant']
        mineralMoles = mineral['moles']

        # If the mineral is a solid solution mineral, sort each solid solution endmember out
        if mineralName in systemSolidSolutions:

            # Obtain the endmembers of that solid solution
            systemEndmembers = system.solid_solutions[mineralName]
            mineralEndmembers = list(mineral['composition'].keys())

            # Iterate over all endmembers in the mineral
            for j in mineralEndmembers:
                # Find index of the endmember in the system minerals, and use data to obtain mass
                endmemberIndex = systemMineralNames.index(j)
                endmemberElements = system.minerals[endmemberIndex].element_comp[1:]
                endmemberMr = np.sum([i * j for i, j in zip(endmemberElements, atomicMasses.values())])
                endmemberProportion = mineral['composition'][j]
                
                endmemberMass = endmemberMr * mineralMoles * endmemberProportion
                masses.append(endmemberMass)

        else:
            # If the mineral is not a solid solution then we can just perform the last calculation
            endmemberIndex = systemMineralNames.index(mineralName)
            endmemberElements = system.minerals[endmemberIndex].element_comp[1:]
            endmemberProportion = 1.0
            endmemberMr = np.sum([i * j for i, j in zip(endmemberElements, atomicMasses.values())])
            endmemberMass = endmemberMr * mineralMoles * endmemberProportion
            masses.append(endmemberMass)
    return np.sum(masses)



def findMineralVolume(mineral, system, moles, P, T):
    """
    Calculate volume of a mineral in cubic centimetres.

    PARAMETERS
    ----------
    mineral : string
        string of pyDEW mineral
    system : pyDEW.System object
        pyDEW.System for calculations
    moles : float
        moles of mineral
    P : float
        pressure in bars
    T : float
        temperature in K

    RETURNS
    -------
    mineralVolume : float
        mineral volume in cubic centimetres

    """

    systemMinerals = [i.abbrev for i in system.minerals]
    if mineral not in systemMinerals:
        mineralVolume = np.nan
    else:
        mineralObject = system.minerals[systemMinerals.index(mineral)]
        mineralVolume = mineralObject.volume(T, P) * moles * 10.0  # get from J/bar to cm3
    return mineralVolume


def findReactantVolume(reactant, system, P, T):
    """
    Find the total sum of volumes of minerals in a reactant in cubic centimetres.

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    system : pyDEW.System object
        pyDEW.System for calculations
    P : float
        pressure in bars
    T : float
        temperature in K

    RETURNS
    -------
    volume : float
        volume of solid in cubic centimetres
    """

    systemMinerals = [i.abbrev for i in system.minerals]
    mineralVolumes = []

    for i in reactant:
        if 'composition' in i.keys():
            SSmoles = i['moles']
            endmembers = list(i['composition'].keys())
            endmemberProportions = list(i['composition'].values())
            for j in range(len(i['composition'].keys())):
                mineralEndmember = endmembers[j]
                mineralEndmemberMoles = endmemberProportions[j] * SSmoles
                mineralVolumes.append(findMineralVolume(mineralEndmember, system,
                                                        mineralEndmemberMoles, P, T))
        else:
            mineral = i['reactant']
            mineralMoles = i['moles']
            mineralVolumes.append(findMineralVolume(mineral, system, mineralMoles, P, T))
    return np.sum(mineralVolumes)


def findReactantDensity(reactant, system, P, T):
    """
    Find the total density of a reactant in g/cm3.

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    system : pyDEW.System object
        pyDEW.System for calculations
    P : float
        pressure in bars
    T : float
        temperature in K

    RETURNS
    -------
    density : float
        density of solid in g/cm3
    """

    reactantMass = findReactantMass(reactant, system)
    reactantVolume = findReactantVolume(reactant, system, P, T)
    return reactantMass / reactantVolume


def reactantToFR(reactant, FRR, system):
    """
    Recalculate reactant for a defined fluid-rock ratio (relative to 1 kg fluid).

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    FRR : float
        fluid-rock ratio
    system : pyDEW.System object
        pyDEW.System for calculations

    RETURNS
    -------
    correctedReactants : list of dictionaries
        pyDEW.Reaction mineral input

    """

    uncorrectedReactantMass = findReactantMass(reactant, system)
    correctedReactants = deepcopy(reactant)
    ratio = 1000.0 / uncorrectedReactantMass

    # This converts the corrected reactants to 1 kg, and then to the correct FRR
    for i in correctedReactants:
        i['moles'] = i['moles'] * ratio / FRR
    return correctedReactants


def reactantToMass(reactant, newMass, system):
    """
    Recalculate reactant for a defined mass.

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    mass : float
        new mass of the solid in grams
    system : pyDEW.System object
        pyDEW.System for calculations

    RETURNS
    -------
    correctedReactants : list of dictionaries
        pyDEW.Reaction mineral input

    """
    uncorrectedReactantMass = findReactantMass(reactant, system)
    correctedReactants = deepcopy(reactant)
    ratio = newMass / uncorrectedReactantMass
    for i in correctedReactants:
        i['moles'] = i['moles'] * ratio
    return correctedReactants


def findSolidComposition(outputFile, system, P, T):
    """
    Reaction to find the running mineral composition of the total solid from 'grand summary' (g.s.)
    tables. Solid is returned as a pd.DataFrame without sorting into solid solutions.

    PARAMETERS
    ----------
    outputFile : string
        path to output file generated by EQ6
    system : pyDEW.System object
        pyDEW.System for calculations
    P : float
        pressure in bars
    T : float
        temperature in K

    RETURNS
    -------
    exportDF : pandas.DataFrame
        DataFrame of solid composition with columns: 'phase/endmember', 'moles', 'grams',
        'volume, cc', 'log zi'

    """

    # Define lookups as 'bookmarks' for where to extract data
    lookup1 = 'log of reaction progress'
    lookup2 = 'grand summary'
    lookup3 = 'mineral saturation state summary'

    # Empty lists for storage
    logZi = []
    startLineNumber = []
    endLineNumber = []

    # Open file, extract info, read file
    file = open(outputFile, 'r')
    for num, line in enumerate(file):

        # Lookup 1 is where we find our running reaction progress
        if lookup1 in line:
            logZi.append(float(line.split()[-1]))

        # Lookup 2 is start of the g.s. table, add 4 here to get to data (headers skipped)
        if lookup2 in line:
            startLineNumber.append(num + 4)

        # Lookup 3 is start of the table following the g.s. table, subtract 12 to get end of g.s.
        if lookup3 in line:
            endLineNumber.append(num - 12)
    file.close()

    # Line number to get our headers is selected from the final g.s. table
    beginHeaders = startLineNumber[-1]-1
    headers = linecache.getline(outputFile, beginHeaders).strip().split('  ')
    headers = [i.strip() for i in headers if i != '']

    for i in range(len(logZi)):

        # Extract all datalines in the g.s. tables
        start = startLineNumber[i]
        end = endLineNumber[i]
        dataLines = []
        for j in range(start, end):
            dataLine = linecache.getline(outputFile, j).strip().split(' ')
            dataLine = [k.strip() for k in dataLine if k != '']
            if dataLine == []:
                continue
            else:
                dataLines.append(dataLine)

        # Convert to floats, except for the mineral names in index 0
        for j in dataLines:
            for k in range(1, len(j)):
                j[k] = float(j[k])

        # Export to pd.DataFrame
        if i == 0:
            exportDF = pd.DataFrame(dataLines)
            exportDF.columns = headers
            exportDF['log zi'] = len(exportDF)*[logZi[i]]
        else:
            tempDF = pd.DataFrame(dataLines)
            tempDF.columns = headers
            tempDF['log zi'] = len(tempDF)*[logZi[i]]
            exportDF = pd.concat([exportDF, tempDF])

    # Calculate the true mineral volume from moles and pressure/temperature of system
    exportDF['volume, cc'] = [findMineralVolume(exportDF['phase/end-member'].iloc[i],
                                                system, exportDF['moles'].iloc[i],
                                                P, T) for i in range(len(exportDF))]
    return exportDF


def findLastSolid(outputFile, system, P, T, notNaNRow=True):
    """
    Returns the composition of the last solid in the same format as required for EQ6 reactants.

    PARAMETERS
    ----------
    outputFile : string
        path to pyDEW.Reaction output file
    system : pyDEW.System
        pyDEW.System for calculations
    P : float
        pressure in bars
    T : float
        temperature in K

    RETURNS
    -------
    reactants : list of dictionaries
        pyDEW.Reaction mineral input

    """

    solidCompositions = findSolidComposition(outputFile, system, P, T)
    maxLogzi = np.max(solidCompositions['log zi'])
    lastSolidDF = solidCompositions[solidCompositions['log zi'] == maxLogzi]

    # Find composition of last solid
    lastSolidMinerals = lastSolidDF['phase/end-member'].tolist()

    # Search the output for each solid solution and format into dictionary
    searchedMinerals = []
    reactants = []
    for i in range(len(lastSolidMinerals)):
        mineral = lastSolidMinerals[i]

        # If mineral has already been searched, continue, else append to list of searched minerals
        if mineral in searchedMinerals:
            continue
        searchedMinerals.append(mineral)

        # If the mineral has solid solutions, calculate for endmembers, else calculate for mineral
        if mineral in list(system.solid_solutions.keys()):
            phases = []  # List to keep track of phases comprising solid solutions
            moles = []  # List to keep information about solid solution molar composition
            volumes = []  # List to keep track of volumes

            # Find mineral solid solution phases, iterate over them
            mineralSS = system.solid_solutions[mineral]
            for j in range(len(mineralSS)):
                dfRow = lastSolidDF[lastSolidDF['phase/end-member'] == mineralSS[j]]
                phases.append(mineralSS[j])
                moles.append(dfRow['moles'].iloc[0])
                volumes.append(dfRow['volume, cc'].iloc[0])
            searchedMinerals.extend(phases)

            # If the number of moles is zero, then the mineral is no longer in the system
            totalMoles = np.sum(moles)
            if totalMoles == 0.0:
                continue

            # Find molar ratio of each solid solution for composition
            molarRatios = [j/totalMoles for j in moles]

            # Formatting into dictionary
            minDict = {'reactant' : mineral}
            minDict['moles'] = totalMoles
            minDict['composition'] = {phases[i]: molarRatios[i] for i in range(len(phases))}
            minDict['volume'] = {phases[i]: volumes[i] for i in range(len(phases))}

        else:
            dfRow = lastSolidDF[lastSolidDF['phase/end-member'] == mineral]

            # If the mineral is not present, then skip the mineral
            if dfRow['moles'].iloc[0] == 0.0:
                continue

            # Formatting into dictionary
            minDict = {'reactant': mineral}
            minDict['moles'] = dfRow['moles'].iloc[0]
            minDict['volume'] = dfRow['volume, cc'].iloc[0]

        # Append dictionaries to list, matching required format
        reactants.append(minDict)
    return reactants


def findSolventMass(tabFile, index=-1):
    """
    Function to find  H2O solvent masses before/after reaction and the ratio between them. All
    masses are presented in grams.

    PARAMETERS
    ----------
    tabfile : string
        path to pyDEW.Reaction tab file
    index : integer
        index of solvent mass to compare to initial solvent mass. Default is -1 (final item)

    RETURNS
    -------
    tuple
        in order: mass of H2O solvent at start of reaction, mass of H2O solvent at end of reaction,
        ratio of mass H2O solvent after reaction to mass H2O solvent before reaction

    """

    # Find start of table
    lookup = 'log fo2'
    file = open(tabFile, 'r')

    # Track where the start of the headers are, data starts 3 lines after the headers
    for num, line in enumerate(file):
        if lookup in line:
            beginHeaders = num + 1
            beginData = num + 3
            break
    file.close()

    # Read headers from the file, trim away excess whitespace
    headers = linecache.getline(tabFile, beginHeaders).strip().split('  ')
    headers = [i.strip() for i in headers if i != '']

    # Determine number of lines in the data by finding where after the headers we find a big break
    count = 0
    while linecache.getline(tabFile, beginData+count).strip() != '':
        count += 1

    # Read data
    dataLines = []
    for i in range(beginData, beginData+count):
        # Get line, split by whitespace to identify values
        dataLine = linecache.getline(tabFile, i).strip().split(' ')

        # Remove whitespaces, then convert strings to float
        dataLine = [j.strip() for j in dataLine if j != '']

        # This check is to make sure that there aren't two decimals sticking to each other
        check = [j.count('.') for j in dataLine]
        if 2 in check:
            checkIndex = check.index(2)
            errorValue = dataLine[checkIndex]
            value1 = errorValue[:8]
            value2 = errorValue[8:]
            dataLine.insert(checkIndex+1, value1)
            dataLine.insert(checkIndex+2, value2)
            dataLine.pop(checkIndex)
        dataLine = [float(j) for j in dataLine]
        dataLines.append(dataLine)

    # Convert data to dataframe, find ratio of H2O at start and at end
    pHData = pd.DataFrame(data=dataLines, columns=headers)
    H2Okg = pHData['kg h2o']
    H2OChange = H2Okg.iloc[index]/H2Okg.iloc[0]
    return 1000.0*H2Okg.iloc[0], 1000.0*H2Okg.iloc[index], H2OChange


def reactantToDF(reactant, system):
    """
    Convert reactant list of dictionaries to pd.DataFrame.

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    system : pyDEW.System object
        pyDEW.System for calculations

    RETURNS
    -------
    exportDF : pandas.DataFrame
        single row pandas DataFrame of mineral moles

    """

    systemMinerals = [i.phase_name for i in system.minerals]
    systemSolidSolutions = list(system.solid_solutions.keys())
    names = []
    moles = []
    for i in range(len(reactant)):
        mineral = reactant[i]
        mineralName = mineral['reactant']
        mineralMoles = mineral['moles']

        # If the mineral is a solid solution mineral, sort each solid solution endmember out
        if mineralName in systemSolidSolutions:

            # Obtain the endmembers of that solid solution
            systemEndmembers = system.solid_solutions[mineralName]
            mineralEndmembers = list(mineral['composition'].keys())

            # Iterate over all endmembers in the mineral
            for j in mineralEndmembers:
                # Find index of the endmember in the system minerals, and use data to obtain mass
                # Exceptions for CPX and KSP, as they share endmembers with other solid solutions
                # We add a small suffix here which can be removed at a later stage
                if mineralName == 'CLINOPYROXENE(SS)':
                    names.append(j+'(CPX)')
                elif mineralName == 'KFELDSPAR(SS)':
                    names.append(j+'(KSP)')
                else:
                    names.append(j)
                endmemberProportion = mineral['composition'][j]
                endmemberMoles = mineralMoles * endmemberProportion
                moles.append(endmemberMoles)

        else:
            # If the mineral is not a solid solution then we can just perform the last calculation
            names.append(mineralName)
            moles.append(mineral['moles'])

    exportDF = pd.DataFrame([moles])
    exportDF.columns = names
    return exportDF


def reactantToVolDF(reactant, system, P, T):
    """
    Convert reactant list of dictionaries to pd.DataFrame of mineral vol%, adapted from
    reactantToDF.

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    system : pyDEW.System object
        pyDEW.System for calculations
    P : float
        pressure in bars
    T : float
        temperature in K

    RETURNS
    -------
    exportDF : pandas.DataFrame
        single row pandas DataFrame of mineral vol%

    """

    systemMinerals = [i.phase_name for i in system.minerals]
    systemSolidSolutions = list(system.solid_solutions.keys())

    names = []
    volPer = []
    for i in range(len(reactant)):
        mineral = reactant[i]
        mineralName = mineral['reactant']
        mineralMoles = mineral['moles']

        # If the mineral is a solid solution mineral, sort each solid solution endmember out
        if mineralName in systemSolidSolutions:

            # Obtain the endmembers of that solid solution
            systemEndmembers = system.solid_solutions[mineralName]
            mineralEndmembers = list(mineral['composition'].keys())

            # Iterate over all endmembers in the mineral
            for j in mineralEndmembers:
                # Find index of the endmember in the system minerals, and use data to obtain mass
                # Exceptions for CPX and KSP, as they share endmembers with other solid solutions
                # We add a small suffix here which can be removed at a later stage
                if mineralName == 'CLINOPYROXENE(SS)':
                    names.append(j+'(CPX)')
                elif mineralName == 'KFELDSPAR(SS)':
                    names.append(j+'(KSP)')
                else:
                    names.append(j)
                endmemberProportion = mineral['composition'][j]
                endmemberMoles = mineralMoles * endmemberProportion
                volPer.append(findMineralVolume(j, system, endmemberMoles, P, T))

        else:
            # If the mineral is not a solid solution then we can just perform the last calculation
            names.append(mineralName)
            volPer.append(findMineralVolume(mineralName, system, mineral['moles'], P, T))

    # Normalise and convert to percentage volumes
    totalVolume = findReactantVolume(reactant, system, P, T)
    volPer = [100*i/totalVolume for i in volPer]

    # Alter column header titles
    names = [i + '_volPER' for i in names]
    exportDF = pd.DataFrame([volPer])
    exportDF.columns = names
    return exportDF


def DFtoReactant(series, system):
    """
    Convert pandas.Series object into a formatted reactant list of dictionaries.

    PARAMETERS
    ----------
    series : pandas.Series object
        pandas Series as formatted in the previous function
    system : pyDEW.System object
        pyDEW.System for calculations

    RETURNS
    -------
    reactantsList : list of dictionaries
        pyDEW.Reaction mineral input

    """

    systemMinerals = [i.phase_name for i in system.minerals]
    systemSolidSolutions = list(system.solid_solutions.keys())
    seriesKeys = series.keys().tolist()
    keyMinerals = []
    for i in seriesKeys:
        # If the mineral is a solid solution value, find the mineral family
        # Exceptions for CPX and KSP, as they share endmembers with other solid solutions
        # We identify the suffix added in reactantToDF and remove it
        if '(CPX)' in i:
            keyMineral = 'CLINOPYROXENE(SS)'
        elif '(KSP)' in i:
            keyMineral = 'KFELDSPAR(SS)'
        elif i not in systemMinerals:
            # This allows us to collect all minerals regardless of the DataFrame columns
            continue
        else:
            if any(i in j for j in system.solid_solutions.values()):
                keyMineral = [k for k, v in system.solid_solutions.items() if i in v][0]
            else:
                keyMineral = i
        if keyMineral not in keyMinerals:
            keyMinerals.append(keyMineral)

    # Now start from key minerals and work way towards endmembers
    reactantsList = []
    for i in keyMinerals:
        if i in systemSolidSolutions:
            endmembers = system.solid_solutions[i]

            # Make sure the code recognises our suffixes when translating into a pyDEW reactant
            if i == 'CLINOPYROXENE(SS)':
                endmembers = [j + '(CPX)' for j in endmembers]
            elif i == 'KFELDSPAR(SS)':
                endmembers = [j + '(KSP)' for j in endmembers]
            endmembers = [j for j in endmembers if j in seriesKeys]
            totalMoles = np.sum([series[j] for j in endmembers])

            # Cases where the number of total moles is zero or NaN
            if totalMoles == 0.0:
                continue
            if np.isnan(totalMoles):
                continue
            endmemberProportions = [series[j]/totalMoles for j in endmembers]

            # Undo any suffix additions before reformatting into a pyDEW reactant
            if i == 'CLINOPYROXENE(SS)' or i == 'KFELDSPAR(SS)':
                endmembers = [j[:-5] for j in endmembers]

            minDict = {'reactant': i,
                       'moles': totalMoles,
                       'composition': {
                           endmembers[j] : endmemberProportions[j] for j in range(len(endmembers))
                       }
                      }
        else:
            if series[i] == 0.0:
                continue
            if np.isnan(series[i]):
                continue
            minDict = {'reactant': i,
                       'moles': series[i]}
        reactantsList.append(minDict)

    # This single-liner removes all the duplicates in the list (which can happen)
    reactantsList = [i for n, i in enumerate(reactantsList) if i not in reactantsList[n + 1:]]

    # Get rid of any NaNs and zeros
    popList = []
    for i in range(len(reactantsList)):
        if np.isnan(reactantsList[i]['moles']):
            popList.append(i)
        elif reactantsList[i]['moles'] == 0.0:
            popList.append(i)
    reactantsList = [i for j, i in enumerate(reactantsList) if j not in popList]
    return reactantsList


def volReactantToReactant(volumeReactant, system, P, T, mass=1000.0):
    """
    Convert reactant volume dictionary into reactant moles dictionary.
    A reactant volume dictionary takes the same form as a pyDEW reactant input dictionary, but has
    'moles' replaced by 'molar', the molar proportion (vol%) of the mineral in the reactant.
    Just a bit of additional functionality so we can convert modal proportions to moles easily.

    PARAMETERS
    ----------
    volumeReactant : list of dictionaries
        dictionary of reactant modal proportions
    system : pyDEW.System object
        pyDEW.System for calculations
    P : float
        pressure in bars
    T : float
        temperature in K
    mass : float
        mass of reactants to return in grams. Default is 1000.0

    RETURNS
    -------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input

    """

    systemMinerals = [i.phase_name for i in system.minerals]
    systemSolidSolutions = list(system.solid_solutions.keys())

    # Quick check to make sure that modal sums up to 100, and normalises if it doesn't
    volumeReactantModalSum = sum([i['modal'] for i in volumeReactant])
    if volumeReactantModalSum != 100.0:
        for i in volumeReactant:
            i['modal'] = i['modal'] * 100.0 / volumeReactantModalSum

    reactant = []
    for i in range(len(volumeReactant)):
        mineral = volumeReactant[i]
        mineralName = mineral['reactant']
        mineralVolProp = mineral['modal'] / 100.0

        # If the mineral is a solid solution mineral, sort each solid solution endmember out
        if mineralName in systemSolidSolutions:

            # Obtain the endmembers of that solid solution
            systemEndmembers = system.solid_solutions[mineralName]
            mineralEndmembers = list(mineral['composition'].keys())
            mineralVolume = 0.0

            # Iterate over all endmembers in the mineral
            for j in mineralEndmembers:

                # Find the volume contributing to that molar fraction
                proportionalVolume = findMineralVolume(j, system, mineral['composition'][j], P, T)
                mineralVolume = mineralVolume + proportionalVolume
            # mineralVolume is the volume of 1 mole of reactant (including solid solutions)
            # mineralMoles is therefore the moles of 1 unit of mineral volume
            mineralMoles = mineralVolProp / mineralVolume
            molDictEntry = {'reactant': mineral['reactant'],
                            'moles': mineralMoles,
                            'composition': mineral['composition']}
            reactant.append(molDictEntry)

        else:
            # If the mineral is not a solid solution then we can just perform the last calculation
            mineralVolume = findMineralVolume(mineralName, system, 1.0, P, T)
            mineralMoles = mineralVolProp / mineralVolume
            molDictEntry = {'reactant': mineral['reactant'],
                            'moles': mineralMoles}
            reactant.append(molDictEntry)

    # Normalise and convert to percentage volumes
    totalMass = findReactantMass(reactant, system)
    totalMoles = sum([i['moles'] for i in reactant])
    for i in reactant:
        i['moles'] = i['moles'] * mass / totalMass
    return reactant


def reactantToVolReactant(reactant, system, P, T):
    """
    Convert reactant moles dictionary into reactant volume dictionary.

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    system : pyDEW.System object
        pyDEW.System for calculations
    P : float
        pressure in bars
    T : float
        temperature in K

    RETURNS
    -------
    volumeReactant : list of dictionaries
        pyDEW.Reaction mineral input formatted for volumes

    """

    systemMinerals = [i.phase_name for i in system.minerals]
    systemSolidSolutions = list(system.solid_solutions.keys())
    totalVolume = findReactantVolume(reactant, system, P, T)

    volumeReactant = []
    for i in range(len(reactant)):
        mineral = reactant[i]
        mineralName = mineral['reactant']
        mineralMoles = mineral['moles']

        # If the mineral is a solid solution mineral, sort each solid solution endmember out
        if mineralName in systemSolidSolutions:
            mineralVolume = []

            # Obtain the endmembers of that solid solution
            systemEndmembers = system.solid_solutions[mineralName]
            mineralEndmembers = list(mineral['composition'].keys())

            # Iterate over all endmembers in the mineral
            for j in mineralEndmembers:
                endmemberProportion = mineral['composition'][j]
                endmemberMoles = mineralMoles * endmemberProportion
                mineralVolume.append(findMineralVolume(j, system, endmemberMoles, P, T))

            # Append to volume reactant
            volDictEntry = {'reactant': mineral['reactant'],
                            'modal': 100.0 * np.sum(mineralVolume) / totalVolume,
                            'composition': mineral['composition']}
            volumeReactant.append(volDictEntry)

        else:
            # If the mineral is not a solid solution then we can just perform the last calculation
            mineralVolume = findMineralVolume(mineralName, system, mineralMoles, P, T)
            volDictEntry = {'reactant': mineral['reactant'],
                            'modal': 100.0 * mineralVolume / totalVolume}
            volumeReactant.append(volDictEntry)
    return volumeReactant


def volReactantToVolDF(volReactant, system, P, T):
    """
    Convert volume percentage reactant list of dictionaries to pd.DataFrame of mineral vol%.

    PARAMETERS
    ----------
    volReactant : list of dictionaries
        pyDEW reactant expressed as modal proportions
    system : pyDEW.System object
        pyDEW.System for calculations
    P : float
        pressure in bars
    T : float
        temperature in K

    RETURNS
    -------
    exportDF : pandas.DataFrame
        single row pandas DataFrame of mineral vol%

    """

    systemMinerals = [i.phase_name for i in system.minerals]
    systemSolidSolutions = list(system.solid_solutions.keys())

    names = []
    volPer = []
    for i in range(len(volReactant)):
        mineral = volReactant[i]
        mineralName = mineral['reactant']
        mineralVolProp = mineral['modal']

        # If the mineral is a solid solution mineral, sort each solid solution endmember out
        if mineralName in systemSolidSolutions:

            # Obtain the endmembers of that solid solution
            systemEndmembers = system.solid_solutions[mineralName]
            mineralEndmembers = list(mineral['composition'].keys())
            endmemberRelVol = []

            # Iterate over all endmembers in the mineral
            for j in mineralEndmembers:
                if mineralName == 'CLINOPYROXENE(SS)':
                    names.append(j+'(CPX)')
                elif mineralName == 'KFELDSPAR(SS)':
                    names.append(j+'(KSP)')
                else:
                    names.append(j)
                endmemberProportion = mineral['composition'][j]  # Find volume of each endmember
                endmemberRelVol.append(findMineralVolume(j, system, endmemberProportion, P, T))
            endmemberVolume = [mineralVolProp*k/np.sum(endmemberRelVol) for k in endmemberRelVol]
            volPer.extend(endmemberVolume)

        else:
            # If the mineral is not a solid solution then we can just perform the last calculation
            names.append(mineralName)
            volPer.append(mineralVolProp)

    # Normalise and convert to percentage volumes
    if np.sum(volPer) != 100.0:
        volPer = [100.0*i/sum(volPer) for i in volPer]

    # Alter column header titles
    names = [i + '_volPER' for i in names]
    exportDF = pd.DataFrame([volPer])
    exportDF.columns = names
    return exportDF


def readfO2(tabFile):
    """
    Function to generate list of fO2 during reaction.
    Functionally the same as pyDEW.eq6_output.eq6output(fluid).table_pH['log fo2'].to_list(),
    but I wrote this function initially for debugging purposes.

    PARAMETERS
    ----------
    tabfile : string
        path to pyDEW.Reaction tab file

    RETURNS
    -------
    fO2 : list
        list of reaction fO2

    """

    # Find start of table
    lookup = 'log fo2'
    file = open(tabFile, 'r')
    for num, line in enumerate(file):
        if lookup in line:
            # track where the start of our headers are, data starts 3 lines after the headers
            beginHeaders = num + 1
            beginData = num + 3
            break
    file.close()

    # Read headers from the file, trim away excess whitespace
    headers = linecache.getline(tabFile, beginHeaders).strip().split('  ')
    headers = [i.strip() for i in headers if i != '']

    # Determine number of lines in the data by finding where after the headers we find a big break
    count = 0
    while linecache.getline(tabFile, beginData+count).strip() != '':
        count += 1

    # Read data
    dataLines = []
    for i in range(beginData, beginData+count):
        # Get line, split by whitespace to identify values
        dataLine = linecache.getline(tabFile, i).strip().split(' ')

        # Remove whitespaces, then convert strings to float
        dataLine = [j.strip() for j in dataLine if j != '']

        # This check is to make sure that there aren't two decimals sticking to each other
        check = [j.count('.') for j in dataLine]
        if 2 in check:
            checkIndex = check.index(2)
            errorValue = dataLine[checkIndex]
            value1 = errorValue[:8]
            value2 = errorValue[8:]
            dataLine.insert(checkIndex+1, value1)
            dataLine.insert(checkIndex+2, value2)
            dataLine.pop(checkIndex)
        dataLine = [float(j) for j in dataLine]
        dataLines.append(dataLine)

    # Convert data to dataframe, find ratio of H2O at start and at end
    pHData = pd.DataFrame(data=dataLines, columns=headers)
    fO2 = pHData['log fo2'].to_list()
    return fO2


def condenseMinerals(mineralDF, system, cutoff=0.1, ordered=True, cumulative=True,
                     retainTime=False, condenseCPX=False):
    """
    Function to reduce solid solution endmember volume proportions into solid solution series
    mineral volumes, e.g., FORSTERITE and FAYALITE are reduced to OLIVINE(SS).

    PARAMETERS
    ----------
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
    condenseCPX : boolean
        if there are two clinopyroxene solid solutions, this subroutine checks if the minerals
        can be combined into one. Default is True

    RETURNS
    -------
    outputDF : pandas.DataFrame
        mineral volume output with reduced mineral series
    """

    systemMinerals = [i.phase_name for i in system.minerals]
    systemSolidSolutions = list(system.solid_solutions.keys())
    seriesKeys = mineralDF.columns

    if 'JADEITE(CPX)_volPER' in seriesKeys and 'ENSTATITE_CL_volPER' in seriesKeys:
        pass
    elif condenseCPX == False:
        pass
    else:
        print('Adjusting cpx')
        # If there is calcic cpx then use that solid solution
        if 'JADEITE(CPX)_volPER' in seriesKeys:
            print('Using calcic (di, he, jd)')
            if 'DIOPSIDE_volPER' in seriesKeys:
                mineralDF['DIOPSIDE(CPX)_volPER'] = [i + j for i, j in zip(
                    mineralDF['DIOPSIDE(CPX)_volPER'], mineralDF['DIOPSIDE_volPER'])]
                mineralDF = mineralDF.drop(columns=['DIOPSIDE_volPER'])
            if 'HEDENBERGITE_volPER' in seriesKeys:
                mineralDF['HEDENBERGITE(CPX)_volPER'] = [i + j for i, j in zip(
                    mineralDF['HEDENBERGITE(CPX)_volPER'], mineralDF['HEDENBERGITE_volPER'])]
                mineralDF = mineralDF.drop(columns=['HEDENBERGITE_volPER'])
        
        # If there is sub-calcic cpx then use that solid solution
        elif 'ENSTATITE_CL_volPER' in seriesKeys:
            print('Using subcalcic (di, he, en)')
            if 'DIOPSIDE(CPX)_volPER' in seriesKeys:
                mineralDF['DIOPSIDE_volPER'] = [i + j for i, j in zip(
                    mineralDF['DIOPSIDE(CPX)_volPER'], mineralDF['DIOPSIDE_volPER'])]
                mineralDF = mineralDF.drop(columns=['DIOPSIDE(CPX)_volPER'])
            if 'HEDENBERGITE(CPX)_volPER' in seriesKeys:
                mineralDF['HEDENBERGITE_volPER'] = [i + j for i, j in zip(
                    mineralDF['HEDENBERGITE(CPX)_volPER'], mineralDF['HEDENBERGITE_volPER'])]
                mineralDF = mineralDF.drop(columns=['HEDENBERGITE(CPX)_volPER'])
    
        # Otherwise default to calcic cpx
        else:
            if 'DIOPSIDE(CPX)_volPER' in seriesKeys or 'DIOPSIDE_volPER' in seriesKeys:
                mineralDF['DIOPSIDE(CPX)_volPER'] = [i + j for i, j in zip(
                    mineralDF['DIOPSIDE(CPX)_volPER'], mineralDF['DIOPSIDE_volPER'])]
                mineralDF = mineralDF.drop(columns=['DIOPSIDE_volPER'])
            if 'HEDENBERGITE(CPX)_volPER' in seriesKeys or 'HEDENBERGITE_volPER' in seriesKeys:
                mineralDF['HEDENBERGITE(CPX)_volPER'] = [i + j for i, j in zip(
                    mineralDF['HEDENBERGITE(CPX)_volPER'], mineralDF['HEDENBERGITE_volPER'])]
                mineralDF = mineralDF.drop(columns=['HEDENBERGITE_volPER'])
        seriesKeys = mineralDF.columns

    # PART 1: FIND THE SOLID SOLUTION SERIES
    keyMinerals = []
    for i in mineralDF.columns:
        if '_volPER' in i:
            minString = i.replace('_volPER', '')
        else:
            continue
        if 'DIOPSIDE' in minString or 'HEDENBERGITE' in minString:
            if '(CPX)' in minString:
                keyMineral = 'CLINOPYROXENE(SS)'
            else:
                keyMineral = 'CPX_SUBCALCIC(SS)'
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


def muO2(T):
    """
    Function to calculate the chemical potential of O2 gas, because there is a coding issue with
    ThermoEngine O2. Written by Simon Matthews, reformatted by KW.

    PARAMETERS
    ----------
    T : float
        Temperature, in K

    RETURNS
    -------
    muO2 : float
        Chemical potential of O2 gas, in J/mol
    """
    # Values from the Berman database
    referenceTemperature = 298.15
    c1 = 23.10248
    c2 = 0.002676
    c3 = 1762835.0
    c5 = 804.8876
    c6 = -18172.91960
    S0 = 205.15

    # Enthalpy
    hs1 = c1*(T - referenceTemperature) + 2.0*c5*(np.sqrt(T) - np.sqrt(referenceTemperature))
    hs2 = -1.0*c3*((1.0 / T) - (1.0 / referenceTemperature)) +c6*np.log(T / referenceTemperature)
    hs3 = 0.5*c2*(T**2 - referenceTemperature**2)
    hs = hs1 + hs2 + hs3

    # Entropy
    ss1 = S0 + c1*np.log(T / referenceTemperature)
    ss2 = -2.0*c5*(1.0/np.sqrt(T) - 1.0/np.sqrt(referenceTemperature))
    ss3 = -0.5*c3*(1.0/(T**2) - 1.0/(referenceTemperature**2))
    ss4 = -1.0*c6*(1.0/T - 1.0/referenceTemperature) + c2*(T - referenceTemperature)
    ss = ss1 + ss2 + ss3 + ss4

    muO2 = hs - T*ss
    return muO2


def FMQ(P, T, database):
    """
    Calculates the log10fO2 at FMQ. Written by Simon Matthews, reformatted by KW.

    PARAMETERS
    ----------
    P : float
        Pressure, in bar
    T : float
        Temperature, in K
    database : thermoengine.model.Database object
        ThermoEngine model.database object

    RETURNS
    -------
    log10fO2 : float
        log10fO2 at FMQ
    """

    fa = database.get_phase('Fa')
    mt = database.get_phase('Mag')
    qz = database.get_phase('Qz')

    muFa = fa.gibbs_energy(T, P)
    muMt = mt.gibbs_energy(T, P)
    muQz = qz.gibbs_energy(T, P)
    muO = muO2(T)
    R = 8.314

    # FMQ stoichiometry (Fe3O4 + 1.5 SiO2 = 1.5 Fe2SiO4 + 0.5 O2)
    deltaG = (1.5 * muFa + 0.5 * muO) - (muMt + 1.5 * muQz)

    # deltaG = -RT ln(K), rearrange for K (eq. constant)
    lnK = -1.0 * deltaG / (R * T)

    # All mineral phases are pure, so K = sqrt(fO2)
    lnfO2 = 2 * lnK

    # Change from ln to log10
    log10fO2 = lnfO2 / np.log(10)
    return log10fO2


def getD0AtomicMass(d0file):
    """
    Extract atomic mass, oxide, and gravimetric factors from a data0 file.

    PARAMETERS
    ----------
    d0file : string
        path to data0 file

    RETURNS
    -------
    df : pandas.DataFrame
        DataFrame with atomic mass, oxide, and gravimetric factors
    """

    lookup1 = 'ELEMENTS, ATOMIC WEIGHTS, OXIDES, AND GRAVIMETRIC FACTORS'
    lookup2 = 'temperatures'
    file = open(d0file, 'r')
    for num, line in enumerate(file):
        if lookup1 in line:
            startNumber = num
        elif lookup2 in line:
            endNumber = num
            break
    file.close()
    beginHeaders = startNumber+1
    headers = linecache.getline(d0file, beginHeaders).strip().split(', ')
    headers[3] = headers[3].replace('AND ','')

    start = startNumber+2
    end = endNumber+1
    dataLines = []
    for j in range(start, end):
        dataLine = linecache.getline(d0file, j).strip().split(' ')
        dataLine = [k.strip() for k in dataLine if k != '']
        if dataLine == []:
            continue
        else:
            dataLines.append(dataLine)
    df = pd.DataFrame(dataLines)
    df.columns = headers
    df.set_index('ELEMENTS', inplace=True)
    df['ATOMIC WEIGHTS'] = df['ATOMIC WEIGHTS'].astype(float)
    df['GRAVIMETRIC FACTORS'] = df['GRAVIMETRIC FACTORS'].astype(float)
    return df


def listOfListSearch(searchList, searchString):
    """
    Return the index of a list in a list of lists if the list has a particular string within it.

    PARAMETERS
    ----------
    searchList : list of lists
        list of lists to query
    searchString : string
        string to query list of lists

    RETURNS
    -------
    integer index of list within list of lists containing string
    """
    for x in searchList:
        for y in x:
            if searchString in y:
                return searchList.index(x)


def getD0MineralElements(d0file):
    """
    Get information on the elements comprising a mineral from the data0 file.

    PARAMETERS
    ----------
    d0file : string
        path to data0 file

    RETURNS
    -------
    df : pandas.DataFrame
        DataFrame of minerals (index) with elemental compositions (table contents, by number of
        moles in that mineral endmember)
    """

    lookup1 = 'endit'

    dataLines = []
    minerals = []
    totalStarts = []

    # Open file, extract info, read file
    file = open(d0file, 'r')
    for num, line in enumerate(file):
        lookup1Check = False
        if lookup1 in line:
            lookup1Check = True
            totalStarts.append(num)
        if lookup1Check == False:
            pass
    file.close()
    start = totalStarts[0]+3
    end = totalStarts[1]+1
    dataLines = []
    for j in range(start, end):
        dataLine = linecache.getline(d0file, j).strip().split(' ')
        dataLine = [k.strip() for k in dataLine if k != '']
        if dataLine == []:
            continue
        else:
            dataLines.append(dataLine)
    divider = ['+----------------------------------------------------------------------------']
    separators = [i for i, x in enumerate(dataLines) if x == divider]
    series = []
    for i in range(len(separators)):
        if i == 0:
            upperSeparator = 0
            lowerSeparator = separators[i]
        elif i == len(separators):
            pass
        else:
            upperSeparator = separators[i-1]+1
            lowerSeparator = separators[i]
        subList = dataLines[upperSeparator:lowerSeparator]
        minerals.append(subList[0][0])
        elements1 = listOfListSearch(subList, 'CHEMICAL')
        elements2 = listOfListSearch(subList, 'SPECIES')
        elements = sum(subList[elements1+1:elements2], [])
        elementsDict = dict(zip(elements[1::2], elements[::2]))
        series.append(pd.Series(elementsDict))
    df = pd.DataFrame(series)
    df.index = minerals
    df = df.fillna(0)
    df = df.astype(float)
    return df


def countReactantElements(reactant, d0File='./working/data0'):
    """
    Count moles of elements forming a reactant.

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    d0file : string
        path to data0 file

    RETURNS
    -------
    molesDict : dictionary
        dictionary of moles of elements
    """

    referenceTable = getD0MineralElements(d0File)
    # Get dictionary of moles for each endmember
    minerals = []
    moles = []
    for i in reactant:
        if 'composition' in i.keys():
            SSmoles = i['moles']
            endmembers = list(i['composition'].keys())
            endmemberProportions = list(i['composition'].values())
            for j in range(len(i['composition'].keys())):
                minerals.append(endmembers[j])
                moles.append(endmemberProportions[j] * SSmoles)
        else:
            minerals.append(i['reactant'])
            moles.append(i['moles'])

    returnSeries = []
    # Now determine the elements
    for i in range(len(minerals)):
        queryMineral = minerals[i]
        refSeries = referenceTable.loc[queryMineral]
        returnSeries.append(refSeries * moles[i])
    df = pd.DataFrame(returnSeries)
    # Remove all zero rows
    df = df.loc[:, (df != 0).any(axis=0)]
    # Determine list of element moles
    elements = df.columns.tolist()
    elementMoles = []
    for i in elements:
        elementMoles.append(np.sum(df[i]))
    molesDict = {elements[i]: elementMoles[i] for i in range(len(elements))}
    return molesDict


def countFluidElements(fluid, H2OMass, inputType='Fluid'):
    """
    Return a dictionary with the moles of elements in the fluid phase. This uses the 'molality'
    table of a pyDEW.Fluid object.

    PARAMETERS
    ----------
    fluid : pyDEW.Fluid object or pandas.Series
        pyDEW object to determine elements from
    H2OMass : float
        mass of solvent (H2O) in the system in g
    inputType : string
        'Fluid' or 'Series' to determine fluid input. Default is 'Fluid'

    RETURNS
    -------
    molesDict : dictionary
        dictionary of moles of elements in fluid
    """

    if inputType == 'Fluid':
        molalities = fluid.elemental_comp['molality']
        elements = molalities.index.tolist()
    elif inputType == 'Series':
        molalities = [10**i for i in fluid]
        elements = fluid.index.tolist()
    moles = [i * H2OMass/1000 for i in molalities]
    molesDict = {elements[i]: moles[i] for i in range(len(elements))}
    molesDict = {k.upper() : v for k,v in molesDict.items()}
    return molesDict


def totalElements(a, b):
    """
    Quick function to add two dictionaries together element-wise.

    PARAMETERS
    ----------
    a, b: dictionaries
        dictionaries that may contain some common elements

    RETURNS
    -------
    dictionary
        dictionary of elements in a and b summed element-wise
    """
    return {i: a.get(i, 0) + b.get(i, 0) for i in set(a).union(b)}


def findFluidMass(fluid, H2Omass, inputType='Fluid', d0file='working/data0', Omolal=None):
    """
    Function to find mass of the fluid from a pyDEW.Fluid object (NOT SOLVENT MASS).

    PARAMETERS
    ----------
    fluid : pyDEW.Fluid object
        pyDEW.Fluid object
    H2OMass : float
        mass of solvent (H2O) in the system in g
    inputType : string
        'Fluid' or 'Series' to determine fluid input. Default is 'Fluid'
    d0file : string
        path to data0 file
    Omolal : float
        O molality if inputType is 'Series'

    RETURNS
    -------
    fluidMass : float
        mass of fluid (H2O + species) in the system
    """

    fluidElements = countFluidElements(fluid, H2Omass, inputType=inputType)
    d0atomicMasses = getD0AtomicMass(d0file)
    fluidMass = 0

    # mass = moles of elements * atomic masses of elements
    for i in fluidElements:
        fluidMass += fluidElements[i] * d0atomicMasses.loc[i]['ATOMIC WEIGHTS']
    if inputType == 'Series':
        fluidMass += Omolal * H2Omass/1000 * d0atomicMasses.loc['O']['ATOMIC WEIGHTS']
    return fluidMass


def findFluidOMolal(outputFile='working/output', logZiValue=None):
    """
    For some godforsaken reason the pyDEW.Reaction object does not include the molality of fluid
    oxygen. This finds that.

    PARAMETERS
    ----------
    outputFile : string
        path to output file generated by EQ6. Default is 'working/output'
    logZiValue : int
        select a specific value from the output file based on logZi? Default is None

    RETURNS
    -------
    df : pd.DataFrame
        DataFrame with oxygen molality at every reaction progress
    or float
        value of oxygen molality at a given log reaction progress
    """

    lookup1 = 'log of reaction progress'
    lookup2 = 'element totals for the aqueous phase'
    logZi = []
    OLineNumber = []
    file = open(outputFile, 'r')
    for num, line in enumerate(file):

        # Lookup 1 is where we find our running reaction progress
        if lookup1 in line:
            logZi.append(float(line.split()[-1]))

        # Lookup 2 is start of the elements
        if lookup2 in line:
            OLineNumber.append(num + 5)
    file.close()
    checkLines = [linecache.getline(outputFile, i).strip().split('  ') for i in OLineNumber]
    molalities = [i[9] for i in checkLines]
    outputDict = {'logZi' : logZi,
                  'O molal' : molalities}
    df = pd.DataFrame(outputDict)
    if logZiValue != None:
        dfSeries = df[df['logZi'] == logZiValue]
        return float(dfSeries['O molal'].iloc[0])
    else:
        df['O molal'] = df['O molal'].astype(float)
        return df
