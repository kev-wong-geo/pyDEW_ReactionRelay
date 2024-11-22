# Functions to subdivide relay function.
# Last updated to include things for fluid and solvent mass: 22/10/2024, KW

import numpy as np
import pyDEW
from thermoengine import model
import pandas as pd
from copy import deepcopy
import linecache
from scipy.interpolate import LinearNDInterpolator
from timeit import default_timer as timer
import os
import shutil

# Additional pyDEW functions are wrapped up in this file
import pyDEW_functions as pDF


def formatFluid(molalities):
    """
    Format to reformat pyDEW.Reaction and pyDEW.Fluid outputs as new fluid inputs.

    PARAMETERS
    ----------
    molalities : dictionary
        pyDEW.Reaction or pyDEW.Fluid output dictionary

    RETURNS
    -------
    formattedMolalities : dictionary
        pyDEW.Fluid input dictionary

    """

    # pyDEW.Reaction and pyDEW.Fluid return differently cased outputs, this corrects for that
    formattedMolalities = {k.upper() : v for k,v in molalities.items()}
    corrDict = {
        'NA':  'NA+',
        'CA': 'CA+2',
        'MG': 'MG+2',
        'AL': 'AL+3',
        'SI': 'H4SIO4(AQ)',
        'C' : 'CO3-2',
        'CL': 'CL-',
        'FE': 'FE+2'}

    # Some molalities can drop too low - this checks for a change in molalities
    commonKeys = set(corrDict).intersection(set(formattedMolalities))
    outputMolalities = {corrDict[k] : formattedMolalities[k] for k in commonKeys}
    return outputMolalities


def createHybridFluid(proportions, P, T, database='EQ3_pickups_GS.xlsx',
                      sheetNames=['DMM','MORB','Sediments']):
    """
    Function to create a hybrid linearly interpolated fluid from fluid endmember compositions.
    Fluid assumed to be linearly proportional of endmembers.

    PARAMETERS
    ----------
    proportions : list of floats
        proportions each fluid endmember contributes to the final fluid
    T : float
        temperature for interpolation, in K
    P : float
        pressure for interpolation, in bars
    database : string
        xlsx database with fluid endmember compositions. Default is 'EQ3_pickups_GS.xlsx'
    sheetNames : list of strings
        names of sheets in xlsx database corresponding to fluid endmembers. Default is
        ['DMM','MORB','Sediments'], corresponding to 'EQ3_pickups_GS.xlsx'

    RETURNS
    -------
    exportDict : dictionary
        dictionary describing elemental composition of hybridised fluid
    exportfO2 : float
        float describing hybridised oxygen fugacity
    """

    assert len(proportions) == len(sheetNames), 'check same item proportions and database'

    for i in range(len(proportions)):

        # To store our new fluid
        if i == 0:
            molalitiesDict = {}

        key = sheetNames[i]
        LithData = pd.read_excel(database, sheet_name=sheetNames[i])
        proportion = proportions[i]/np.sum(proportions)

        # Find the species comprising the fluid
        columnHeaders = list(LithData.columns)
        speciesHeaders = columnHeaders[2:]

        # Extract pressures and temperatures
        dataPressures = LithData['P (kbar)'].values
        dataTemperatures = LithData['T (degC)'].values

        # This iterates over all species and linearly interpolates across them based on the
        # molalities pickup sheet
        for j in range(len(speciesHeaders)):
            speciesData = LithData[speciesHeaders[j]]

            # Makes sure that the species aren't all constant, which makes the interpolation wacky
            if speciesData.tolist() == [speciesData.tolist()[0]]*len(speciesData):
                if i == 0:
                    molalitiesDict[speciesHeaders[j]] = speciesData[0] * proportion
                else:
                    molalitiesDict[speciesHeaders[j]] = molalitiesDict[speciesHeaders[j]] + (
                        speciesData[0] * proportion)

            # Otherwise we set up the interpolator for the species given pressure and temperature
            else:
                interp = LinearNDInterpolator(
                    list(zip(dataTemperatures.tolist(), dataPressures.tolist())),
                    speciesData.tolist())

                # Reconvert back to original units here
                interpMolal = interp(T-273.15, P/1000)
                if i == 0:
                    molalitiesDict[speciesHeaders[j]] = float(interpMolal) * proportion
                elif speciesHeaders[j] not in molalitiesDict.keys():
                    molalitiesDict[speciesHeaders[j]] = float(interpMolal) * proportion
                else:
                    molalitiesDict[speciesHeaders[j]] = molalitiesDict[speciesHeaders[j]] + (
                        float(interpMolal) * proportion)
    exportDict = formatFluid(molalitiesDict)
    exportfO2 = molalitiesDict['fO2']
    return exportDict, exportfO2


def failedReactionOutput(reactant, fluid, solventMass, system, P, T, errorTag=0,
                         d0file='working/data0'):
    """
    Function to return a compatible output if the reaction has returned an error.

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    fluid : pyDEW.Fluid object
        pyDEW.Fluid object
    solventMass : float
        mass of solvent in grams
    system : pyDEW.System object
        pyDEW.System for calculations
    P : float
        pressure in bars
    T : float
        temperature in K
    errorTag : int or None
        tag for error messages. 0: reaction has failed, 1: fluid-rock ratio too low
    d0file : string
        path to data0 file

    RETURNS
    -------
    outputDict : dictionary of dictionaries
        dictionary including information on the following:
            - 'Properties' : physical properties of the reactants/products
            - 'Solid' : solid product formatted as pyDEW.Reaction input dictionaries
            - 'Elements' : elemental molalities of the fluid (log molal). Note keys are FLUID.
            - 'Species' : molalities of species in the fluid (log molal)
            - 'Volumes' : volume percentages of minerals in the solid
    """

    initialReactantMass = pDF.findReactantMass(reactant, system)
    initialFluidMass = pDF.findFluidMass(fluid, solventMass, inputType='Fluid',
                                         d0file='working/data0')
    initialFRR = solventMass / initialReactantMass
    if errorTag == 0:
        errorMessage = 'pyDEW.Reaction has failed for undetermined reason'
    elif errorTag == 1:
        errorMessage = 'Fluid-rock ratio is below specified cut-off'
    elif errorTag == 2:
        errorMessage = 'EQ3 failure: fluid failed to equilibrate'
    elif errorTag is None:
        errorMessage = ''
    else:
        errorMessage = 'unspecified error'
    physicalProperties = {
        'Temperature (K)' : T,
        'Pressure (bar)' : P,
        'initial solid mass' : initialReactantMass,
        'initial solid volume' : pDF.findReactantVolume(reactant, system, P, T),
        'initial solid density' : pDF.findReactantDensity(reactant, system, P, T),
        'final solid mass' : initialReactantMass,
        'final solid volume' : pDF.findReactantVolume(reactant, system, P, T),
        'final solid density' : pDF.findReactantDensity(reactant, system, P, T),
        'initial solvent mass' : solventMass,
        'final solvent mass' : solventMass,
        'initial fluid mass' : initialFluidMass,
        'final fluid mass' : initialFluidMass,
        'initial FRR' : initialFRR,
        'final FRR' : initialFRR,
        'initial fluid fO2' : fluid.fO2,
        'final fluid fO2' : fluid.fO2,
        'initial fluid pH' : fluid.pH,
        'final fluid pH' : fluid.pH,
        'max log zi (pyDEW elements)' : -999.0,
        'max log zi (pyDEW species and DEW output)' : -999.0,
        'error' : errorMessage
        }

    # Format the fluid elements and species - these are not log so we log them
    fluidElements = fluid.elemental_comp['molality'].to_dict()
    fluidElements = {k.title() : np.log10(v) for  k, v in fluidElements.items()}  # to match reaction formatting
    fluidSpecies = pd.Series(fluid.aqueous_species['molality'].values,
                             index=fluid.aqueous_species['species']).to_dict()
    fluidSpecies.update((i, np.log10(j)) for i, j in fluidSpecies.items())

    outputDict = {}
    outputDict['Properties'] = physicalProperties
    outputDict['Solid'] = reactant
    outputDict['Elements'] = fluidElements
    outputDict['Species'] = fluidSpecies
    outputDict['Volumes'] = pDF.reactantToVolDF(reactant, system, P, T).iloc[0].to_dict()
    outputDict['Produced'] = {}
    return outputDict


def reactionByMass(reactant, fluid, solventMass, system, P, T, outputFile='working/output',
                   tabFile='working/tab', d0file='working/data0', **kwargs):
    """
    Function to perform a pyDEW calculation by mass (instead of relative to 1 kg of fluid).

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    fluid : pyDEW.Fluid object
        pyDEW.Fluid object - this is kept separate for debugging purposes
    solventMass : float
        mass of solvent in grams
    system : pyDEW.System object
        pyDEW.System for calculations
    P : float
        pressure in bars
    T : float
        temperature in K
    outputFile : string
        path to pyDEW.Reaction output file
    tabFile : string
        path to pyDEW.Reaction tab file
    d0file : string
        path to data0 file
    **kwargs : optional keyword arguments
        keyword arguments for pyDEW.Reaction

    RETURNS
    -------
    outputDict : dictionary of dictionaries
        dictionary including information on the following:
            - 'Properties' : physical properties of the reactants/products
            - 'Solid' : solid product formatted as pyDEW.Reaction input dictionaries
            - 'Elements' : elemental molalities of the fluid (log molal)
            - 'Species' : molalities of species in the fluid (log molal)
            - 'Volumes' : volume percentages of minerals in the solid

    """

    #### PREAMBLE ####
    # Obtain mass of reactant and FRR, recalculate reactant relative to 1 kg of reactant fluid
    initialReactantMass = pDF.findReactantMass(reactant, system)
    initialFluidMass = pDF.findFluidMass(fluid, solventMass, inputType='Fluid', d0file=d0file)
    initialFRR = solventMass / initialReactantMass
    workingReactant1kg = pDF.reactantToFR(reactant, initialFRR, system)
    initialReactant1kgMoles = np.sum([j['moles'] for j in workingReactant1kg])
    dewReaction = pyDEW.Reaction(fluid, T, P, workingReactant1kg, zimax=initialReactant1kgMoles,
                                 calculation_mode='titration')

    # correction factor for mass scaling - note that this is often not 1 kg
    solidModulate  = solventMass / pDF.findSolventMass(tabFile)[0]

    # Sometimes an empty dataframe is returned - this flags it up
    if isinstance(dewReaction.minerals, pd.DataFrame) == False:
        raise pyDEW.core.RunError('Mineral output is not a pd.DataFrame')

    lastReactionDF = pDF.findSolidComposition(outputFile, system, P, T)

    #### PROCESSING ####
    # Get maximum valid log zi and corresponding index
    # Species (output) have different log zi to the elements (tab)
    lastValidElementsLogZi = dewReaction.elements.last_valid_index()
    lastValidElementsLogZiIndex = dewReaction.elements.index.get_loc(lastValidElementsLogZi)
    if not isinstance(lastValidElementsLogZiIndex, int):
        if isinstance(lastValidElementsLogZiIndex, slice):
            # Use start instead of stop because there can be a string of nans at the end of the
            # DataFrame
            lastValidElementsLogZiIndex = lastValidElementsLogZiIndex.start - 1
        #lastValidElementsLogZiIndex = np.max(lastValidElementsLogZiIndex)

    # Species log zi and output log zi are the same, but I keep them distinct here for clarity
    speciesLogZiList = list(dewReaction.species_concs.index)
    lastValidSpeciesLogZi = min(speciesLogZiList, key=lambda x:abs(x-lastValidElementsLogZi))
    lastValidSpeciesLogZiIndex = speciesLogZiList.index(lastValidSpeciesLogZi)
    lastValidOutputLogZi = lastValidSpeciesLogZi
    lastValidOutputLogZiIndex = lastValidSpeciesLogZiIndex

    # Get dataframe of last solid (from 1 kg solvent), account for repeated solid solution
    # endmembers
    lastSolid1kg = lastReactionDF[lastReactionDF['log zi'] == lastValidOutputLogZi]
    lastSolid1kgHeaders = lastSolid1kg['phase/end-member'].tolist()
    lastSolid1kgValues = lastSolid1kg['moles'].tolist()
    if 'CLINOPYROXENE(SS)' in lastSolid1kgHeaders:
        index = lastSolid1kgHeaders.index('CLINOPYROXENE(SS)')
        lastSolid1kgHeaders[index+1] = 'DIOPSIDE(CPX)'
        lastSolid1kgHeaders[index+2] = 'JADEITE(CPX)'
        lastSolid1kgHeaders[index+3] = 'HEDENBERGITE(CPX)'
    if 'KFELDSPAR(SS)' in lastSolid1kgHeaders:
        index = lastSolid1kgHeaders.index('KFELDSPAR(SS)')
        lastSolid1kgHeaders[index+1] = 'KFELDSPAR(KSP)'
        lastSolid1kgHeaders[index+2] = 'ALBITE(KSP)'

    #### FORMAT RELATIVE TO 1 KG INITIAL FLUID ####
    # New dataframe for our final reactant (from 1 kg fluid)
    lastSolid1kgDF = pd.DataFrame([lastSolid1kgValues])
    lastSolid1kgDF.columns = lastSolid1kgHeaders

    # Set the headers up so we find mineral endmembers but not solid solution series
    acceptableHeaders = [i.props['phase_name'] for i in system.minerals]
    acceptableHeaders.extend(['DIOPSIDE(CPX)', 'JADEITE(CPX)', 'HEDENBERGITE(CPX)',
                              'KFELDSPAR(KSP)', 'ALBITE(KSP)'])
    lastSolid1kgDF = lastSolid1kgDF[lastSolid1kgDF.columns.intersection(acceptableHeaders)]

    # Remove any duplicate columns
    if len(lastSolid1kgDF.columns) != len(set(lastSolid1kgDF.columns)):
        lastSolid1kgDF = lastSolid1kgDF.iloc[:, ~lastSolid1kgDF.columns.duplicated()]

    # Convert pandas output into reactant list of dictionaries, find fluid mass change from 1 kg
    lastSolid1kgProduct = pDF.DFtoReactant(lastSolid1kgDF.iloc[0], system)

    workingReactant1kgMass = pDF.findReactantMass(workingReactant1kg, system)
    lastSolid1kgMass = pDF.findReactantMass(lastSolid1kgProduct, system)
    lastSolvent1kgMass = pDF.findSolventMass(tabFile)[1]
    lastFluid1kgOMolal = pDF.findFluidOMolal(logZiValue=lastValidSpeciesLogZi)
    lastFluid1kgMass = pDF.findFluidMass(dewReaction.elements.iloc[lastValidElementsLogZiIndex],
                                         lastSolvent1kgMass, inputType='Series',
                                         d0file=d0file, Omolal=lastFluid1kgOMolal)

    #### FORMAT RELATIVE TO INITIAL FLUID MASS (ADJUSTED) ####
    # Multiply product solid values by solidModulate to get the true solid mass
    # (i.e., for whatever amount of fluid there is before correction to 1 kg for F/R)
    adjustedLastSolidProduct = []
    for j in lastSolid1kgProduct:
        j['moles'] = j['moles'] * solidModulate
        adjustedLastSolidProduct.append(j)
    adjustedLastSolidDF = pDF.reactantToDF(adjustedLastSolidProduct, system)
    adjustedLastSolidDF.reset_index().T.drop_duplicates().T
    adjustedLastSolventMass = lastSolvent1kgMass * solidModulate
    adjustedLastFluidMass = lastFluid1kgMass * solidModulate
    adjustedLastSolidMass = pDF.findReactantMass(adjustedLastSolidProduct, system)
    adjustedLastSolidVolume = pDF.findReactantVolume(adjustedLastSolidProduct, system, P, T)
    adjustedLastSolidDensity = pDF.findReactantDensity(adjustedLastSolidProduct, system, P, T)
    finalFRR = adjustedLastFluidMass / adjustedLastSolidMass

    # pyDEW.Reaction.fO2 has a memory issue - this is the correct method of obtaining fO2
    eq6Output = pyDEW.eq6_output.eq6output(fluid, tab_filepath='working/tab',
                                           output_filepath='working/output')
    finalfO2 = eq6Output.table_pH['log fo2'].iloc[lastValidElementsLogZiIndex]
    initialpH = eq6Output.table_pH['ph'].iloc[0]
    finalpH = eq6Output.table_pH['ph'].iloc[lastValidElementsLogZiIndex]

    # Dictionary to store the physical properties of the system
    physicalProperties = {
        'Temperature (K)' : T,
        'Pressure (bar)' : P,
        'initial solid mass' : initialReactantMass,
        'initial solid volume' : pDF.findReactantVolume(reactant, system, P, T),
        'initial solid density' : pDF.findReactantDensity(reactant, system, P, T),
        'final solid mass' : adjustedLastSolidMass,
        'final solid volume' : adjustedLastSolidVolume,
        'final solid density' : adjustedLastSolidDensity,
        'initial solvent mass' : solventMass,
        'final solvent mass' : adjustedLastSolventMass,
        'initial fluid mass' : initialFluidMass,
        'final fluid mass' : adjustedLastFluidMass,
        'initial FRR' : initialFRR,
        'final FRR' : finalFRR,
        'initial fluid fO2' : fluid.fO2,
        'final fluid fO2' : finalfO2,
        'initial fluid pH' : initialpH,
        'final fluid pH' : finalpH,
        'max log zi (pyDEW elements)' : lastValidElementsLogZi,
        'max log zi (pyDEW species and DEW output)' : lastValidSpeciesLogZi,
        'error' : ''
        }

    # Return dictionary of dictionaries
    outputDict = {}
    outputDict['Properties'] = physicalProperties
    outputDict['Solid'] = adjustedLastSolidProduct
    outputDict['Elements'] = dewReaction.elements.iloc[lastValidElementsLogZiIndex].to_dict()
    outputDict['Species'] = dewReaction.species_concs.loc[lastValidSpeciesLogZi].to_dict()
    outputDict['Volumes'] = pDF.reactantToVolDF(adjustedLastSolidProduct,
                                                system, P, T).iloc[0].to_dict()
    
    # Really annoyingly the elements and their solid solutions do not match up either to the logZi indices
    # of the elements nor to each other. We need to check for clashes again. If those clashes exist, then we
    # opt for the second option.
    producedElements = dewReaction.minerals.loc[lastValidElementsLogZi]
    producedSolidSolutions = dewReaction.solid_solutions.loc[lastValidElementsLogZi]
    if isinstance(producedElements, pd.DataFrame):
        producedElements = producedElements.iloc[-1]
    if isinstance(producedSolidSolutions, pd.DataFrame):
        producedSolidSolutions = producedSolidSolutions.iloc[-1]
    
    producedElementsCorrected = {k : (10**v) * solidModulate for k,v in producedElements.items()}
    producedExport = {}
    for i in producedElementsCorrected:
        if '(SS)' in i:
            systemEndmembers = system.solid_solutions[i]
            for j in systemEndmembers:
                endmemberProportion = producedSolidSolutions[j]
                endmemberMoles = endmemberProportion * producedElementsCorrected[i]
                producedExport[j] = endmemberMoles
        else:
            producedExport[i] = producedElementsCorrected[i]
    outputDict['Produced'] = producedExport
    return outputDict


def combineOutputs(outputs, system):
    """
    Function to combine output dictionaries from previous function into pandas DataFrames

    PARAMETERS
    ----------
    outputs : list of dictionary of dictionaries
        list of outputs generated by the 'reactionByMass' function
    system : pyDEW.System object
        pyDEW.System for calculations

    RETURNS
    -------
    list of four pandas DataFrames for each output

    """

    physicalList = []
    solidList = []
    elementsList = []
    speciesList = []
    volumesList = []
    producedList = []
    for i in range(len(outputs)):
        physicalList.append(pd.Series(outputs[i]['Properties']))
        solidList.append(pDF.reactantToDF(outputs[i]['Solid'], system).iloc[0])
        elementsList.append(pd.Series(outputs[i]['Elements']))
        speciesList.append(pd.Series(outputs[i]['Species']))
        volumesList.append(pd.Series(outputs[i]['Volumes']))
        producedList.append(pd.Series(outputs[i]['Produced']))
    physicalDF = pd.concat(physicalList, axis=1).T
    solidDF = pd.concat(solidList, axis=1).T
    elementsDF = pd.concat(elementsList, axis=1).T
    speciesDF = pd.concat(speciesList, axis=1).T
    volumesDF = pd.concat(volumesList, axis=1).T
    producedDF = pd.concat(producedList, axis=1).T
    return [physicalDF, solidDF, elementsDF, speciesDF, volumesDF, producedDF]


def combineTimes(combinedOutputs, startingTime=0, exportPath='export'):
    """
    Function to combine the combined outputs generated in the previous function.

    PARAMETERS
    ----------
    combinedOutputs : list of pandas.DataFrames
        list of pandas.DataFrames comprised of combined pyDEW reaction outputs
    exportPath : string
        path to export Excel file. Default is 'export'

    RETURNS
    -------
    Excel file comprising six sheets
    """

    # Add a time column
    for i in range(len(combinedOutputs)):
        time = startingTime + i + 1
        timeColumn = [time] * len(combinedOutputs[i][0])
        for j in combinedOutputs[i]:
            j['Time'] = timeColumn

    physicalOutputs = [i[0] for i in combinedOutputs]
    solidOutputs = [i[1] for i in combinedOutputs]
    elementsOutputs = [i[2] for i in combinedOutputs]
    speciesOutputs = [i[3] for i in combinedOutputs]
    volumesOutputs = [i[4] for i in combinedOutputs]
    producedOutputs = [i[5] for i in combinedOutputs]

    physicalExportDF = pd.concat(physicalOutputs)
    solidExportDF = pd.concat(solidOutputs)
    elementsExportDF = pd.concat(elementsOutputs)
    speciesExportDF = pd.concat(speciesOutputs)
    volumesExportDF = pd.concat(volumesOutputs)
    producedExportDF = pd.concat(producedOutputs)

    # Fiddling to move the time column to the end if new minerals pop in
    columns = [col for col in solidExportDF.columns if col != 'Time'] + ['Time']
    solidExportDF = solidExportDF[columns]
    columns = [col for col in volumesExportDF.columns if col != 'Time'] + ['Time']
    volumesExportDF = volumesExportDF[columns]

    with pd.ExcelWriter(exportPath+'.xlsx') as writer:
        physicalExportDF.to_excel(writer, sheet_name='properties', index=False)
        solidExportDF.to_excel(writer, sheet_name='solid_minerals', index=False)
        elementsExportDF.to_excel(writer, sheet_name='fluid_elements', index=False)
        speciesExportDF.to_excel(writer, sheet_name='fluid_species', index=False)
        volumesExportDF.to_excel(writer, sheet_name='mineral_volPER', index=False)
        producedExportDF.to_excel(writer, sheet_name='produced_minerals', index=False)
    return


def combineExcelOutputs(files, exportPath):
    """
    Function to combine Excel output files.

    PARAMETERS
    ----------
    files : list of strings
        list of paths to xlsx files generated by the decompressionRelay function
    exportPath : string
        path to export combined file

    RETURNS
    -------
    single xlsx file of combined data

    """

    for i in range(len(files)):
        if i == 0:
            excelFile0 = pd.read_excel(files[i], sheet_name=None)
            sheetNames = excelFile0.keys()
            homeDict = deepcopy(excelFile0)
        else:
            excelFilei = pd.read_excel(files[i], sheet_name=None)
            for j in sheetNames:
                homeDict[j] = pd.concat([homeDict[j], excelFilei[j]])
    with pd.ExcelWriter(exportPath+'.xlsx') as writer:
        for k, v in homeDict.items():
            v.reset_index(inplace=True, drop=True)
            if 'Time' in v.columns:
                columns = [col for col in v.columns if col != 'Time'] + ['Time']
                v = v[columns]
            v.to_excel(writer, sheet_name=k, index=False)

    return


def createUniformInput(reactant, system, reactantMass, pressures, temperatures, startingTime=0):
    """
    Create an input pandas DataFrame compatible for use with the relay function.

    PARAMETERS
    ----------
    reactant : list of dictionaries
        pyDEW.Reaction mineral input
    system : pyDEW.System object
        pyDEW.System for calculations
    reactantMass : float
        mass of the reactant in g
    pressures : list of floats
        list of pressures in bars
    temperatures : list of floats
        list of temperatures in K
    startingTime : int
        integer representing time at the start of the reaction. Default is 0

    RETURNS
    -------
    inputDF : pandas.DataFrame
        pandas DataFrame formatted for relay function
    """

    assert len(pressures) == len(temperatures), 'pressures and temperatures are different lengths'
    reactant = pDF.reactantToMass(reactant, reactantMass, system)
    reactantDF = pDF.reactantToDF(reactant, system)
    inputDF = pd.concat([reactantDF] * len(pressures))
    inputDF['Pressure (bar)'] = pressures
    inputDF['Temperature (K)'] = temperatures
    inputDF['Time'] = [startingTime]*len(pressures)
    inputDF.reset_index(inplace=True, drop=True)
    return inputDF


def createHeterogeneousInput(reactants, system, reactantMasses, intervals, pressures, temperatures,
                             startingTime=0):
    """
    Create an input pandas DataFrame compatible for use with the relay function, with a
    changing solid defined by ratios of the number of steps

    PARAMETERS
    ----------
    reactants : list of lists of dictionaries
        list of pyDEW.Reaction mineral input
    system : pyDEW.System object
        pyDEW.System for calculations
    reactantMasses : list of floats
        masses of each reactant in g
    intervals : list of floats
        intervals to divide reactants by
    pressures : list of floats
        list of pressures in bars
    temperatures : list of floats
        list of temperatures in K
    startingTime : int
        integer representing time at the start of the reaction. Default is 0

    RETURNS
    -------
    inputDF : pandas.DataFrame
        pandas DataFrame formatted for relay function
    """

    assert len(pressures) == len(temperatures), 'pressures and temperatures are different lengths'
    assert len(reactants) == len(reactantMasses), 'reactants and masses are different lengths'
    assert len(reactants) == len(intervals), 'reactants and intervals are different lengths'
    # Get the interval equivalents for PT
    splitFractions = np.cumsum([i * len(pressures) / np.sum(intervals) for i in intervals])
    splitFractions = [round(i) for i in splitFractions]
    inputSubDF = []
    for i in range(len(splitFractions)):
        if i == 0:
            rows = splitFractions[0]
        else:
            rows = splitFractions[i] - splitFractions[i-1]
        reactantMass = pDF.reactantToMass(reactants[i], reactantMasses[i], system)
        reactantDF = pDF.reactantToDF(reactantMass, system)
        inputSubDF.append(pd.concat([reactantDF] * rows))
    inputDF = pd.concat(inputSubDF)
    inputDF['Pressure (bar)'] = pressures
    inputDF['Temperature (K)'] = temperatures
    inputDF['Time'] = [startingTime]*len(pressures)
    inputDF.reset_index(inplace=True, drop=True)
    return inputDF


def relayInputFromOutput(outputFile, time):
    """
    Create an input from a relay output file.

    PARAMETERS
    ----------
    outputFile : string
        path to Excel outputFile generated from relay function
    time : int
        timestep from which to extract data

    RETURNS
    -------
    inputDF : pandas.DataFrame
        pandas DataFrame formatted for relay function

    """

    inputSolids = pd.read_excel(outputFile, sheet_name='solid_minerals')
    inputProperties = pd.read_excel(outputFile, sheet_name='properties')
    inputReactant = inputSolids[inputSolids['Time']==time]
    inputPropertiesTime = inputProperties[inputProperties['Time']==time]
    inputPressure = inputPropertiesTime['Pressure (bar)'].to_list()
    inputTemperature = inputPropertiesTime['Temperature (K)'].to_list()
    inputReactant['Pressure (bar)'] = inputPressure
    inputReactant['Temperature (K)'] = inputTemperature
    inputReactant.reset_index(drop=True, inplace=True)
    return inputReactant


def decompressionRelay(startingColumn, startingFluid, startingfO2, system, timesteps=10,
                       startingTime=0, solventMass=None, startingFRR=1.0, terminalFRR=0.1,
                       molalityLimit=-18.0, exportPath='data/export',
                       outputToRead='working/output', tabToRead='working/tab',
                       startFromFile=None, combinedOutput=True,
                       debugLabels=False, **kwargs):
    """
    Decompression relay as a function of time. This models a column of rock getting flushed with a
    fluid periodically from the base of the rock column.

    PARAMETERS
    ----------
    startingColumn : pandas.DataFrame
        pandas DataFrame with each column a different mineral composition (moles), with two
        additional columns 'Pressure (bar)' and 'Temperature (K)'
    startingFluid : dict
        dictionary of elemental molalities describing the composition of the flushing fluid.
        NB: this is separated from the rest of the function to allow for fluids from different
        pressures and temperatures to the first reaction
    startingfO2 : float
        fO2 of the fluid entering the system
    system : pyDEW.System object
        pyDEW.System for calculations
    timesteps : int
        number of times to flush our solid with the fluid. Default is 10
    startingTime : int
        integer representing time at the start of the reaction. Default is 0
    solventMass : float or None
        starting solvent mass entering the base of the system, in g. If None, use startingFRR.
        Supercedes startingFRR input. Default is None
    startingFRR : float
        fluid-rock ratio (FRR) at the column base between first rock and fluid. Default is 1.0
    terminalFRR : float
        fluid-rock ratio (FRR) to terminate reactions. Default is 0.1
    molalityLimit : float
        log molality to consider a fluid exhausted in a particular element. Default is -18.0
    exportPath : string
        path to export xlsx file. Default is 'data/export'
    outputToRead : string
        path to pyDEW.Reaction output file. Default is 'working/output'
    tabToRead : string
        path to pyDEW.Reaction tab file. Default is 'working/tab'
    startFromFile : string
        path to output file to use as starting composition, from startingTime. Supercedes
        startingColumn input. Default is 'None'
    combinedOutput : boolean
        if startFromFile, combine previous file with new results? Default is True
    debugLabels : boolean
        save EQ3/6 inputs/outputs to a folder in the working directory
    **kwargs for pyDEW.Fluid and pyDEW.Reaction functions

    RETURNS
    -------
    xlsx file, exported to exportPath
        Excel file with solid, element, and fluid speciation data

    """
    print('-----RELAY STARTING-----')
    print('Relay function initiated')
    start = timer()
    t = 0
    allCombinedOutputs = []
    totalCalculations = 0
    if debugLabels:
        if not os.path.exists('working/debug'):
            os.makedirs('working/debug')
    while t < timesteps:
        if t == 0:
            if startFromFile is None:
                workingColumn = startingColumn
                print('Starting from scratch...')
            else:
                workingColumn = relayInputFromOutput(startFromFile, startingTime)
                print(f'Starting from file ({startFromFile}) at t={startingTime}')
        else:
            workingColumn = lastColumn

        # Track errors with these parameters
        calculations = 0
        passes = 0
        errors = 0
        outputs = []

        ##### DECOMPRESSION RELAY #####
        for n in range(len(workingColumn)):
            # if t == 2:
            #     print(t, n)
            workingParameters = workingColumn.iloc[n]
            workingReactant = pDF.DFtoReactant(workingParameters, system)
            workingT = workingParameters['Temperature (K)']
            workingP = workingParameters['Pressure (bar)']

            ##### FLUID #####
            # At start, get parameters from the input, else use results from last reaction
            if n == 0:
                if solventMass is None:
                    workingFRR = startingFRR
                    if t == 0:
                        slabSolventMass = startingFRR * pDF.findReactantMass(workingReactant, system)
                else:
                    workingFRR = solventMass / pDF.findReactantMass(workingReactant, system)
                    slabSolventMass = solventMass
                workingSolventMass = slabSolventMass
                workingFluidMolalities = startingFluid
                workingfO2 = startingfO2
            else:
                reactionInput = deepcopy(reactionOutput)
                workingFRR = reactionInput['Properties']['final FRR']
                workingSolventMass = reactionInput['Properties']['final solvent mass']
                workingfO2 = reactionInput['Properties']['final fluid fO2']
                workingFluidMolalities = formatFluid(reactionInput['Elements'])

                # Format from log molal to molal, check for very low molalities
                workingFluidMolalities.update(
                    (k, 10**v) for k, v in workingFluidMolalities.items())
                for i in workingFluidMolalities:
                    if workingFluidMolalities[i] < 10.0**(molalityLimit):
                        workingFluidMolalities[i] = 0.0

            # Suppress H2CO3(AQ) - shouldn't matter according to Guillaume
            nxmods = [['H2CO3(AQ)', 0, -1, 0]]
            errorTag = 0

            ##### FLUID #####
            try:
                workingFluid = pyDEW.Fluid(system, workingT, workingP,
                           molalities=workingFluidMolalities, fO2=workingfO2,
                           uebal='H+', uacion='CL-', nxmods=nxmods,
                           aH2O_mode='molfraction')
            except IndexError:
                errorTag = 2

            if debugLabels:
                shutil.copy('working/input', f'working/debug/input_EQ3_{t}_{n}')
                shutil.copy('working/output', f'working/debug/output_EQ3_{t}_{n}')

            ##### REACTION #####
            try:
                # If we are below our FRR threshold, flag error
                if workingFRR < terminalFRR:
                    passes+=1
                    errorMessage = f'F/R threshold ({terminalFRR}) achieved at {n=},{t=}'
                    errorTag = 1
                    raise pyDEW.core.RunError(errorMessage)

                if errorTag == 2:
                    errorMessage = f'Fluid failed to equilibrate at {n=},{t=}'
                    raise pyDEW.core.RunError(errorMessage)

                reactionOutput = reactionByMass(workingReactant, workingFluid,
                                                workingSolventMass, system,
                                                workingP, workingT)
                calculations+=1
            except (KeyError, pyDEW.core.RunError):
                errors+=1
                if errorTag != 1  or errorTag != 2:
                    errorTag = 0
                reactionOutput = failedReactionOutput(workingReactant, workingFluid,
                                                      workingSolventMass, system,
                                                      workingP, workingT, errorTag)
            if debugLabels:
                shutil.copy('working/input', f'working/debug/input_EQ6_{t}_{n}')
                shutil.copy('working/output', f'working/debug/output_EQ6_{t}_{n}')
            outputs.append(reactionOutput)
        combinedOutputs = combineOutputs(outputs, system)
        allCombinedOutputs.append(combinedOutputs)
        totalCalculations+=calculations
        if timesteps - t > 1:
            lastColumn = deepcopy(combinedOutputs[1])
            lastColumn['Temperature (K)'] = combinedOutputs[0]['Temperature (K)'].values
            lastColumn['Pressure (bar)'] = combinedOutputs[0]['Pressure (bar)'].values
            print(f't={startingTime+t+1}, {calculations=}, {errors=} of which FRR {passes=}')
            t+=1
        else:
            # Need to add starting conditions to the excel file
            # Combine times into a temporary file
            combineTimes(allCombinedOutputs, startingTime, exportPath+'_temp')
            dfStartSolid = startingColumn
            if 'Time' not in dfStartSolid.columns:
                dfStartSolid['Time'] = [startingTime] * len(startingColumn)
            startingFluidMod = deepcopy(startingFluid)
            startingFluidMod.update(
                (i, np.log10(j)) for i, j in startingFluidMod.items())
            dfStartFluid = pd.DataFrame([startingFluidMod])
            if 'Time' not in dfStartFluid.columns:
                dfStartFluid['Time'] = [startingTime]
            print(f't={startingTime+t+1}, {calculations=}, {errors=} of which FRR {passes=}')
            with pd.ExcelWriter(exportPath+'_temp.xlsx', engine='openpyxl', mode='a') as writer:
                dfStartSolid.to_excel(writer, sheet_name='starting_solid', index=False)
                dfStartFluid.to_excel(writer, sheet_name='starting_fluid', index=False)

            # If we need to stitch to another Excel file, we do that here, else just rename the temp
            if startFromFile != None:
                if combinedOutput:
                    print(f'Combining output with {startFromFile}')
                    combineExcelOutputs([startFromFile, exportPath+'_temp.xlsx'], exportPath)
                    os.remove(exportPath+'_temp.xlsx')
            else:
                os.rename(exportPath+'_temp.xlsx', exportPath+'.xlsx')

            print('-----RELAY COMPLETE-----')
            end = timer()
            print(f'Total iterations : {timesteps*len(startingColumn)}')
            print(f'Total successful DEW-EQ3/6 calculations : {totalCalculations}')
            print(f'Total runtime : {round(end-start, 1)} seconds')
            print(f'Total runtime : {round((end-start)/60, 1)} minutes')
            print(f'Data exported to: {exportPath}.xlsx')
            print('')
            return
