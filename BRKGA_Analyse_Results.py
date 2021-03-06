#####################################################################
###Script to generate the results reports for the BRKGA algorithm ###
#####################################################################


############################################################################################
###-----------------------------Parâmetros globais do código-----------------------------###
############################################################################################

#Bibliotecas a ser importadas
import pandas as pd
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import math

#diretorio onde queremos colocar as instancias
PATHInstancia = '/Users/LuisDias/Desktop/BRKGA_Asset_GRID_Laplace/resultados/BRKGA_cplex/'
PATH_benchmark = '/Users/LuisDias/Desktop/BRKGA_Asset_GRID_Laplace/resultados/benchmark/'

##################################################
###-----Funcoes para estudar os resultados-----###
##################################################

#Funcao para agregar os resultados
def join_results(FitnessData,PATHInstancia,Filename,colnames):

    #Ler os resultados de uma instancia em particular
    Results = pd.read_csv(PATHInstancia + Filename,sep='\t',names=colnames)

    #Colocar versão do modelo BRKGA utilizado com base no nome da insancia!!!!
    Results['ModelVersion'] = Filename[0:6]

    #Agregar resultados
    NewFitnessData = FitnessData.append(Results, ignore_index=True)

    return NewFitnessData

#Funcao para juntar os resultados das instancias para um dado ambito do estudo
def search_results_to_study(path_instancia,object_of_study,colnames, remove_first_row = False):

    #Ler todos os ficheiros que estão na pasta
    File_list = os.listdir(path_instancia)

    #Filtrar pelos ficheiros de interesse
    File_list = [f for f in File_list if object_of_study in f]

    #Dataframe que vai guardar os resultados
    Results = pd.DataFrame(columns=colnames)

    #Ler todos os resultados e agregá-los numa lista
    for filename in File_list:
        Results = join_results(Results,path_instancia,filename,colnames)

    #Verificar se é necessário retirar a primeira linha
    if remove_first_row == True:
        Results = Results.drop([0]).reset_index(drop=True)


    return Results


#Funcao para analisar os resultados do benchmark
def evaluate_benchmark_results(BenchmarkData, BaselineBenchmarkValues):

    #Filtrar pelas colunas de interesse
    BenchmarkData = BenchmarkData.filter(items=['Instance','ModelVersion','Generation','Period','Time_window_Length','Accumulated_Fitness'])
    BaselineBenchmarkValues = BaselineBenchmarkValues.filter(items=['Instance','ModelVersion','Generation','Period','Time_window_Length','Accumulated_Fitness'])

    #Converter as colunas numéricas para o formato correto
    for variable in ['Generation','Period','Time_window_Length','Accumulated_Fitness']:
        BenchmarkData[variable] = pd.to_numeric(BenchmarkData[variable], errors='coerce')
        BaselineBenchmarkValues[variable] = pd.to_numeric(BaselineBenchmarkValues[variable], errors='coerce')

    #Filtrar pelo último periodo da geração
    MaxGeneration = BaselineBenchmarkValues.Generation.max()
    BenchmarkData = BenchmarkData.loc[(BenchmarkData.Generation == MaxGeneration)]
    BaselineBenchmarkValues = BaselineBenchmarkValues.loc[(BaselineBenchmarkValues.Generation == MaxGeneration)]

    #Filtrar pelo último periodo
    BenchmarkData['LastPeriod'] = BenchmarkData['Period'] - BenchmarkData['Time_window_Length'] + 1
    BenchmarkData = BenchmarkData.loc[(BenchmarkData.LastPeriod == 0)]
    BaselineBenchmarkValues['LastPeriod'] = BaselineBenchmarkValues['Period'] - BaselineBenchmarkValues['Time_window_Length'] + 1
    BaselineBenchmarkValues = BaselineBenchmarkValues.loc[(BaselineBenchmarkValues.LastPeriod == 0)]

    #Calcular o valor do fitness
    BenchmarkData = BenchmarkData.groupby(["Instance", "ModelVersion"], as_index=False)["Accumulated_Fitness"].mean()
    BaselineBenchmarkValues = BaselineBenchmarkValues.groupby(["Instance", "ModelVersion"], as_index=False)["Accumulated_Fitness"].mean()

    #Retirar a lista de instancias e variantes do modelo com solução benchmark
    ListaInstancias = np.unique(BenchmarkData['Instance'])
    ListaModelVersion = np.unique(BenchmarkData['ModelVersion'])

    #Iniciar a coluna do gap
    Resultados = BenchmarkData
    Resultados['Accumulated_Fitness_baseline'] = "Not_found"
    Resultados['Solution_method_GAP'] = "No_solution"

    #Calcular o gap para as instancias e variantes do modelo com solução no benchmark
    for instancia in ListaInstancias:
        for modelo in ListaModelVersion:

            #Melhor solução
            BenchmarkCombinationValue = BenchmarkData['Accumulated_Fitness'].loc[(BenchmarkData.Instance == instancia) & (BenchmarkData.ModelVersion == modelo)]

            #Solução do método de solução
            BaselineCombinationValue = BaselineBenchmarkValues['Accumulated_Fitness'].loc[(BaselineBenchmarkValues.Instance == instancia) & (BaselineBenchmarkValues.ModelVersion == modelo)]

            #Calculo do GAP
            GAP_result = (BaselineCombinationValue - BenchmarkCombinationValue) / BenchmarkCombinationValue

            #Alocar resultado do GAP
            Resultados['Solution_method_GAP'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo)] = GAP_result

            # Juntar valor do solution method fitness
            Resultados['Accumulated_Fitness_baseline'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo)] = BaselineCombinationValue

    return Resultados


#Funcao para analisar a scenario diversity
def ScenarioDiversityPlot(PATHInstancia, PATH_scenario_diversity_filename):

    #Ler os dados
    scenario_diversity_data = pd.read_csv(PATHInstancia + PATH_scenario_diversity_filename, sep='\t')

    #Converter as colunas numéricas para o formato correto
    scenario_diversity_data['Generation'] = pd.to_numeric(scenario_diversity_data['Generation'], errors='coerce')
    scenario_diversity_data['Best_Solution_Value'] = pd.to_numeric(scenario_diversity_data['Best_Solution_Value'], errors='coerce')
    scenario_diversity_data['Worst_Solution_Value'] = pd.to_numeric(scenario_diversity_data['Worst_Solution_Value'], errors='coerce')

    #Plot dos dados da primeira geração
    initial_scenario_diversity_data = scenario_diversity_data.loc[(scenario_diversity_data.Generation == 1)]
    plt.plot(initial_scenario_diversity_data['Best_Solution_Value'], initial_scenario_diversity_data['Worst_Solution_Value'], 'o', color='red')

    # Plot dos dados da última geração
    LastPeriod = scenario_diversity_data['Generation'].max()
    last_scenario_diversity_data = scenario_diversity_data.loc[(scenario_diversity_data.Generation == LastPeriod)]
    plt.plot(last_scenario_diversity_data['Best_Solution_Value'], last_scenario_diversity_data['Worst_Solution_Value'], 'o', color='green')

    #Corrigir as labels do plot
    plt.title(PATH_scenario_diversity_filename)
    plt.legend(['Initial Generation','Last Generation'])
    plt.xlabel("Best Solution")
    plt.ylabel("Worst Solution")
    plt.show()

#Funcao para analisar a solution robustness
def evaluate_solution_robustness(RobustnessData):


    #Filtrar pelas colunas de interesse
    RobustnessData = RobustnessData.filter(items=['Instance','ModelVersion','Generation','Period','Time_window_Length','Accumulated_Fitness'])

    #Converter as colunas numéricas para o formato correto
    for variable in ['Generation','Period','Time_window_Length','Accumulated_Fitness']:
        RobustnessData[variable] = pd.to_numeric(RobustnessData[variable], errors='coerce')

    #Filtrar pelo último periodo
    RobustnessData['LastPeriod'] = RobustnessData['Period'] - RobustnessData['Time_window_Length'] + 1
    RobustnessData = RobustnessData.loc[(RobustnessData.LastPeriod == 0)]

    #Calcular o valor do fitness
    Resultados = RobustnessData.groupby(["Instance", "ModelVersion","Generation"], as_index=False)["Accumulated_Fitness"].mean()

    return Resultados

##########################################################
###-----Rotina para gerar os diferentes resultados-----###
##########################################################

####Ler dados

#Ler os resultado fitness
FitnessValues = search_results_to_study(PATHInstancia, 'BRKGA_solution_fitness', ['Instancia','Fitness'])

#Ler os resultado do tempo
TimeValues = search_results_to_study(PATHInstancia, 'BRKGA_time', ['Instancia','Tempo_computacional'])

#Ler os resultados do benchmark
BenchmarkValues = search_results_to_study(PATH_benchmark, 'benchmark_solution', ['Instance','Generation','Solution','Scenario','Period','Assets_number','Time_window_Length',
                                                                                 'Maintenance_Types','Fitness','Defined_Budget','Over_Budget','RUL_Dispersion','Budget_GAP',
                                                                                 'End_of_Horizon_Effect','OM_Costs','OM_PlannedCosts','OM_UnlannedCosts','Accumulated_Fitness',
                                                                                 'Accumulated_Budget','Accumulated_Overbudget','Accumulated_RUL','Accumulated_Budget_Gap',
                                                                                 'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts'],True)

#Ler os resultados da solução baseline
BaselineBenchmarkValues = search_results_to_study(PATH_benchmark, 'benchmark_baseline_solution', ['Instance','Generation','Solution','Scenario','Period','Assets_number','Time_window_Length',
                                                                                 'Maintenance_Types','Fitness','Defined_Budget','Over_Budget','RUL_Dispersion','Budget_GAP',
                                                                                 'End_of_Horizon_Effect','OM_Costs','OM_PlannedCosts','OM_UnlannedCosts','Accumulated_Fitness',
                                                                                 'Accumulated_Budget','Accumulated_Overbudget','Accumulated_RUL','Accumulated_Budget_Gap',
                                                                                 'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts'],True)

#Ler os resultados do benchmark
SolutionRobustnessValues = search_results_to_study(PATHInstancia, 'solution_robustness', ['Instance','Generation','Solution','Scenario','Period','Assets_number','Time_window_Length',
                                                                                 'Maintenance_Types','Fitness','Defined_Budget','Over_Budget','RUL_Dispersion','Budget_GAP',
                                                                                 'End_of_Horizon_Effect','OM_Costs','OM_PlannedCosts','OM_UnlannedCosts','Accumulated_Fitness',
                                                                                 'Accumulated_Budget','Accumulated_Overbudget','Accumulated_RUL','Accumulated_Budget_Gap',
                                                                                 'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts'],True)

####Analisar resultados

#Calcular o gap entre a solução do benchmark e a solução do método de solução
GapResults = evaluate_benchmark_results(BenchmarkValues, BaselineBenchmarkValues)

#Calcular a robustez dos resultados
RobustnessResults = evaluate_solution_robustness(SolutionRobustnessValues)

#Analizar a diversidade dos cenários
ScenarioDiversityPlot(PATHInstancia, 'R0H0E1_N30TW10_LowUncLowRiskHighImp_scenario_diversity.csv')