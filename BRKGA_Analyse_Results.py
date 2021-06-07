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
import matplotlib.pyplot as plt

#diretorio onde queremos ler os resultados
PATHInstancia = '/Users/LuisDias/Desktop/resultados_stopping_criterion/BRKGA_cplex/'
PATH_benchmark = '/Users/LuisDias/Desktop/resultados_stopping_criterion/benchmark/'

#Diretorio onde queremos guardar os resultados
PATH_results = '/Users/LuisDias/Desktop/Doutoramento DEGI/A-Papers LUIS DIAS/3_paper/5 - Resultados/BRKGA_Asset_GRID_Laplace_stopping_criterion/resultados/'

#Caracterista da distribuição da condição inicial dos ativos
InstanceFamily = ["Clustered", "Concentrated", "Random"]

#Caracteristicas das instancias a analisar nos resultados
CharacteristicList = ['Clustered','Concentrated','Random','N30','TW10','HighUnc','LowUnc','HighRisk','LowRisk','HighImp','LowImp']
ColumnNames = ['Instance_type','Instance_type','Instance_type','Portfolio_size','Planning_horizon','Uncertainty_level','Uncertainty_level','Risk_level','Risk_level','Maintenance_level','Maintenance_level']

##################################################
###-----Funcoes para estudar os resultados-----###
##################################################

#Funcao para agregar os resultados
def join_results(FitnessData,PATHInstancia,Filename,colnames):

    #Ler os resultados de uma instancia em particular
    Results = pd.read_csv(PATHInstancia + Filename, sep='\t', index_col=False)

    #Colocar versão do modelo BRKGA utilizado com base no nome da insancia!!!!
    Results['ModelVersion'] = Filename[0:6]

    #Agregar resultados
    NewFitnessData = FitnessData.append(Results)

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
def evaluate_benchmark_results(BenchmarkData, InitialScenarioBenchmarkValues, BRKGA_Values, FamilyType):

    #Filtrar pelas colunas de interesse
    BenchmarkData = BenchmarkData.filter(items=['Instance','ModelVersion','Generation','Period','Time_window_Length','Accumulated_Fitness'])
    InitialScenarioBenchmarkValues = InitialScenarioBenchmarkValues.filter(items=['Instance','ModelVersion','Generation','Period','Time_window_Length','Accumulated_Fitness'])
    BRKGA_Values = BRKGA_Values.filter(items=['Instance','ModelVersion','Generation','Period','Time_window_Length','Accumulated_Fitness'])

    #Filtrar para uma familia de instancias
    BenchmarkData = BenchmarkData[BenchmarkData['Instance'].str.contains(FamilyType)]
    InitialScenarioBenchmarkValues = InitialScenarioBenchmarkValues[InitialScenarioBenchmarkValues['Instance'].str.contains(FamilyType)]
    BRKGA_Values = BRKGA_Values[BRKGA_Values['Instance'].str.contains(FamilyType)]

    #Converter as colunas numéricas para o formato correto
    for variable in ['Generation','Period','Time_window_Length','Accumulated_Fitness']:
        BenchmarkData[variable] = pd.to_numeric(BenchmarkData[variable], errors='coerce')
        InitialScenarioBenchmarkValues[variable] = pd.to_numeric(InitialScenarioBenchmarkValues[variable], errors='coerce')
        BRKGA_Values[variable] = pd.to_numeric(BRKGA_Values[variable], errors='coerce')

    #Filtrar pelo último periodo da geração
    MaxGeneration = InitialScenarioBenchmarkValues.Generation.max()
    BenchmarkData = BenchmarkData.loc[(BenchmarkData.Generation == MaxGeneration)]
    InitialScenarioBenchmarkValues = InitialScenarioBenchmarkValues.loc[(InitialScenarioBenchmarkValues.Generation == MaxGeneration)]
    BRKGA_Values = BRKGA_Values.loc[(BRKGA_Values.Generation == MaxGeneration)]

    #Filtrar pelo último periodo
    BenchmarkData['LastPeriod'] = BenchmarkData['Period'] - BenchmarkData['Time_window_Length'] + 1
    BenchmarkData = BenchmarkData.loc[(BenchmarkData.LastPeriod == 0)]
    InitialScenarioBenchmarkValues['LastPeriod'] = InitialScenarioBenchmarkValues['Period'] - InitialScenarioBenchmarkValues['Time_window_Length'] + 1
    InitialScenarioBenchmarkValues = InitialScenarioBenchmarkValues.loc[(InitialScenarioBenchmarkValues.LastPeriod == 0)]
    BRKGA_Values['LastPeriod'] = BRKGA_Values['Period'] - BRKGA_Values['Time_window_Length'] + 1
    BRKGA_Values = BRKGA_Values.loc[(BRKGA_Values.LastPeriod == 0)]

    #Calcular o valor do fitness
    BenchmarkData = BenchmarkData.groupby(["Instance", "ModelVersion"], as_index=False)["Accumulated_Fitness"].mean()
    InitialScenarioBenchmarkValues = InitialScenarioBenchmarkValues.groupby(["Instance", "ModelVersion"], as_index=False)["Accumulated_Fitness"].mean()
    BRKGA_Values = BRKGA_Values.groupby(["Instance", "ModelVersion"], as_index=False)["Accumulated_Fitness"].mean()

    #Retirar a lista de instancias e variantes do modelo com solução benchmark
    ListaInstancias = np.unique(BenchmarkData['Instance'])
    ListaModelVersion = np.unique(BenchmarkData['ModelVersion'])

    #Iniciar a coluna do gap
    Resultados = BenchmarkData
    Resultados['Accumulated_Fitness_Initial_Scenario'] = "Not_found"
    Resultados['Accumulated_Fitness_BRKGA'] = "Not_found"
    Resultados['Solution_method_GAP_Initial_Scenario'] = "No_solution"
    Resultados['Solution_method_GAP_BRKGA'] = "No_solution"

    #Calcular o gap para as instancias e variantes do modelo com solução no benchmark
    for instancia in ListaInstancias:
        for modelo in ListaModelVersion:

            #Analisar melhor solução caso exista
            BenchmarkCombinationValue = BenchmarkData['Accumulated_Fitness'].loc[(BenchmarkData.Instance == instancia) & (BenchmarkData.ModelVersion == modelo)]

            #Solução do método de solução
            BaselineCombinationValue = InitialScenarioBenchmarkValues['Accumulated_Fitness'].loc[(InitialScenarioBenchmarkValues.Instance == instancia) & (InitialScenarioBenchmarkValues.ModelVersion == modelo)]

            #Solução do método de solução
            BRKGACombinationValue = BRKGA_Values['Accumulated_Fitness'].loc[(BRKGA_Values.Instance == instancia) & (BRKGA_Values.ModelVersion == modelo)]

            #Calcular GAP caso existam os dois valores
            if (BenchmarkCombinationValue.empty == False) & (BaselineCombinationValue.empty == False):
                # Calculo do GAP
                BenchmarkValue = BenchmarkCombinationValue.values[0]
                BaselineValue = BaselineCombinationValue.values[0]
                GAP_result = (BenchmarkValue - BaselineValue) / (BenchmarkValue + 1)

                #Alocar resultado do GAP
                Resultados['Solution_method_GAP_Initial_Scenario'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo)] = GAP_result

                # Juntar valor do solution method fitness
                Resultados['Accumulated_Fitness_Initial_Scenario'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo)] = BaselineValue

            #Calcular GAP caso existam os dois valores
            if (BenchmarkCombinationValue.empty == False) & (BRKGACombinationValue.empty == False):
                # Calculo do GAP
                BenchmarkValue = BenchmarkCombinationValue.values[0]
                BaselineValue = BRKGACombinationValue.values[0]
                GAP_result = (BenchmarkValue - BaselineValue) / (BenchmarkValue + 1)

                #Alocar resultado do GAP
                Resultados['Solution_method_GAP_BRKGA'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo)] = GAP_result

                # Juntar valor do solution method fitness
                Resultados['Accumulated_Fitness_BRKGA'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo)] = BaselineValue

    return Resultados

#Funcao para analisar a scenario diversity
def ScenarioDiversityPlot(PATHInstancia, instance_name, PATH_scenario_diversity_filename, FamilyType):

    #Ler os dados
    scenario_diversity_data = pd.read_csv(PATHInstancia + PATH_scenario_diversity_filename, sep='\t')

    #Filtrar o tipo de instancia
    scenario_diversity_data = scenario_diversity_data[scenario_diversity_data['Instance'].str.contains(FamilyType)]

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
    plt.title(f'{FamilyType}_{instance_name}')
    plt.legend(['Initial Generation','Last Generation'])
    plt.xlabel("Best Solution")
    plt.ylabel("Worst Solution")
    #plt.xlim(0)
    #plt.ylim(0)
    plt.show()

#Funcao para analisar a fitness da solução
def FitnessPlot(PATHInstancia, instance_name, PATH_fitness_values_filename, FamilyType):

    #Ler os dados
    fitness_data = pd.read_csv(PATHInstancia + PATH_fitness_values_filename, sep='\t', index_col=False)

    #Filtrar o tipo de instancia
    fitness_data = fitness_data[fitness_data['Instance'].str.contains(FamilyType)]

    #Converter as colunas numéricas para o formato correto
    fitness_data['Generation'] = pd.to_numeric(fitness_data['Generation'], errors='coerce')
    fitness_data['Fitness_value'] = pd.to_numeric(fitness_data['Fitness_value'], errors='coerce')

    #Plot dos dados
    plt.plot(fitness_data['Generation'], fitness_data['Fitness_value'], 'o', color='green')

    #Corrigir as labels do plot
    plt.title(f'{FamilyType}_{instance_name}')
    plt.xlabel("Generation")
    plt.ylabel("Fitness_value")
    plt.show()

#Funcao para analisar a fitness do cenário
def ScenarioFitnessPlot(PATHInstancia, instance_name, PATH_fitness_values_filename, FamilyType):

    #Ler os dados
    fitness_data = pd.read_csv(PATHInstancia + PATH_fitness_values_filename, sep='\t', index_col=False)

    #Filtrar o tipo de instancia
    fitness_data = fitness_data[fitness_data['Instance'].str.contains(FamilyType)]

    #Converter as colunas numéricas para o formato correto
    fitness_data['Generation'] = pd.to_numeric(fitness_data['Generation'], errors='coerce')
    fitness_data['Scenario_Fitness_value'] = pd.to_numeric(fitness_data['Scenario_Fitness_value'], errors='coerce')

    #Plot dos dados
    plt.plot(fitness_data['Generation'], fitness_data['Scenario_Fitness_value'], 'o', color='green')

    #Corrigir as labels do plot
    plt.title(f'{FamilyType}_{instance_name}')
    plt.xlabel("Generation")
    plt.ylabel("Scenario_Fitness_value")
    plt.show()

#Funcao para analisar a solution robustness
def evaluate_solution_robustness(BRKGAData_with_stopping_criterion, FamilyType, Generation_without_stopping_criterion, Generation_with_stopping_criterion):

    #Filtrar pelas colunas de interesse
    BRKGAData_with_stopping_criterion = BRKGAData_with_stopping_criterion.filter(items=['Instance', 'ModelVersion', 'Generation','Scenario', 'Period', 'Time_window_Length', 'Accumulated_Fitness'])

    #Filtrar para uma familia de instancias
    BRKGAData_with_stopping_criterion = BRKGAData_with_stopping_criterion[BRKGAData_with_stopping_criterion['Instance'].str.contains(FamilyType)]

    #Converter as colunas numéricas para o formato correto
    for variable in ['Generation','Period','Time_window_Length','Accumulated_Fitness','Scenario']:
        BRKGAData_with_stopping_criterion[variable] = pd.to_numeric(BRKGAData_with_stopping_criterion[variable], errors='coerce')

    #Filtrar pelos valores que dizem respeito aos resultados do BRKGA para o número de gerações sem stopping criterion
    BRKGAData_with_stopping_criterion = BRKGAData_with_stopping_criterion.loc[(BRKGAData_with_stopping_criterion.Generation == Generation_without_stopping_criterion) | (BRKGAData_with_stopping_criterion.Generation == Generation_with_stopping_criterion)]


    #Filtrar pelo último periodo
    BRKGAData_with_stopping_criterion['LastPeriod'] = BRKGAData_with_stopping_criterion['Period'] - BRKGAData_with_stopping_criterion['Time_window_Length'] + 1
    BRKGAData_with_stopping_criterion = BRKGAData_with_stopping_criterion.loc[(BRKGAData_with_stopping_criterion.LastPeriod == 0)]

    #Calcular o valor do fitness
    BRKGAData_with_stopping_criterion = BRKGAData_with_stopping_criterion.groupby(["Instance", "ModelVersion","Generation","Scenario"], as_index=False)["Accumulated_Fitness"].mean()

    #Retirar a lista de instancias e variantes do modelo com solução benchmark
    ListaInstancias = np.unique(BRKGAData_with_stopping_criterion['Instance'])
    ListaModelVersion = np.unique(BRKGAData_with_stopping_criterion['ModelVersion'])
    ListaScenarios = np.unique(BRKGAData_with_stopping_criterion['Scenario'])

    #Iniciar as colunas de interesse
    Resultados = BRKGAData_with_stopping_criterion[BRKGAData_with_stopping_criterion.Generation==Generation_with_stopping_criterion]
    Resultados = Resultados.rename(columns = {'Accumulated_Fitness': 'Fitness_with_stopping_criterion'}, inplace = False)
    Resultados['Fitness_without_stopping_criterion'] = -1

    #Calcular o gap para as instancias e variantes do modelo com solução no benchmark
    for instancia in ListaInstancias:
        for modelo in ListaModelVersion:
            for scen in ListaScenarios:

                # Solução do método de solução
                CombinationValue = BRKGAData_with_stopping_criterion['Accumulated_Fitness'].loc[(BRKGAData_with_stopping_criterion.Instance == instancia) &
                    (BRKGAData_with_stopping_criterion.ModelVersion == modelo) & (BRKGAData_with_stopping_criterion.Scenario == scen) & (BRKGAData_with_stopping_criterion.Generation == Generation_without_stopping_criterion)]

                #Alocar o valor encontrado
                if CombinationValue.empty == True:
                    CombinationValue = -1
                else:
                    CombinationValue = CombinationValue.values[0]

                # Juntar valor do solution method fitness
                Resultados['Fitness_without_stopping_criterion'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo) & (Resultados.Scenario == scen)] = CombinationValue

    return Resultados

#Funcao para extrair as caracteristicas das instancias
def get_instance_characteristics(data, CharacteristicList, ColumnNames):

    #Create characteristic if it does not exist
    for column in np.unique(ColumnNames):
        if column not in data:
            data[column] = -1

    #Get characteristic
    for characteristic, column in zip(CharacteristicList,ColumnNames):
        data[column][data['Instance'].str.contains(characteristic)] = characteristic

    return data

#Funcao para construir os boxplots que permitem avaliar a robustez das soluções
def plot_solution_robustness(data,ModelVersion,StudyCharacteristics):

    #Filter the results for a specific model version
    data = data[data['ModelVersion'].str.contains(ModelVersion)]

    #Generate boxplot data
    boxplot_data = []
    for characteristic in StudyCharacteristics:
        boxplot_data.append(data['Fitness_without_stopping_criterion'][data['Instance'].str.contains(characteristic)])
        boxplot_data.append(data['Fitness_with_stopping_criterion'][data['Instance'].str.contains(characteristic)])

    # Creating plot
    box = plt.boxplot(boxplot_data, patch_artist=True)
    plt.xticks(range(1,len(StudyCharacteristics)*2+1,2),StudyCharacteristics)

    colors = ['blue', 'green']
    position = 0
    for patch in box['boxes']:
        patch.set_facecolor(colors[position])

        #Color position
        position += 1
        if position > 1:
            position = 0

    # show plot
    plt.title(f'{ModelVersion} - Solution Robustness')
    plt.show()

##########################################################
###-----Rotina para gerar os diferentes resultados-----###
##########################################################

####Ler dados

#Ler os resultado fitness com a evolução do fitness por geracao
FitnessValues = search_results_to_study(PATHInstancia, 'BRKGA_solution_fitness', ['Instance','Generation','Fitness_value','None'])

#Ler os resultado fitness com a evolução do fitness do cenario por geracao
ScenarioFitnessValues = search_results_to_study(PATHInstancia, 'BRKGA_scenario_fitness', ['Instance','Generation','Scenario_Fitness_value','None'])

#Ler os resultado do tempo do BRKGA
TimeValues = search_results_to_study(PATHInstancia, 'BRKGA_time_BRKGA', ['Instance','Generation','Time','None'])

#Ler os resultado do tempo do benchmark
TimeBenchmarkValues = search_results_to_study(PATHInstancia, 'BRKGA_time_benchmark', ['Instance','Generation','Time_benchmark','Time_benchmark_with_fixed_solution'])

#Ler os resultados do tempo do benchmark
TimeSubProblemValues = search_results_to_study(PATHInstancia, 'BRKGA_time_sub_problem', ['Instance','Generation','Time_sub_problem','None'])

#Ler os resultados do stopping criterion
StoppingCriterionValues = search_results_to_study(PATHInstancia, 'stopping_criterion', ['Instance','Stopping_Generation','Fitness_stopping_criterion','Scenario_stopping_criterion'])

#Ler os resultados do output mais detalhado do BRKGA
BaselineValues = search_results_to_study(PATHInstancia, 'baseline_solution', ['Instance','Generation','Solution','Scenario','Period','Assets_number','Time_window_Length',
                                                                                   'Maintenance_Types','Fitness','Defined_Budget','Over_Budget','RUL_Dispersion','Budget_GAP',
                                                                                   'End_of_Horizon_Effect','OM_Costs','OM_PlannedCosts','OM_UnlannedCosts','Accumulated_Fitness',
                                                                                   'Accumulated_Budget','Accumulated_Overbudget','Accumulated_RUL','Accumulated_Budget_Gap',
                                                                                   'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts'],True)


#Ler os resultados do benchmark
BenchmarkValues = search_results_to_study(PATH_benchmark, 'benchmark_solution.csv', ['Instance','Generation','Solution','Scenario','Period','Assets_number','Time_window_Length',
                                                                                 'Maintenance_Types','Fitness','Defined_Budget','Over_Budget','RUL_Dispersion','Budget_GAP',
                                                                                 'End_of_Horizon_Effect','OM_Costs','OM_PlannedCosts','OM_UnlannedCosts','Accumulated_Fitness',
                                                                                 'Accumulated_Budget','Accumulated_Overbudget','Accumulated_RUL','Accumulated_Budget_Gap',
                                                                                 'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts'],True)

#Ler os resultados da solução baseline
BaselineBenchmarkValues = search_results_to_study(PATH_benchmark, 'initial_scenario', ['Instance','Generation','Solution','Scenario','Period','Assets_number','Time_window_Length',
                                                                                 'Maintenance_Types','Fitness','Defined_Budget','Over_Budget','RUL_Dispersion','Budget_GAP',
                                                                                 'End_of_Horizon_Effect','OM_Costs','OM_PlannedCosts','OM_UnlannedCosts','Accumulated_Fitness',
                                                                                 'Accumulated_Budget','Accumulated_Overbudget','Accumulated_RUL','Accumulated_Budget_Gap',
                                                                                 'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts'],True)

#Ler os resultados da solução BRKGA
BRKGA_Values = search_results_to_study(PATH_benchmark, 'benchmark_solution_BRKGA', ['Instance','Generation','Solution','Scenario','Period','Assets_number','Time_window_Length',
                                                                                                  'Maintenance_Types','Fitness','Defined_Budget','Over_Budget','RUL_Dispersion','Budget_GAP',
                                                                                                  'End_of_Horizon_Effect','OM_Costs','OM_PlannedCosts','OM_UnlannedCosts','Accumulated_Fitness',
                                                                                                  'Accumulated_Budget','Accumulated_Overbudget','Accumulated_RUL','Accumulated_Budget_Gap',
                                                                                                  'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts'],True)

####Analisar resultados

#Calcular o gap entre a solução do benchmark e a solução do método de solução
GapResults = evaluate_benchmark_results(BenchmarkValues, BaselineBenchmarkValues, BRKGA_Values, InstanceFamily[0])
for name in InstanceFamily[1:]:
    NewGapResults = evaluate_benchmark_results(BenchmarkValues, BaselineBenchmarkValues, BRKGA_Values, name)
    GapResults = pd.concat([GapResults, NewGapResults])

#Calcular a robustez dos resultados
RobustnessResults = evaluate_solution_robustness(BaselineValues, InstanceFamily[0], 1000, 9999)
for name in InstanceFamily[1:]:
    NewRobustnessResults = evaluate_solution_robustness(BaselineValues, name, 1000, 9999)
    RobustnessResults = pd.concat([RobustnessResults, NewRobustnessResults])

#Extrair caracteristicas das instancias
GapResults = get_instance_characteristics(GapResults, CharacteristicList, ColumnNames)
RobustnessResults = get_instance_characteristics(RobustnessResults, CharacteristicList, ColumnNames)
TimeBenchmarkValues = get_instance_characteristics(TimeBenchmarkValues, CharacteristicList, ColumnNames)
TimeValues = get_instance_characteristics(TimeValues, CharacteristicList, ColumnNames)
TimeSubProblemValues = get_instance_characteristics(TimeSubProblemValues, CharacteristicList, ColumnNames)
StoppingCriterionValues = get_instance_characteristics(StoppingCriterionValues, CharacteristicList, ColumnNames)

#Plot dos resultados
instance_name = 'R0H1E0_N30TW10_HighUncHighRiskHighImp'

# Analizar a diversidade dos cenários
for name in InstanceFamily:
    ScenarioDiversityPlot(PATHInstancia, instance_name, f'{instance_name}_scenario_diversity.csv', name)

#Analisar a fitness function de uma solução em particular
for name in InstanceFamily:
    FitnessPlot(PATHInstancia, instance_name, f'{instance_name}_BRKGA_solution_fitness.csv', name)

#Analisar a fitness function de um cenário em particular
for name in InstanceFamily:
    ScenarioFitnessPlot(PATHInstancia, instance_name, f'{instance_name}_BRKGA_scenario_fitness.csv', name)

#Analisar robustez das soluções para cada versão
plot_solution_robustness(RobustnessResults,'R0H0E0',['Clustered','Concentrated','Random'])
plot_solution_robustness(RobustnessResults,'R0H1E0',['Clustered','Concentrated','Random'])
plot_solution_robustness(RobustnessResults,'R0H0E1',['Clustered','Concentrated','Random'])
plot_solution_robustness(RobustnessResults,'R0H1E1',['Clustered','Concentrated','Random'])

#Analisar robustez das soluções mediante as caracteristicas das instancias
plot_solution_robustness(RobustnessResults,'R0H0E0',['HighUnc','LowUnc','HighRisk','LowRisk','HighImp','LowImp'])
plot_solution_robustness(RobustnessResults,'R0H1E0',['HighUnc','LowUnc','HighRisk','LowRisk','HighImp','LowImp'])
plot_solution_robustness(RobustnessResults,'R0H0E1',['HighUnc','LowUnc','HighRisk','LowRisk','HighImp','LowImp'])
plot_solution_robustness(RobustnessResults,'R0H1E1',['HighUnc','LowUnc','HighRisk','LowRisk','HighImp','LowImp'])

#Export dos resultados
GapResults.to_csv(path_or_buf=f'{PATH_results}Solution_quality.csv',index=False)
RobustnessResults.to_csv(path_or_buf=f'{PATH_results}Robustness_results_quality.csv', index=False)
FitnessValues.to_csv(path_or_buf=f'{PATH_results}Fitness_solution.csv',index=False)
ScenarioFitnessValues.to_csv(path_or_buf=f'{PATH_results}Scenario_Fitness_solution.csv',index=False)
TimeBenchmarkValues.to_csv(path_or_buf=f'{PATH_results}Tempo_computacional_benchmark.csv',index=False)
TimeValues.to_csv(path_or_buf=f'{PATH_results}Tempo_computacional_BRKGA.csv',index=False)
TimeSubProblemValues.to_csv(path_or_buf=f'{PATH_results}Tempo_computacional_BRKGA_sub_problem.csv',index=False)
StoppingCriterionValues.to_csv(path_or_buf=f'{PATH_results}Stopping_criterion.csv',index=False)