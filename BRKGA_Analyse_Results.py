#####################################################################
###Script to generate the results reports for the BRKGA algorithm ###
#####################################################################

#Bibliotecas a ser importadas
from BRKGA_utils import *

##################################################
###-----Funcoes para estudar os resultados-----###
##################################################

#Funcao para agregar os resultados
#Parâmetros:
#FitnessData - Dados a processar
#PATHInstancia - Caminho para a pasta dos dados a processar
#Filename - Nome do ficheiro a processar
def join_results(FitnessData,PATHInstancia,Filename):

    #Ler os resultados de uma instancia em particular
    Results = pd.read_csv(PATHInstancia + Filename, sep='\t', index_col=False)

    #Colocar versão do modelo BRKGA utilizado com base no nome da insancia!!!!
    Results['ModelVersion'] = Filename[0:6]

    #Colocar a combinação de parâmetros (soluções e cenários)
    Results['SolutionParameterization'] = Filename.split('_sol_')[1].split('_scen_')[0]
    Results['ScenarioParameterization'] = Filename.split('_sol_')[1].split('_scen_')[1]

    #Extrair caracteristicas das instancias
    Results = get_instance_characteristics(Results, CharacteristicList, ColumnNames,'Instance')
    Results = get_instance_characteristics(Results, Solution_Parameters_CharacteristicList, Solution_Parameters_ColumnNames, 'SolutionParameterization')
    Results = get_instance_characteristics(Results, Scenario_Parameters_CharacteristicList, Scenario_Parameters_ColumnNames, 'ScenarioParameterization')

    #Agregar resultados
    NewFitnessData = FitnessData.append(Results)

    return NewFitnessData

#Funcao para juntar os resultados das instancias para um dado ambito do estudo
#Parâmetros:
#path_instancia - Caminho para a pasta dos dados a processar
#object_of_study - Tipo de ficheiro que queremos analisar (ex: Fitness das soluções, Tempo computacional)
#colnames - Nome das colunas que estamos à espera de encontrar
#remove_first_row - Valor 'True' caso queiramos remover a primeira linha dos dados
def search_results_to_study(path_instancia,object_of_study,colnames, remove_first_row = False):

    #Ler todos os ficheiros que estão na pasta
    File_list = os.listdir(path_instancia)

    #Filtrar pelos ficheiros de interesse
    File_list = [f for f in File_list if object_of_study in f]

    #Dataframe que vai guardar os resultados
    Results = pd.DataFrame(columns=colnames)

    #Ler todos os resultados e agregá-los numa lista
    for filename in File_list:
        Results = join_results(Results,path_instancia,filename)

    #Verificar se é necessário retirar a primeira linha
    if remove_first_row == True:
        Results = Results.drop([0]).reset_index(drop=True)


    return Results

#Funcao para analisar os resultados do benchmark
#Parâmetros:
#BenchmarkData - Dados que referem ao modelo benchmark na população de cenários final
#InitialScenarioBenchmarkValues - Dados que referem ao modelo benchmark na população de cenários inicial
#BRKGA_Values - Resultados obtidos a partir do BRKGA
#FamilyType - Tipo de instância que queremos analisar ('Clustered, 'Random', 'Concentrated')
def evaluate_benchmark_results(BenchmarkData, InitialScenarioBenchmarkValues, BRKGA_Values, FamilyType):

    #Filtrar pelas colunas de interesse
    BenchmarkData = BenchmarkData.filter(items=['Instance','ModelVersion','Generation','Period','Time_window_Length','Accumulated_Budget','Accumulated_Overbudget','Cplex_GAP'])
    InitialScenarioBenchmarkValues = InitialScenarioBenchmarkValues.filter(items=['Instance','ModelVersion','Generation','Period','Time_window_Length','Accumulated_Budget','Accumulated_Overbudget','Cplex_GAP'])
    BRKGA_Values = BRKGA_Values.filter(items=['Instance','ModelVersion','Generation','Period','Time_window_Length','Accumulated_Budget','Accumulated_Overbudget','Cplex_GAP'])

    #Filtrar para uma familia de instancias
    BenchmarkData = BenchmarkData[BenchmarkData['Instance'].str.contains(FamilyType)]
    InitialScenarioBenchmarkValues = InitialScenarioBenchmarkValues[InitialScenarioBenchmarkValues['Instance'].str.contains(FamilyType)]
    BRKGA_Values = BRKGA_Values[BRKGA_Values['Instance'].str.contains(FamilyType)]

    #Converter as colunas numéricas para o formato correto
    for variable in ['Generation','Period','Time_window_Length','Accumulated_Budget','Accumulated_Overbudget','Cplex_GAP']:
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
    BenchmarkData = BenchmarkData.groupby(["Instance", "ModelVersion"], as_index=False)["Accumulated_Budget","Accumulated_Overbudget","Cplex_GAP"].mean()
    InitialScenarioBenchmarkValues = InitialScenarioBenchmarkValues.groupby(["Instance", "ModelVersion"], as_index=False)["Accumulated_Budget","Accumulated_Overbudget"].mean()
    BRKGA_Values = BRKGA_Values.groupby(["Instance", "ModelVersion"], as_index=False)["Accumulated_Budget","Accumulated_Overbudget"].mean()

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
            BenchmarkCombinationValue = BenchmarkData['Accumulated_Budget'].loc[(BenchmarkData.Instance == instancia) & (BenchmarkData.ModelVersion == modelo)]

            #Solução do método de solução
            BaselineCombinationValue = InitialScenarioBenchmarkValues['Accumulated_Budget'].loc[(InitialScenarioBenchmarkValues.Instance == instancia) & (InitialScenarioBenchmarkValues.ModelVersion == modelo)]

            #Solução do método de solução
            BRKGACombinationValue = BRKGA_Values['Accumulated_Budget'].loc[(BRKGA_Values.Instance == instancia) & (BRKGA_Values.ModelVersion == modelo)] + BRKGA_Values['Accumulated_Overbudget'].loc[(BRKGA_Values.Instance == instancia) & (BRKGA_Values.ModelVersion == modelo)]

            #Calcular GAP caso existam os dois valores
            if (BenchmarkCombinationValue.empty == False) & (BaselineCombinationValue.empty == False):

                # Calculo do GAP
                BenchmarkValue = BenchmarkData['Accumulated_Budget'].loc[(BenchmarkData.Instance == instancia) & (BenchmarkData.ModelVersion == modelo)].values[0] + BenchmarkData['Accumulated_Overbudget'].loc[(BenchmarkData.Instance == instancia) & (BenchmarkData.ModelVersion == modelo)].values[0]
                BaselineValue = InitialScenarioBenchmarkValues['Accumulated_Budget'].loc[(InitialScenarioBenchmarkValues.Instance == instancia) & (InitialScenarioBenchmarkValues.ModelVersion == modelo)].values[0] + InitialScenarioBenchmarkValues['Accumulated_Overbudget'].loc[(InitialScenarioBenchmarkValues.Instance == instancia) & (InitialScenarioBenchmarkValues.ModelVersion == modelo)].values[0]
                GAP_result = (BenchmarkValue - BaselineValue) / (BenchmarkValue + 1)

                #Alocar resultado do GAP
                Resultados['Solution_method_GAP_Initial_Scenario'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo)] = GAP_result

                # Juntar valor do solution method fitness
                Resultados['Accumulated_Fitness_Initial_Scenario'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo)] = BaselineValue

            #Calcular GAP caso existam os dois valores
            if (BenchmarkCombinationValue.empty == False) & (BRKGACombinationValue.empty == False):

                # Calculo do GAP
                BenchmarkValue = BenchmarkData['Accumulated_Budget'].loc[(BenchmarkData.Instance == instancia) & (BenchmarkData.ModelVersion == modelo)].values[0] + BenchmarkData['Accumulated_Overbudget'].loc[(BenchmarkData.Instance == instancia) & (BenchmarkData.ModelVersion == modelo)].values[0]
                BaselineValue = BRKGA_Values['Accumulated_Budget'].loc[(BRKGA_Values.Instance == instancia) & (BRKGA_Values.ModelVersion == modelo)].values[0] + BRKGA_Values['Accumulated_Overbudget'].loc[(BRKGA_Values.Instance == instancia) & (BRKGA_Values.ModelVersion == modelo)].values[0]
                GAP_result = (BenchmarkValue - BaselineValue) / (BenchmarkValue + 1)

                #Alocar resultado do GAP
                Resultados['Solution_method_GAP_BRKGA'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo)] = GAP_result

                # Juntar valor do solution method fitness
                Resultados['Accumulated_Fitness_BRKGA'].loc[(Resultados.Instance == instancia) & (Resultados.ModelVersion == modelo)] = BaselineValue

    return Resultados

#Funcao para analisar a fitness da solução
#Parâmetros:
#PATHInstancia - Caminho para a pasta dos dados a processar
#instance_name - Nome da instância que queremos analisar
#PATH_fitness_values_filename - Diretorio em que se encontra a instância
#FamilyType - Tipo de instância que queremos analisar ('Clustered, 'Random', 'Concentrated')
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
    plt.ylim((0,fitness_data['Fitness_value'].max()*1.10))
    #plt.xlim((0,500))
    plt.xlabel("Generation")
    plt.ylabel("Fitness_value")

    #guardar o plot
    plt.savefig(f'{PATH_results}{instance_name}_solution.png')
    plt.show()

#Funcao para analisar a fitness do cenário
#Parâmetros:
#PATHInstancia - Caminho para a pasta dos dados a processar
#instance_name - Nome da instância que queremos analisar
#PATH_fitness_values_filename - Diretorio em que se encontra a instância
#FamilyType - Tipo de instância que queremos analisar ('Clustered, 'Random', 'Concentrated')
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

    #guardar o plot
    plt.savefig(f'{PATH_results}{instance_name}_scenario.png')
    plt.show()

#Funcao para analisar a solution robustness
#Parâmetros:
#BRKGAData_with_stopping_criterion - Resultados do BRKGA utilizando o stopping criterion que foi desenvolvido
#FamilyType - Tipo de instância que queremos analisar ('Clustered, 'Random', 'Concentrated')
#Generation_without_stopping_criterion - Número de gerações utilizado como limite para os resultados do BRKGA sem critério de paragem.
#Generation_with_stopping_criterion - Número de gerações utilizado como limite para os resultados do BRKGA com critério de paragem.
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

#Funcao para construir os boxplots que permitem avaliar a robustez das soluções
#Parâmetros:
#data - Resultados que queremos analisar
#ModelVersion - Versão do modelo que queremos analisar (ex: R0H0E0)
#StudyCharacteristics - Conjunto de características que queremos analisar na instância (ex: Uncertainty level).
#EvaluateModelVersion - Valor 'True' se queremos analisar ao nível das versão do modelo. Caso contrário, analisamos ao nível da instância.
def plot_solution_robustness(data,ModelVersion,StudyCharacteristics,EvaluateModelVersion = False):

    #Calcular o valor do fitness
    data = data.groupby(["Instance", "ModelVersion"], as_index=False)["Fitness_with_stopping_criterion","Fitness_without_stopping_criterion"].mean()
    data = data.reset_index()

    #Calcular a diferença relativa entre o fitness com critério de paragem vs sem critério de paragem [GAP=(com_criterio-sem_criterio)/sem_criterio]
    data['Relative_diference(%)'] = (data['Fitness_with_stopping_criterion']-data['Fitness_without_stopping_criterion'])/data['Fitness_without_stopping_criterion'] * 100

    #Filter the results for a specific model version
    data = data[data['ModelVersion'].str.contains(ModelVersion)]

    #Generate boxplot data
    boxplot_data = []
    for characteristic in StudyCharacteristics:
        if EvaluateModelVersion==False:
            boxplot_data.append(data['Relative_diference(%)'][data['Instance'].str.contains(characteristic)])
        else:
            boxplot_data.append(data['Relative_diference(%)'][data['ModelVersion'].str.contains(characteristic)])

    # Creating plot
    plt.boxplot(boxplot_data, patch_artist=True)
    plt.xticks(range(1,len(StudyCharacteristics)+1,1),StudyCharacteristics)
    plt.ylim((-25,25))
    plt.ylabel('Relative diference (%)')

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
                                                                                   'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts','Cplex_GAP'],True)


#Ler os resultados do benchmark
BenchmarkValues = search_results_to_study(PATH_benchmark, 'benchmark_solution.csv', ['Instance','Generation','Solution','Scenario','Period','Assets_number','Time_window_Length',
                                                                                 'Maintenance_Types','Fitness','Defined_Budget','Over_Budget','RUL_Dispersion','Budget_GAP',
                                                                                 'End_of_Horizon_Effect','OM_Costs','OM_PlannedCosts','OM_UnlannedCosts','Accumulated_Fitness',
                                                                                 'Accumulated_Budget','Accumulated_Overbudget','Accumulated_RUL','Accumulated_Budget_Gap',
                                                                                 'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts','Cplex_GAP'],True)

#Ler os resultados da solução baseline
BaselineBenchmarkValues = search_results_to_study(PATH_benchmark, 'initial_scenario', ['Instance','Generation','Solution','Scenario','Period','Assets_number','Time_window_Length',
                                                                                 'Maintenance_Types','Fitness','Defined_Budget','Over_Budget','RUL_Dispersion','Budget_GAP',
                                                                                 'End_of_Horizon_Effect','OM_Costs','OM_PlannedCosts','OM_UnlannedCosts','Accumulated_Fitness',
                                                                                 'Accumulated_Budget','Accumulated_Overbudget','Accumulated_RUL','Accumulated_Budget_Gap',
                                                                                 'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts','Cplex_GAP'],True)

#Ler os resultados da solução BRKGA
BRKGA_Values = search_results_to_study(PATH_benchmark, 'benchmark_solution_BRKGA', ['Instance','Generation','Solution','Scenario','Period','Assets_number','Time_window_Length',
                                                                                                  'Maintenance_Types','Fitness','Defined_Budget','Over_Budget','RUL_Dispersion','Budget_GAP',
                                                                                                  'End_of_Horizon_Effect','OM_Costs','OM_PlannedCosts','OM_UnlannedCosts','Accumulated_Fitness',
                                                                                                  'Accumulated_Budget','Accumulated_Overbudget','Accumulated_RUL','Accumulated_Budget_Gap',
                                                                                                  'Accumulated_End_of_Horizon','Accumulated_OM_Costs','Accumulated_OM_PlannedCosts','Accumulated_OM_UnlannedCosts','Cplex_GAP'],True)

####Analisar resultados

#Calcular o gap entre a solução do benchmark e a solução do método de solução
GapResults = evaluate_benchmark_results(BenchmarkValues, BaselineBenchmarkValues, BRKGA_Values, InstanceFamily[0])
for name in InstanceFamily[1:]:
    NewGapResults = evaluate_benchmark_results(BenchmarkValues, BaselineBenchmarkValues, BRKGA_Values, name)
    GapResults = pd.concat([GapResults, NewGapResults])

#Calcular a robustez dos resultados
RobustnessResults = evaluate_solution_robustness(BaselineValues, InstanceFamily[0], 1000, 999)
for name in InstanceFamily[1:]:
    NewRobustnessResults = evaluate_solution_robustness(BaselineValues, name, 1000, 999)
    RobustnessResults = pd.concat([RobustnessResults, NewRobustnessResults])

#Plot dos resultados
instance_name = 'R0H0E0_N30TW10_HighUncLowRiskHighImp'

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
plot_solution_robustness(RobustnessResults,'',['R0H0E0','R0H1E0','R0H0E1','R0H1E1'],True)

#Analisar robustez mediante o horizonte de planeamento
plot_solution_robustness(RobustnessResults,'R0H0E0',['TW5','TW10'])
plot_solution_robustness(RobustnessResults,'R0H0E1',['TW5','TW10'])

#Analisar robustez das soluções para cada tipo de distribuição do RUL
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

#######Esta secção é para melhorar na versão do código final##########

#Analise específica (este código depois é para retirar) ->falta distinguir por janela temporal e número de ativos
# File_list = os.listdir(PATHInstancia)
# File_list_fitness =  [f for f in File_list if 'BRKGA_solution_fitness' in f]
# File_list_diversity =  [f for f in File_list if 'scenario_diversity' in f]
# for file,diversity in zip(File_list_fitness,File_list_diversity):
#     PlotTitle = file.split('R0H0E0_')[1].split('_High')[0] + '|' + file.split('_sol_')[1].split('_scen_')[0] + "|" + file.split('_sol_')[1].split('_scen_')[1]
#     FitnessPlot(PATHInstancia, PlotTitle, file, 'Clustered')
#     ScenarioDiversityPlot(PATHInstancia, PlotTitle, diversity, 'Clustered')
