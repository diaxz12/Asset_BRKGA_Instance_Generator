#####################################################################
###Jobs Generator for BRKGA algorithm runs on FEUP Computing grid ###
#####################################################################


############################################################################################
###-----------------------------Parâmetros globais do código-----------------------------###
############################################################################################

#Bibliotecas a ser importadas
import numpy as np

#Nome da pasta que guarda as instancias
InstancesFolder = 'Tuning_BRKGA_Asset_GRID_Laplace_stopping_criterion'

#diretorio onde queremos colocar as instancias
PATHJobs=f'/Users/LuisDias/Desktop/Doutoramento DEGI/A-Papers LUIS DIAS/3_paper/5 - Resultados/{InstancesFolder}/Job_scripts'

#parametros do gerador de jobs
NumeroInstancias = 1 #numero de instancias a gerar por cada classe de instancia (N[X])
AssetNumberInstances=np.array([20,100]) #Lista do numero de ativos
TimeWindow = np.array([5,10]) #Lista de Planning horizons
TimeLimit = 48 #Tempo limite que o job pode ser executado no grid (em Horas)
NumberOfThreads = 4 #Número de CPU cores por job
ComputerPartition = 'batch' #Partição de computadores do grid onde irá correr o job ('batch ou big)
ComputerRAM = 8 #Tamanho da RAM que cada job tem disponível
GridFolderPATH = f'/homes/up201202787/{InstancesFolder}' #PATH da pasta no grid que incorpora as diferentes combinações
BRKGAGenerations = 999 #Number of generations to run the BRKGA algorithm
BRKGAScenarios = 20 #Number of generated scenarios per generation
BRKGASolutions_factor = 8 * 2 #Number of genes per period for the solution multiplied by a factor of 2 (o numero de genes da solucao por periodo * fator do BRKGA - Ver papers literatura)
#ModelVariations = ["0 0 0", "0 1 0", "0 0 1", "0 1 1"] #Variantes do modelo que pretendemos quanto ao seu impacto nos resultados para uma determinada instancia
ModelVariations = ["0 0 0"]

#Nível de risco da falha
#Penalty_multiplier = ["LowRisk","HighRisk"] #Relação de proporcionalidade entre o custo da falha e o custo de substituição para os dois níveis de risco (Custo Falha = Penalty_multiplier * Custo_substituicao)
Penalty_multiplier = ["HighRisk"]

#Distribuicao do RUL
#InstanceFamily = ["Clustered", "Concentrated", "Random"] #Caracterista da distribuição da condição inicial dos ativos
InstanceFamily = ["Clustered"] #Caracterista da distribuição da condição inicial dos ativos

#A primeira coluna diz respeito ao nivel de incerteza (ex: Low Uncertainty) e as restantes dizem respeito ao valor dos periodos considerados (T=5,T=10,T=20)
#UncertaintyLevel = ["LowUnc","HighUnc"] #Valor minimo para a variabilidade da degradação (atualizar se os T mudarem -> ver excel)
UncertaintyLevel = ["HighUnc"]

#Manutencao
#A primeira coluna diz respeito à eficácia da manutenção (Low impact or high impact) e as restantes ao ratio imposto para o tipo de ação de manutenção
#ratio = ["LowImp","HighImp"] #Define o impacto da manutenção nos ativos
ratio = ["LowImp"] #Define o impacto da manutenção nos ativos

#Hiperparâmetros do BRKGA
SolutionProportionMultiplier = [2, 5]
Solution_EliteProportion = [0.10, 0.25]
Solution_MutantProportion = [0.1, 0.3]
#Solution_InheritanceProbability = [0.5, 0.80]
Solution_InheritanceProbability = [0.65]
Scenario_EliteProportion = [0.10, 0.25]
Scenario_MutantProportion = [0.1, 0.3]
Scenario_InheritanceProbability = [0.5, 0.65, 0.80]

#############################################################
###-----Rotina para gerar cada um dos respetivos jobs-----###
#############################################################

#Colocar as variantes dos hiperparametros para as soluções (BRKGA)
Solution_BRKGA_parameters_combination = [f'{SolutionProportionMultiplier[0]} {Solution_EliteProportion[0]} {Solution_MutantProportion[0]} {Solution_InheritanceProbability[0]}']
Scenario_BRKGA_parameters_combination = [f'{Scenario_EliteProportion[0]} {Scenario_MutantProportion[0]} {Scenario_InheritanceProbability[0]}']
for proportion in SolutionProportionMultiplier:
    for elite, scen_elite in zip(Solution_EliteProportion,Scenario_EliteProportion):
        for mutant, scen_mutant in zip(Solution_MutantProportion,Scenario_MutantProportion):
            for inheritance, scen_inheritance in zip(Solution_InheritanceProbability,Scenario_InheritanceProbability):
                Solution_BRKGA_parameters_combination.append(f'{proportion} {elite} {mutant} {inheritance}')

#Colocar as variantes dos hiperparametros para os cenários (BRKGA)
for scen_elite in Scenario_EliteProportion:
    for scen_mutant in Scenario_MutantProportion:
        for scen_inheritance in Scenario_InheritanceProbability:
            Scenario_BRKGA_parameters_combination.append(f'{scen_elite} {scen_mutant} {scen_inheritance}')

# verificacao do diretorio
print(PATHJobs)

# escolher o horizonte para gerar na classe de instancias
for Family in InstanceFamily: #Distribuição do RUL dos ativos
    # Familia de instancia a ser gerada
    print(Family)

    # Counter que permite ajustar a incerteza para um TimeWindow em especifico
    UncertaintyPeriodCount = 0

    #Gerar as instancias face o numero de ativos e periodos de planeamento no horizonte em questão
    for PlanningPeriods in TimeWindow: #Tipos de horizontes de planeamento

        #Atualizar o counter
        UncertaintyPeriodCount +=1

        #Calcular o numero de soluções a gerar no BRKGA
        BRKGASolutions = PlanningPeriods * BRKGASolutions_factor

        #Percorrer as combinações
        for AssetNumber in AssetNumberInstances: #Tipos de portfolio de ativos

            # contador para identificar o numero da instancia
            InstanceGenerationOrder = 0

            for Uncertainty in UncertaintyLevel: #Niveis de incerteza
                for FailureRisk in Penalty_multiplier: #Niveis de risco
                    for Maintenance in ratio: #Niveis de impacto da manutenção
                        for Variation in ModelVariations: #Variantes do modelo que pretendemos testar
                            for sol_parameter in Solution_BRKGA_parameters_combination: #Combinação parâmetros do BRKGA
                                for scen_parameter in Scenario_BRKGA_parameters_combination: #Combinação parâmetros do BRKGA

                                    # contador para identificar o numero da instancia
                                    InstanceGenerationOrder = 0

                                    for contador in range(0,NumeroInstancias):

                                        # atualizar o contador
                                        InstanceGenerationOrder += 1

                                        #Criar o nome do job
                                        InstanceName = f'{Family}_N{AssetNumber}TW{PlanningPeriods}{Uncertainty}{FailureRisk}{Maintenance}_{InstanceGenerationOrder}'

                                        # Abrir job script
                                        Job = open(PATHJobs + "/Job_" + Variation.replace(" ","") + "_" + InstanceName + "_" + sol_parameter.replace(" ","_") + "_" + scen_parameter.replace(" ","_") + ".sh", "w")

                                        #Construir o header do job script
                                        Job.write("#!/bin/bash\n\n#Submit script with: sbatch thefilename\n")

                                        #Especificar o time limit da corrida do job
                                        Job.write(f'#SBATCH --time={TimeLimit}:00:00 # walltime\n')

                                        #Especificar o numero de CPU cores
                                        Job.write(f'#SBATCH --ntasks={NumberOfThreads} # number of processor cores (i.e. tasks)\n')

                                        #Especificar o numero de nós a utilizar no grid (só iremos utilizar em todas as corridas 1 nó para facilitar os testes computacionais)
                                        Job.write("#SBATCH --nodes=1   # number of nodes\n")

                                        #Especificar a partição do grid a utilizar
                                        Job.write(f'#SBATCH -p {ComputerPartition} # partition(s)\n')

                                        #Especificar o tamanho da RAM disponível por job
                                        Job.write(f'#SBATCH --mem-per-cpu={ComputerRAM}G   # memory per CPU core\n')

                                        #Especificar o nome do job a utilizar
                                        Job.write(f'#SBATCH -J "{InstanceName}" # job name\n\n')

                                        #Criar a parameterização do executável
                                        Job.write("# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE\n")
                                        Job.write(f'cd {GridFolderPATH} \n')

                                        #Criar as combinações de corridas que são precisas de ser geradas
                                        Job.write(f'./main {InstanceName} {Variation} {BRKGASolutions} {BRKGAScenarios} {BRKGAGenerations} {Uncertainty} {FailureRisk} {Maintenance} {sol_parameter} {scen_parameter}')

                                        #Sinalizar fim do bash script
                                        Job.write("\n\n# End of bash script")
                                        Job.close()
