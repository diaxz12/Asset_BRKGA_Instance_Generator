#####################################################################
###Jobs Generator for BRKGA algorithm runs on FEUP Computing grid ###
#####################################################################


############################################################################################
###-----------------------------Parâmetros globais do código-----------------------------###
############################################################################################

#Bibliotecas a ser importadas
import numpy as np

#diretorio onde queremos colocar as instancias
PATHJobs='/Users/LuisDias/Desktop/Doutoramento DEGI/A-Papers LUIS DIAS/3_paper/5 - Resultados/BRKGA_Asset_GRID_Laplace/Job_scripts'

#parametros do gerador de jobs
NumeroInstancias = 1 #numero de instancias a gerar por cada classe de instancia (N[X])
AssetNumberInstances=np.array([30]) #Lista do numero de ativos
TimeWindow = np.array([5,10,20]) #Lista de Planning horizons
TimeLimit = 72 #Tempo limite que o job pode ser executado no grid (em Horas)
NumberOfThreads = 1 #Número de CPU cores por job
ComputerPartition = 'batch' #Partição de computadores do grid onde irá correr o job ('batch ou big)
ComputerRAM = 8 #Tamanho da RAM que cada job tem disponível
GridFolderPATH = '/homes/up201202787/BRKGA_Asset_GRID_Laplace' #PATH da pasta no grid que incorpora as diferentes combinações
BRKGAGenerations = 5000 #Number of generations to run the BRKGA algorithm
BRKGAScenarios = 50 #Number of generated scenarios per generation
BRKGASolutions = 50 #Number of generated solutions per generation

#Nível de risco da falha
Penalty_multiplier = ["LowRisk","HighRisk"] #Relação de proporcionalidade entre o custo da falha e o custo de substituição para os dois níveis de risco (Custo Falha = Penalty_multiplier * Custo_substituicao)

#Distribuicao do RUL
#InstanceFamily = ["Clustered", "Concentrated", "Random"] #Caracterista da distribuição da condição inicial dos ativos
InstanceFamily = ["Clustered"] #Caracterista da distribuição da condição inicial dos ativos

#A primeira coluna diz respeito ao nivel de incerteza (ex: Low Uncertainty) e as restantes dizem respeito ao valor dos periodos considerados (T=5,T=10,T=20)
UncertaintyLevel = ["LowUnc","HighUnc"] #Valor minimo para a variabilidade da degradação (atualizar se os T mudarem -> ver excel)

#Manutencao
#A primeira coluna diz respeito à eficácia da manutenção (Low impact or high impact) e as restantes ao ratio imposto para o tipo de ação de manutenção
ratio = ["LowImp","HighImp"] #Define o impacto da manutenção nos ativos


#############################################################
###-----Rotina para gerar cada um dos respetivos jobs-----###
#############################################################

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

        #Percorrer as combinações
        for AssetNumber in AssetNumberInstances: #Tipos de portfolio de ativos

            # contador para identificar o numero da instancia
            InstanceGenerationOrder = 0

            for Uncertainty in UncertaintyLevel: #Niveis de incerteza
                for FailureRisk in Penalty_multiplier: #Niveis de risco
                    for Maintenance in ratio: #Niveis de impacto da manutenção

                        # contador para identificar o numero da instancia
                        InstanceGenerationOrder = 0

                        for contador in range(0,NumeroInstancias):

                            # atualizar o contador
                            InstanceGenerationOrder += 1

                            #Criar o nome do job
                            InstanceName = Family + "_N" + str(AssetNumber) + "TW" + str(PlanningPeriods) + Uncertainty + FailureRisk + Maintenance + "_" + str(InstanceGenerationOrder)

                            # Abrir job script
                            Job = open(PATHJobs + "/Job_" + InstanceName + ".sh", "w")

                            #Construir o header do job script
                            Job.write("#!/bin/bash\n\n#Submit script with: sbatch thefilename\n")

                            #Especificar o time limit da corrida do job
                            Job.write("#SBATCH --time=" + str(TimeLimit) + ":00:00   # walltime\n")

                            #Especificar o numero de CPU cores
                            Job.write("#SBATCH --ntasks=" + str(NumberOfThreads) + "   # number of processor cores (i.e. tasks)\n")

                            #Especificar o numero de nós a utilizar no grid (só iremos utilizar em todas as corridas 1 nó para facilitar os testes computacionais)
                            Job.write("#SBATCH --nodes=1   # number of nodes\n")

                            #Especificar a partição do grid a utilizar
                            Job.write("#SBATCH -p " + ComputerPartition + "   # partition(s)\n")

                            #Especificar o tamanho da RAM disponível por job
                            Job.write("#SBATCH --mem-per-cpu=" + str(ComputerRAM) + "G   # memory per CPU core\n")

                            #Especificar o nome do job a utilizar
                            Job.write("#SBATCH -J '" + InstanceName + "'   # job name\n\n")

                            #Criar a parameterização do executável
                            Job.write("# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE\n")
                            Job.write("cd " + GridFolderPATH + "\n")

                            #Criar as combinações de corridas que são precisas de ser geradas
                            Job.write("./main " + InstanceName + " 0 0 0 " + str(BRKGASolutions) + " " + str(BRKGAScenarios) + " "
                                      + str(BRKGAGenerations) + " " + Uncertainty + " " + FailureRisk + " " + Maintenance + "\n")

                            Job.write("./main " + InstanceName + " 1 0 0 " + str(BRKGASolutions) + " " + str(BRKGAScenarios) + " "
                                      + str(BRKGAGenerations) + " " + Uncertainty + " " + FailureRisk + " " + Maintenance + "\n")

                            Job.write("./main " + InstanceName + " 0 1 0 " + str(BRKGASolutions) + " " + str(BRKGAScenarios) + " "
                                      + str(BRKGAGenerations) + " " + Uncertainty + " " + FailureRisk + " " + Maintenance + "\n")

                            Job.write("./main " + InstanceName + " 0 0 1 " + str(BRKGASolutions) + " " + str(BRKGAScenarios) + " "
                                      + str(BRKGAGenerations) + " " + Uncertainty + " " + FailureRisk + " " + Maintenance + "\n")

                            Job.write("./main " + InstanceName + " 1 1 1 " + str(BRKGASolutions) + " " + str(BRKGAScenarios) + " "
                                      + str(BRKGAGenerations) + " " + Uncertainty + " " + FailureRisk + " " + Maintenance + "\n")

                            #Sinalizar fim do bash script
                            Job.write("# End of bash script")
