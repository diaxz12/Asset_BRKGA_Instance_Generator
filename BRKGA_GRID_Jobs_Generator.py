#####################################################################
###Jobs Generator for BRKGA algorithm runs on FEUP Computing grid ###
#####################################################################


############################################################################################
###-----------------------------Parâmetros globais do código-----------------------------###
############################################################################################

#Bibliotecas a ser importadas
import pandas as pd
import numpy as np

#diretorio onde queremos colocar as instancias
PATHInstancia='/Users/LuisDias/Desktop/Doutoramento DEGI/A-Papers LUIS DIAS/3_paper/5 - Resultados/Computational_Runs_Asset_BRKGA'

#parametros do gerador de instancias
NumeroInstancias = 3 #numero de instancias a gerar por cada classe de instancia (N[X])
AssetNumberInstances=np.array([30]) #Lista do numero de ativos
TimeWindow = np.array([5,10,20]) #Lista de Planning horizons

#Nível de risco da falha
Penalty_multiplier = ["LowRisk","HighRisk"] #Relação de proporcionalidade entre o custo da falha e o custo de substituição para os dois níveis de risco (Custo Falha = Penalty_multiplier * Custo_substituicao)

#Distribuicao do RUL
#InstanceFamily = ["Clustered", "Concentrated", "Random"] #Caracterista da distribuição da condição inicial dos ativos
InstanceFamily = ["Clustered"] #Caracterista da distribuição da condição inicial dos ativos

#A primeira coluna diz respeito ao nivel de incerteza (ex: Low Uncertainty) e as restantes dizem respeito ao valor dos periodos considerados (T=5,T=10,T=20)
Uncertainty = ["LowUnc","HighUnc"] #Valor minimo para a variabilidade da degradação (atualizar se os T mudarem -> ver excel)

#Manutencao
#A primeira coluna diz respeito à eficácia da manutenção (Low impact or high impact) e as restantes ao ratio imposto para o tipo de ação de manutenção
ratio = ["LowImp","HighImp"] #Define o impacto da manutenção nos ativos


#############################################################
###-----Rotina para gerar cada um dos respetivos jobs-----###
#############################################################

# verificacao do diretorio
print(PATHInstancia)

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

            for Uncertainty in Uncertainty: #Niveis de incerteza
                for FailureRisk in Penalty_multiplier: #Niveis de risco
                    for Maintenance in ratio: #Niveis de impacto da manutenção
                        for contador in range(0,NumeroInstancias):

                            # Abrir job script
                            Job = open(PATHInstancia + "/" + "Job_N" + str(AssetNumber) + "TW" + str(TimeWindow) + Uncertainty + FailureRisk + Maintenance + "_" + str(InstanceGenerationOrder) + ".sh", "w")

                            #Construir job script (fiquei por aqui)