#############################################
###Instance Generator for BRKGA algorithm ###
#############################################


############################################################################################
###-----------------------------Parâmetros globais do código-----------------------------###
############################################################################################

#Bibliotecas a ser importadas
import pandas as pd
import numpy as np
from scipy.stats import gamma
import random
import matplotlib.pyplot as plt
import math

#diretorio onde queremos colocar as instancias
PATHInstancia='/Users/LuisDias/Desktop/Doutoramento DEGI/A-Papers LUIS DIAS/3_paper/5 - Resultados/BRKGA_Asset_GRID_Laplace'

#parametros do gerador de instancias
NumeroInstancias = 1 #numero de instancias a gerar por cada classe de instancia (N[X])
AssetNumberInstances=np.array([30]) #Lista do numero de ativos
TimeWindow = np.array([5,10,20]) #Lista de Planning horizons
MaintenanceTypes = 3 #Tipos de manutenção a considerar
AssetMaxHealth = 100 #Condição máxima dos ativos
MaxReplacementCost = 400 #Intervalo superior do custo de substituição
MinReplacementCost = 200 #Intervalo inferior do custo de substituição
RiskFreeRateMinus = 0.00774 #Used rate to calculate the net present value of the defined budget for each period
RiskFreeRatePlus = 0.08 #Used rate to calculate the net present value of the over budget for each period

#Nível de risco da falha
Penalty_multiplier = [["LowRisk",2],["HighRisk",10]] #Relação de proporcionalidade entre o custo da falha e o custo de substituição para os dois níveis de risco (Custo Falha = Penalty_multiplier * Custo_substituicao)

#Distribuicao do RUL
#InstanceFamily = ["Clustered", "Concentrated", "Random"] #Caracterista da distribuição da condição inicial dos ativos
InstanceFamily = ["Clustered"] #Caracterista da distribuição da condição inicial dos ativos
ClusteredAssetsPortion = 0.5 #Indica a proporção de ativos que se encontram no cluster de má condição
MinimumInitialConditionMultiplier = 0.10 #Condição mínima que é gerada para cada ativo (em proporção face a condição máxima)
GoodConditionMultiplier = 0.70 #Condição mínima que é gerada para cada ativo para ser considerado como um ativo em boa condição (em proporção face a condição máxima)
BadConditionMultiplier = 0.30 #Condição mínima que é gerada para cada ativo para ser considerado como um ativo em má condição (em proporção face a condição máxima)
ConditionGapMultiplier = 0.20 #Diferença máxima entre o ativo em melhor e pior condição no portfolio (em proporção face a condição máxima)

#A primeira coluna diz respeito ao nivel de incerteza (ex: Low Uncertainty) e as restantes dizem respeito ao valor dos periodos considerados (T=5,T=10,T=20)
UncertaintyLowerBound = [["LowUnc",4.4,1.6,0.6],["HighUnc",8.7,3.2,1.1]] #Valor minimo para a variabilidade da degradação (atualizar se os T mudarem -> ver excel)
UncertaintyUpperBound = [["LowUnc",8.7,3.2,1.1],["HighUnc",17.5,6.4,2.3]] #Valor máximo para a variabilidade da degradação (atualizar se os T mudarem -> ver excel)
FailuresPerPlanningHorizon = 1 #Número médio mínimo de falhas por cada horizonte de planeamento
SampleSize = 200 #Numero de registos de degradacao a gerar para cada ativo através da distribuição gamma

#Manutencao
#A primeira coluna diz respeito à eficácia da manutenção (Low impact or high impact) e as restantes ao ratio imposto para o tipo de ação de manutenção
ratio = [["LowImp",0.5, 0.35, 0.20],["HighImp",0.9, 0.70, 0.50]] #Define o impacto da manutenção nos ativos
ratioMargin = 0.15 #Margem do ratio para defenir o lower bound do impacto das manutenções
UnitCostImprovementMultiplier = 1.5 #Margem minima para o custo da manutenção por unidade de recuperação da condição do ativo

############################################################################################
###-----Funcoes para gerar a instancia do problema original e do problema aproximado-----###
############################################################################################

# função para gerar a matriz de degradação de cada ativo
def simular_degradacao_linear(AssetNumber,ParametrosGammaDistribution,SampleSize=200):

    # guardar resultados da degradacao
    matriz_degradacao = np.zeros( (AssetNumber, SampleSize) )

    # gamma process para cada ativo
    for i in range(0,AssetNumber):
        for j in range(0,SampleSize):
            matriz_degradacao[i, j] = round(gamma.ppf(random.uniform(0, 1), a = ParametrosGammaDistribution['shape'][i], scale = ParametrosGammaDistribution['scale'][i], loc = 0), 2)

    # Ordenar as linhas da matriz de degradação
    matriz_degradacao.sort(axis=1)


    return matriz_degradacao

# função que permite calcular os parametros da distribuição de gamma face os valores indicados para o nunero de falhas por periodo (F) e a variabilidade da degradação (theta)
def estimar_gamma_parametros(TimeWindow,AssetMaxHealth,FailuresPerPlanningHorizon,Uncertainty,VerifyPlot = True):

    #Calcular o scale parameter da distribuicao gamma
    GammaScaleParameter = (TimeWindow * Uncertainty * Uncertainty) / (FailuresPerPlanningHorizon * AssetMaxHealth)

    #Calcular o shape parameter da distribuicao gamma
    GammaShapeParameter = (AssetMaxHealth * FailuresPerPlanningHorizon) / (TimeWindow * GammaScaleParameter)

    #Gerar plot da distribuição do RUL
    Iterations = 1000
    RUL_ativos = np.zeros(Iterations)

    #Gerar plot da distribuição do RUL caso seja pedido
    if VerifyPlot == True:

        #Verificar a distribuição do RUL para os parâmetros calculados
        for i in range(0,Iterations):

            #Atualizar variaveis auxiliares
            RUL_iteracao = 0
            Condicao_Ativo = AssetMaxHealth

            #Ciclo para calcular o RUL do ativo
            while Condicao_Ativo > 0:

                #Calcular a degradação do ativo e o respetivo RUL
                RUL_iteracao += 1
                Condicao_Ativo -= round(gamma.ppf(random.uniform(0, 1), a=GammaShapeParameter, scale=GammaScaleParameter, loc=0), 2)

            # Guardar o RUL do ativo quando este falha
            RUL_ativos[i] = RUL_iteracao

        #Mostrar o desvio padrão
        print("Desvio padrão = " + str(round(np.std(RUL_ativos),2)))

        # Gerar o Histograma
        n, bins, patches = plt.hist(x=RUL_ativos, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.9)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('RUL')
        plt.ylabel('Frequency')
        plt.title(r'RUL distribution $(k=' + str(GammaShapeParameter) + ', beta=' + str(GammaScaleParameter) + ')$')
        maxfreq = n.max()

        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

        #show plot
        plt.show()

    return GammaShapeParameter, GammaScaleParameter

#funcao para estimar os parametros mais adequados para combinação do horizonte planeamento face a variabilidade desejada
def GenerateUncertaintyMatrix(TimeWindowIncrement,UncertaintyIncrement,AssetMaxHealth,FailuresPerPlanningHorizon):


    #Matriz com os resultados desejados
    MatrizParametros = np.zeros((20,4))

    # Iniciar o calculo da matriz dos parametros
    Uncertainty = 0

    for u in range(0,MatrizParametros.shape[0]):

        #Atualizar parametro da variabilidade
        Uncertainty += UncertaintyIncrement

        #Estado da iteracao
        print("Uncertainty=" + str(Uncertainty))

        #Fazer reset da variavel TimeWindow
        TimeWindow = 0

        for t in range(0, MatrizParametros.shape[1]):

            #Atualizar parametro do horizonte de planeamento
            TimeWindow += TimeWindowIncrement

            # Estado da iteracao
            print("TimeWindow=" + str(TimeWindow))

            # Calcular o scale parameter da distribuicao gamma
            GammaScaleParameter = (TimeWindow * Uncertainty * Uncertainty) / (FailuresPerPlanningHorizon * AssetMaxHealth)

            # Calcular o shape parameter da distribuicao gamma
            GammaShapeParameter = (AssetMaxHealth * FailuresPerPlanningHorizon) / (TimeWindow * GammaScaleParameter)

            # Gerar plot da distribuição do RUL
            Iterations = 1000
            RUL_ativos = np.zeros(Iterations)

            # Verificar a distribuição do RUL para os parâmetros calculados
            for i in range(0, Iterations):

                # Atualizar variaveis auxiliares
                RUL_iteracao = 0
                Condicao_Ativo = AssetMaxHealth

                # Ciclo para calcular o RUL do ativo
                while Condicao_Ativo > 0:
                    # Calcular a degradação do ativo e o respetivo RUL
                    RUL_iteracao += 1
                    Condicao_Ativo -= round(gamma.ppf(random.uniform(0, 1), a=GammaShapeParameter, scale=GammaScaleParameter, loc=0), 2)

                # Guardar o RUL do ativo quando este falha
                RUL_ativos[i] = RUL_iteracao

            #Calcular desvio padrao da distribuição do RUL
            MatrizParametros[u,t] = round(np.std(RUL_ativos),2)

    return MatrizParametros

# função que permite criar o ficheiro de texto
def criar_instancia_original_problem(InstancePath, InstanceID, AssetNumber, TimeWindow, Uncertainty, FailureRisk, Maintenance, MaintenanceTypes, RiskFreeRateMinus, RiskFreeRatePlus, SampleSize, InitialHealth, DegradationMatrix, CostReplacingAsset, CostFailure, CostAction, MaintenanceEffect, AssetAssetMaxHealth):

    # Abrir instancias
    Instance = open(InstancePath + "/" + "Instance_N" + str(AssetNumber) + "TW" + str(TimeWindow) + Uncertainty + FailureRisk + Maintenance + "_" + str(InstanceID) + ".txt", "w")

    # AssetNumber
    Instance.write("Asset Number\n" + str(AssetNumber) + "\n\n")

    # TimeWindow
    Instance.write("Time Window\n" + str(TimeWindow) + "\n\n")

    # MaintenanceTypes
    Instance.write("Maintenance Types\n" + str(MaintenanceTypes) + "\n\n")

    # RiskFreeRateMinus
    Instance.write("Risk Free Rate Minus\n" + str(RiskFreeRateMinus) + "\n\n")

    # RiskFreeRatePlus
    Instance.write("Risk Rate Plus\n" + str(RiskFreeRatePlus) + "\n\n")

    # RiskFreeRatePlus
    Instance.write("Sample Size\n" + str(SampleSize) + "\n\n")

    # InitialHealth
    Instance.write("Initial Condition\n")

    for i in range(0,AssetNumber):
        if (i < AssetNumber - 1):
            Instance.write(str(InitialHealth[i]) + " ")
        else:
            Instance.write(str(InitialHealth[i]) + "\n\n")



    # Degradation Matrix
    Instance.write("Degradation Matrix\n")

    for i in range(0,AssetNumber):
        for j in range(0,DegradationMatrix.shape[1]):
            if j < DegradationMatrix.shape[1] - 1:
                Instance.write(str(DegradationMatrix[i, j]) + " ")
            else:
                Instance.write(str(DegradationMatrix[i, j]) + "\n")
        if i == AssetNumber - 1:
            Instance.write("\n")

    # CostReplacingAsset
    Instance.write("Replacement Cost\n")
    for i in range(0,AssetNumber):
        if (i < AssetNumber - 1):
            Instance.write(str(CostReplacingAsset[i]) + " ")
        else:
            Instance.write(str(CostReplacingAsset[i]) + "\n\n")

    # CostFailure
    Instance.write("Failure Cost\n")
    for i in  range(0,AssetNumber):
        if i < AssetNumber - 1:
            Instance.write(str(CostFailure[i]) + " ")
        else:
            Instance.write(str(CostFailure[i]) + "\n\n")

    # CostAction
    Instance.write("Maintenance Action\n")
    for a in range(0,MaintenanceTypes):
        for i in range(0,AssetNumber):
            if i < AssetNumber - 1:
                Instance.write(str(CostAction[i, a]) + " ")
            else:
                Instance.write(str(CostAction[i, a]) + "\n")
        if a == MaintenanceTypes - 1:
            Instance.write("\n")

    # MaintenanceEffect
    Instance.write("Maintenance Action Effect\n")
    for a in range(0,MaintenanceTypes):
        for i in range(0,AssetNumber):
            if i < AssetNumber - 1:
                Instance.write(str(MaintenanceEffect[i, a]) + " ")
            else:
                Instance.write(str(MaintenanceEffect[i, a]) + "\n")

        if a == MaintenanceTypes - 1:
            Instance.write("\n")

    # Asset Max Health
    Instance.write("Asset Max Health\n" + str(AssetAssetMaxHealth) + "\n\n")

    # Max Replacement Cost
    Instance.write("Max Replacement Cost\n" + str(max(CostReplacingAsset)) + "\n\n")

    # Min Replacement Cost
    Instance.write("Min Replacement Cost\n" + str(min(CostReplacingAsset)) + "\n\n")

    # Max Maintenance Action Cost
    Instance.write("Max Maintenance Action Cost\n")

    #Vetor com os custos maximos e minimos para cada tipo de ação de manutenção
    CostActionMax = CostAction.max(0)
    CostActionMin = CostAction.min(0)

    for i in range(0,MaintenanceTypes):
        if i < MaintenanceTypes - 1:
            Instance.write(str(CostActionMax[i]) + " ")
        else:
            Instance.write(str(CostActionMax[i]) + "\n\n")

    # Min Maintenance Action Cost
    Instance.write("Min Maintenance Action Cost\n")

    for i in range(0,MaintenanceTypes):
        if i < MaintenanceTypes - 1:
            Instance.write(str(CostActionMin[i]) + " ")
        else:
            Instance.write(str(CostActionMin[i]))

    Instance.close()

# função para validar as instancias geradas
def verificador_de_instancias(AssetNumber, PlanningPeriods, InitialHealth, Family, CostReplacingAsset, CostAction, AssetDegradation, MaintenanceEffect):

    #Por defeito a resposta será afirmativa de forma a que a instância seja válida face os requesitos definidos
    InstanceValid = True

    #Verificar se existem condição de ativos negativas ou superiores ao máximo definido
    for i in range(0, AssetNumber):
        if InitialHealth[i] < MinimumInitialConditionMultiplier * AssetMaxHealth:
            print("Asset" + str(i) + " has an initial condition bellow the one defined in the global parameters of " + str(MinimumInitialConditionMultiplier * AssetMaxHealth))

            #Sempre que um critério de verificação for verificado a instancia é negada
            InstanceValid = False

        if InitialHealth[i] > AssetMaxHealth:
            print("Asset" + str(i) + " has an initial condition above the defined best condition in the global parameters of " + str(AssetMaxHealth))

            #Sempre que um critério de verificação for verificado a instancia é negada
            InstanceValid = False


    #Verificar se a condição dos ativos corresponde ao tipo de instância que foi definido
    if Family == "Clustered":
        for i in range(0, AssetNumber):
            AssetNumberGoodCondition = AssetNumber * ClusteredAssetsPortion
            if i < AssetNumberGoodCondition and InitialHealth[i] < GoodConditionMultiplier * AssetMaxHealth:
                print("Asset" +  str(i) + " does not comply with criteria of an asset in good condition defined for the Instance Family Clustered")

                # Sempre que um critério de verificação for verificado a instancia é negada
                InstanceValid = False

            elif i >= AssetNumberGoodCondition and InitialHealth[i] > BadConditionMultiplier * AssetMaxHealth:
                print("Asset" + str(i) + " does not comply with criteria of an asset in bad condition defined for the Instance Family Clustered")

                #Sempre que um critério de verificação for verificado a instancia é negada
                InstanceValid = False

    elif Family == "Concentrated":

        #Calcular o gap entre o ativo em melhor condição face ao ativo em pior condição
        AssetConditionGap = np.max(InitialHealth) - np.min(InitialHealth)

        #Quando o condition Gap está acima a instance family concentrated tem que ser descartada
        if AssetConditionGap > ConditionGapMultiplier * AssetMaxHealth:
            print("The Instance Family Concentrated does not comply with the defined Condition Gap of " + str(ConditionGapMultiplier * AssetMaxHealth))

            #Sempre que um critério de verificação for verificado a instancia é negada
            InstanceValid = False

    #Verificar se o pressuposto do numero de falhas por horizonte de planeamento é cumprido
    for i in range(0, AssetNumber):

        # compute the asset remaining useful life (RUL)
        BestAssetRUL = AssetMaxHealth / AssetDegradation[i]
        ComputedFailuresPerPlanningHorizon = PlanningPeriods / BestAssetRUL

        #verify the number of failure per planning horizon
        if ComputedFailuresPerPlanningHorizon > FailuresPerPlanningHorizon * 1.15: #tem que se dar uma margem pois existe casos que é suposto ser um bocadinho maior
            print("Asset" + str(i) + " has an average of failures above the defined parameter of FailuresPerPlanningHorizon= " + str(FailuresPerPlanningHorizon))

            #Sempre que um critério de verificação for verificado a instancia é negada
            InstanceValid = False

    #Verificar se os limites dos custos de substituição são respeitados
    for i in range(0, AssetNumber):

        if CostReplacingAsset[i] > MaxReplacementCost:
            print("Asset" + str(i) + "has a planned replacement cost above the specified maximum value")

        if CostReplacingAsset[i] < MinReplacementCost:
            print("Asset" + str(i) + "has a planned replacement cost bellow the specified minimum value")

    #Verificar se os limites dos custos de substituição por falha são respeitados
    for i in range(0, AssetNumber):

        if CostFailure[i] != CostReplacingAsset[i] * Penalty_multiplier[0][1] and CostFailure[i] != CostReplacingAsset[i] * Penalty_multiplier[1][1]:
            print("Asset" + str(i) + "has a unplanned replacement cost which does not respects the imposed proportionality in the Penalty_multiplier variable")

    #Verificar o custo beneficio das ações de manutenção face as ações de substituição
    for i in range(0, AssetNumber):
        for j in range(0, MaintenanceTypes):

            MaintenanceActionUnitaryCost = round((AssetDegradation[i] * MaintenanceEffect[i, j]) / CostAction[i, j], 4)
            ReplacementUnitaryCost = round(AssetMaxHealth / CostReplacingAsset[i], 4)
            WorstMaintenanceActionUnitaryCost = round(AssetMaxHealth / (CostReplacingAsset[i] * UnitCostImprovementMultiplier), 4)

            if MaintenanceActionUnitaryCost >  ReplacementUnitaryCost:
                print("Asset" + str(i) + " has a unitary Maintenance cost of " + str(MaintenanceActionUnitaryCost) + " that is better than the unitary cost of an asset replacement " + str(ReplacementUnitaryCost))

                # Sempre que um critério de verificação for verificado a instancia é negada
                InstanceValid = False

            if MaintenanceActionUnitaryCost <  WorstMaintenanceActionUnitaryCost:
                print("Asset" + str(i) + " has a unitary Maintenance cost of " + str(MaintenanceActionUnitaryCost) + " that is above the specified maximum unitary cost of an asset replacement " + str(WorstMaintenanceActionUnitaryCost))

                # Sempre que um critério de verificação for verificado a instancia é negada
                InstanceValid = False

    #Verificar se a degradação média gerada respeita as caracteristicas impostas
    for i in range(0, AssetNumber):
        if round(PlanningPeriods / (AssetMaxHealth / AssetDegradation[i]),0) >  FailuresPerPlanningHorizon:
            print("Asset" + str(i) + "has an average number of failure above the defined value of " + str(FailuresPerPlanningHorizon))

    #Falta uma função que verifique se o nível de incerteza imposto na degradação é respeitado!!!!!!!!!

    #Verificar se o impacto da manutenção não faz degradar ainda mais o ativo ou se melhora a condição
    for i in range(0, AssetNumber):
        for j in range(0, MaintenanceTypes):
            if  MaintenanceEffect[i, j] < 0:
                print("It is not possible to have a negative maintenance effect value")
            if MaintenanceEffect[i, j] > 1:
                print("It is not possible to have a maintenance effect above the value of 1. If it does the asset condition improvement will be superior to its degradation.")

    return InstanceValid

#########################################################
###-----Rotina para gerar cada uma das instancias-----###
########################################################

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
            for Uncertainty in ["LowUnc","HighUnc"]: #Niveis de incerteza
                for FailureRisk in ["LowRisk","HighRisk"]: #Niveis de risco
                    for Maintenance in ["LowImp", "HighImp"]: #Niveis de impacto da manutenção

                        # contador para identificar o numero da instancia
                        InstanceGenerationOrder = 0

                        for contador in range(0,NumeroInstancias):

                            #Obter instancia que cumpre os requesitos
                            VerificarInstancia = False


                            # atualizar o contador
                            InstanceGenerationOrder += 1

                            # verificacao iteracao atual do ciclo
                            print("Iteration " + str(InstanceGenerationOrder))

                            # Condicao inicial dos ativos
                            InitialHealth =np.zeros(AssetNumber)

                            #No caso do "Random" apenas temos em atenção a condição mínima do ativo
                            if Family == "Random":
                                for i in range(0,AssetNumber):
                                    InitialHealth[i] = round(random.uniform(MinimumInitialConditionMultiplier * AssetMaxHealth,AssetMaxHealth))

                            # No caso do "Clustered" temos em atenção que metade dos ativos têm que estar em boa condição e a restante metade em má condição
                            elif Family == "Clustered":
                                for i in range(0, AssetNumber):
                                    AssetNumberGoodCondition = round(ClusteredAssetsPortion * AssetNumber, 0)
                                    if i < AssetNumberGoodCondition:
                                        InitialHealth[i] = round(random.uniform(GoodConditionMultiplier * AssetMaxHealth, AssetMaxHealth))
                                    else:
                                        InitialHealth[i] = round(random.uniform(MinimumInitialConditionMultiplier * AssetMaxHealth, BadConditionMultiplier * AssetMaxHealth))

                            #No caso do "Concentrated" temos de ter em atenção à distância entre o ativo em melhor e pior condição no portfolio
                            elif Family == "Concentrated":
                                #Gerar a condição inicial para o primeiro ativo da iteração (valor de referência)
                                InitialHealth[0] = round(random.uniform(MinimumInitialConditionMultiplier * AssetMaxHealth,AssetMaxHealth))

                                #Calcular os limites da distribuição uniforme em conformidade
                                if InitialHealth[0] + ConditionGapMultiplier * AssetMaxHealth <= AssetMaxHealth:
                                    #Neste caso a condição inicial que foi gerada será a pior
                                    UpperBound = InitialHealth[0] + ConditionGapMultiplier * AssetMaxHealth
                                    LowerBound = InitialHealth[0]
                                else:
                                    # Neste caso a condição inicial que foi gerada será a melhor
                                    UpperBound = InitialHealth[0]
                                    LowerBound = InitialHealth[0] - ConditionGapMultiplier * AssetMaxHealth

                                #Gerar a condição inicial para os restantes ativos (excluimos o primeiro ativo)
                                for i in range(1, AssetNumber):
                                    InitialHealth[i] = int(random.uniform(LowerBound, UpperBound))

                            # Gerar degradacoes média e do cenario
                            ParametrosGammaDistribution = pd.DataFrame(np.zeros( (AssetNumber, 2) ), columns=['shape', 'scale'])
                            AssetDegradation = np.zeros(AssetNumber)

                            #Para todos os ativos é importante parametrizar a distribuição gamma
                            for i in range(0,AssetNumber):

                                #Parametrizar a distribuiçã gamma de acordo com os parâmetros especificados
                                if Uncertainty == "LowUnc":
                                    UncertaintyLevel = round(random.uniform(UncertaintyLowerBound[0][UncertaintyPeriodCount],UncertaintyUpperBound[0][UncertaintyPeriodCount]),2)
                                elif Uncertainty == "HighUnc":
                                    UncertaintyLevel = round(random.uniform(UncertaintyLowerBound[1][UncertaintyPeriodCount],UncertaintyUpperBound[1][UncertaintyPeriodCount]),2)

                                # shape and scale parameter da gaussian distribution
                                ParametrosGammaDistribution['shape'][i], ParametrosGammaDistribution['scale'][i] = estimar_gamma_parametros(PlanningPeriods,AssetMaxHealth,
                                                                                                                                            FailuresPerPlanningHorizon,UncertaintyLevel,False)

                                # Dado que estamos a utilizar a distribuição gamma basta multiplicar os parametros para obter a degração média
                                AssetDegradation[i] = ParametrosGammaDistribution['shape'][i] * ParametrosGammaDistribution['scale'][i]

                            # Degradacao do periodo
                            DegradationMatrix = simular_degradacao_linear(AssetNumber, ParametrosGammaDistribution, SampleSize)

                            # gerar os custos para as ações de substituição devido a falha
                            CostReplacingAsset = np.zeros(AssetNumber)
                            for i in range(0, AssetNumber):
                                CostReplacingAsset[i] = int(random.uniform(MinReplacementCost, MaxReplacementCost))

                            # gerar os custos de falha
                            CostFailure = np.zeros(AssetNumber)
                            for i in range(0,AssetNumber):
                                if FailureRisk == "LowRisk":
                                    CostFailure[i] = int(Penalty_multiplier[0][1] * CostReplacingAsset[i])
                                elif FailureRisk == "HighRisk":
                                    CostFailure[i] = int(Penalty_multiplier[0][1] * CostReplacingAsset[i])

                            # atencao que o beneficio das manutencoes sao calculados atraves do valor esperado da degradacao dos ativos
                            MaintenanceEffect = np.zeros((AssetNumber,MaintenanceTypes))
                            for i in range(0,AssetNumber):
                                for j in range(0,MaintenanceTypes):
                                    if Maintenance == "LowImp":
                                        MaintenanceEffect[i, j] = round(random.uniform(ratio[0][j+1] - ratioMargin, ratio[0][j+1]), 2)
                                    elif Maintenance == "HighImp":
                                        MaintenanceEffect[i, j] = round(random.uniform(ratio[1][j+1] - ratioMargin, ratio[1][j+1]), 2)


                            # gerar os custos para as ações de manutenção
                            CostAction = np.zeros((AssetNumber,MaintenanceTypes))
                            CostAction_verification = np.zeros((AssetNumber, MaintenanceTypes)) #devido aos arredondamentos esta variável serve para apenas e unica exclusivamente para validar a instância e não constitui o falor final da mesma
                            for i in range(0,AssetNumber):
                                for j in range(0,MaintenanceTypes):

                                    # e preciso calcular o ratio de custo por unidade de forma a evitar que seja mais benefica a manutencao do que a substituicao de forma sistematica
                                    GeneratedUnitaryCost = round(random.uniform(AssetMaxHealth / (CostReplacingAsset[i] * UnitCostImprovementMultiplier), AssetMaxHealth / CostReplacingAsset[i]), 4)

                                    #Calcular o custo da manutenção face o custo por unidade gerado
                                    CostAction_verification[i, j] = round(MaintenanceEffect[i, j] * AssetDegradation[i] / GeneratedUnitaryCost,4)
                                    CostAction[i, j] = int(MaintenanceEffect[i, j] * AssetDegradation[i] / GeneratedUnitaryCost)


                            #Verificar os parâmetros da instância que foram criados
                            VerificarInstancia = verificador_de_instancias(AssetNumber, PlanningPeriods, InitialHealth, Family, CostReplacingAsset, CostAction_verification, AssetDegradation, MaintenanceEffect)

                            #A instancia é guardade sempre que não for encontrado nenhum problema com os parâmetros que foram gerados
                            if VerificarInstancia == True:

                                # Atualizar o path da instancia para a pasta respetiva para qual se está a gerar a instancia
                                FinalPATHInstancia = PATHInstancia + "/" + Family + "_" + Uncertainty + FailureRisk + Maintenance + "/data"

                                # Mostrar o path completo para a instancia gerada
                                print("Folder path = " + FinalPATHInstancia)

                                # Create instance para o problema original
                                criar_instancia_original_problem(FinalPATHInstancia, InstanceGenerationOrder, AssetNumber, PlanningPeriods, Uncertainty, FailureRisk, Maintenance, MaintenanceTypes, RiskFreeRateMinus, RiskFreeRatePlus, SampleSize,
                                                                 InitialHealth, DegradationMatrix, CostReplacingAsset, CostFailure, CostAction, MaintenanceEffect, AssetMaxHealth)