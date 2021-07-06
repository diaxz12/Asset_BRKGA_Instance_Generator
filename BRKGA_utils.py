#########################################
###Funções genéricas para o BRKGA     ###
#########################################


############################################################################################
###-----------------------------Parâmetros globais do código-----------------------------###
############################################################################################

#Bibliotecas a ser importadas
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

#Nome da pasta que guarda os resultados
ResultsFolder = 'Tuning_resultados_stopping_criterion_completo'

#diretorio onde queremos ler os resultados
PATHInstancia = f'/Users/LuisDias/Desktop/{ResultsFolder}/BRKGA_cplex/'
PATH_benchmark = f'/Users/LuisDias/Desktop/{ResultsFolder}/benchmark/'

#Diretorio onde queremos guardar os resultados
#PATH_results = f'/Users/LuisDias/Desktop/Doutoramento DEGI/A-Papers LUIS DIAS/3_paper/5 - Resultados/{ResultsFolder}/resultados/'
PATH_results = f'/Users/LuisDias/Desktop/{ResultsFolder}/'

#Caracterista da distribuição da condição inicial dos ativos
InstanceFamily = ["Clustered", "Concentrated", "Random"]

#Caracteristicas das instancias a analisar nos resultados
CharacteristicList = ['Clustered','Concentrated','Random','N20', 'N100','TW5','TW10','HighUnc','LowUnc','HighRisk','LowRisk','HighImp','LowImp']
ColumnNames = ['Instance_type','Instance_type','Instance_type','Portfolio_size','Portfolio_size','Planning_horizon','Planning_horizon'
    ,'Uncertainty_level','Uncertainty_level','Risk_level','Risk_level','Maintenance_level','Maintenance_level']

#Caracteristicas da parameterizacao (soluções e cenários)
Solution_Parameters_CharacteristicList = ['A1','A2','A5','Pe10','Pe17','Pe25','Pm10','Pm20','Pm30','rh50','rh65','rh80']
Solution_Parameters_ColumnNames = ['Solution_size_factor','Solution_size_factor','Solution_size_factor',
                                   'Solution_Elite_proportion','Solution_Elite_proportion','Solution_Elite_proportion',
                                   'Solution_Mutant_proportion','Solution_Mutant_proportion','Solution_Mutant_proportion',
                                   'Solution_Inheritance_probability','Solution_Inheritance_probability','Solution_Inheritance_probability']

Scenario_Parameters_CharacteristicList = ['Pe10','Pe17','Pe25','Pm10','Pm20','Pm30','rh50','rh65','rh80']
Scenario_Parameters_ColumnNames = ['Scenario_Elite_proportion','Scenario_Elite_proportion','Scenario_Elite_proportion',
                                   'Scenario_Mutant_proportion','Scenario_Mutant_proportion','Scenario_Mutant_proportion',
                                   'Scenario_Inheritance_probability','Scenario_Inheritance_probability','Scenario_Inheritance_probability']

#Lista de ficheiros na pasta BRKGA_cplex dos resultados
File_list = os.listdir(PATHInstancia)

##################################################
###-----Funcoes para estudar os resultados-----###
##################################################

#Funcao para extrair as caracteristicas das instancias
def get_instance_characteristics(data, CharacteristicList, ColumnNames, ReferenceColname):

    #Create characteristic if it does not exist
    for column in np.unique(ColumnNames):
        if column not in data:
            data[column] = -1

    #Get characteristic
    for characteristic, column in zip(CharacteristicList,ColumnNames):
        data[column][data[ReferenceColname].str.contains(characteristic)] = characteristic

    return data

#Função que permite avaliar a dispersão dos resultados
def calculate_dispersion(PATHInstancia, PATH_scenario_diversity_filename, FamilyType, generation, plot_results = False):

    #Ler os dados
    scenario_diversity_data = pd.read_csv(PATHInstancia + PATH_scenario_diversity_filename, sep='\t')

    #Filtrar o tipo de instancia
    scenario_diversity_data = scenario_diversity_data[scenario_diversity_data['Instance'].str.contains(FamilyType)]

    #Converter as colunas numéricas para o formato correto
    scenario_diversity_data['Generation'] = pd.to_numeric(scenario_diversity_data['Generation'], errors='coerce')
    scenario_diversity_data['Best_Solution_Value'] = pd.to_numeric(scenario_diversity_data['Best_Solution_Value'], errors='coerce')
    scenario_diversity_data['Worst_Solution_Value'] = pd.to_numeric(scenario_diversity_data['Worst_Solution_Value'], errors='coerce')

    #Dados da primeira geração
    scenario_diversity_data = scenario_diversity_data.loc[(scenario_diversity_data.Generation == generation)].reset_index(drop=True)
    NumberScenarios = scenario_diversity_data.shape[0]
    GridSize = int(NumberScenarios * 0.8)

    #Calcular dispersão ideal para a primeira geração
    delta_x = (scenario_diversity_data['Best_Solution_Value'].max() - scenario_diversity_data['Best_Solution_Value'].min())/GridSize#delta ideal no eixo dos x
    delta_y = (scenario_diversity_data['Worst_Solution_Value'].max() - scenario_diversity_data['Worst_Solution_Value'].min())/GridSize#delta ideal no eixo dos y

    #Criar a grelha que irá permitir verificar a localização dos pontos
    x_axis_grid_coordinates = np.arange(scenario_diversity_data['Best_Solution_Value'].min()-delta_x,scenario_diversity_data['Best_Solution_Value'].max()+delta_x,delta_x)
    y_axis_grid_coordinates = np.arange(scenario_diversity_data['Worst_Solution_Value'].min()-delta_y,scenario_diversity_data['Worst_Solution_Value'].max()+delta_y,delta_y)

    #Calcular a dispersão na grelha
    result_grid = np.zeros(GridSize**2)
    counter = 0
    for y in range(1,GridSize):
        for x in range(1,GridSize):
            for v in range(0,NumberScenarios):
                if (scenario_diversity_data['Best_Solution_Value'][v] >= x_axis_grid_coordinates[x-1]) & (scenario_diversity_data['Best_Solution_Value'][v] < x_axis_grid_coordinates[x]):
                    if (scenario_diversity_data['Worst_Solution_Value'][v] >= y_axis_grid_coordinates[y-1]) & (scenario_diversity_data['Worst_Solution_Value'][v] < y_axis_grid_coordinates[y]):
                        result_grid[counter] = result_grid[counter] + 1
            #atualizar o counter
            counter += 1

    #Calcular o número de pontos que estão dentro de circunferências (basicamente têm uma distância euclideana ao ponto mais próximo inferior à diagonal do quadrado da grelha)
    distance_limit = np.sqrt(delta_x**2+delta_y**2)
    number_of_conquered_circles = 0
    for point in range(0,NumberScenarios-1):
        for v in range(point+1,NumberScenarios):
            euclidean_distance= np.sqrt((scenario_diversity_data['Best_Solution_Value'][point]-scenario_diversity_data['Best_Solution_Value'][v])**2
                                        +(scenario_diversity_data['Worst_Solution_Value'][point]-scenario_diversity_data['Worst_Solution_Value'][v])**2)
            if euclidean_distance < distance_limit:
                number_of_conquered_circles += 1


    #Scale dos valores com base no máximo e no mínimo
    std_scenario_diversity_data = scenario_diversity_data.copy()
    WorstValue = np.array([std_scenario_diversity_data['Worst_Solution_Value'].max()]*NumberScenarios)
    BestValue = np.array([std_scenario_diversity_data['Best_Solution_Value'].min()]*NumberScenarios)
    std_scenario_diversity_data['Worst_Solution_Value'] = (np.array(scenario_diversity_data['Worst_Solution_Value'])-BestValue)/(WorstValue-BestValue)
    std_scenario_diversity_data['Best_Solution_Value'] = (np.array(scenario_diversity_data['Best_Solution_Value'])-BestValue)/(WorstValue-BestValue)

    #Calcular o desvio padrão das distâncias ao ponto mais perto
    result_std = np.zeros(NumberScenarios-1)
    std_scenario_diversity_data = std_scenario_diversity_data.sort_values(by=['Best_Solution_Value']).reset_index(drop=True)
    for v in range(0,NumberScenarios-1):
        result_std[v] = np.sqrt((std_scenario_diversity_data['Best_Solution_Value'][v]-std_scenario_diversity_data['Best_Solution_Value'][v+1])**2
                                +(std_scenario_diversity_data['Worst_Solution_Value'][v]-std_scenario_diversity_data['Worst_Solution_Value'][v+1])**2)

    #Calcular dispersão com base na grelha (o número máxixo é igual ao número de cenários menos os dois extremos que são criados)
    number_of_conquered_squares = len(result_grid[result_grid>0])
    distance_square_root = round(result_std.std(),4)

    #Construção do plot com os quadrados
    Plot_title = f'points={number_of_conquered_squares} | std={distance_square_root}  | Overlaped_circles={number_of_conquered_circles}'
    if plot_results == True:
        plt.figure()
        fig = plt.figure()
        ax = fig.gca()
        ax.set(xlabel='Best Solution', ylabel='Worst Solution',title=f'{Plot_title}')
        ax.set_xlim(xmin=x_axis_grid_coordinates.min(),xmax=x_axis_grid_coordinates.max())
        ax.set_ylim(ymin=y_axis_grid_coordinates.min(),ymax=y_axis_grid_coordinates.max())
        ax.set_xticks(x_axis_grid_coordinates)
        ax.set_yticks(y_axis_grid_coordinates)
        plt.plot(scenario_diversity_data['Best_Solution_Value'], scenario_diversity_data['Worst_Solution_Value'], 'ro')
        plt.grid()
        plt.show()

    return number_of_conquered_squares,distance_square_root, number_of_conquered_circles

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

    #guardar o plot
    plt.savefig(f'{PATH_results}{instance_name}_diversity.png')
    plt.show()

#create function to calculate Manhattan distance (https://www.statology.org/manhattan-distance-python/) - old functions
def manhattan(a,b):

    #Calculate the difference between two vectors
    distance = 0
    for val1, val2 in zip(a,b):
        distance += np.abs(int(val1)-int(val2))

    return distance

#Funcao para medir a dispersão dos cenários - old functions
def meaure_diversity_dispersion(PATHInstancia, PATH_scenario_diversity_filename, FamilyType, generation):

    #Ler os dados
    scenario_diversity_data = pd.read_csv(PATHInstancia + PATH_scenario_diversity_filename, sep='\t')

    #Filtrar o tipo de instancia
    scenario_diversity_data = scenario_diversity_data[scenario_diversity_data['Instance'].str.contains(FamilyType)]

    #Converter as colunas numéricas para o formato correto
    scenario_diversity_data['Generation'] = pd.to_numeric(scenario_diversity_data['Generation'], errors='coerce')
    scenario_diversity_data['Best_Solution_Value'] = pd.to_numeric(scenario_diversity_data['Best_Solution_Value'], errors='coerce')
    scenario_diversity_data['Worst_Solution_Value'] = pd.to_numeric(scenario_diversity_data['Worst_Solution_Value'], errors='coerce')

    #Dados da primeira geração
    scenario_diversity_data = scenario_diversity_data.loc[(scenario_diversity_data.Generation == generation)].reset_index(drop=True)
    NumberScenarios = scenario_diversity_data.shape[0]

    #Calcular dispersão ideal para a primeira geração
    delta_x = (scenario_diversity_data['Best_Solution_Value'].max() - scenario_diversity_data['Best_Solution_Value'].min())/(NumberScenarios-1)#delta ideal no eixo dos x
    delta_y = (scenario_diversity_data['Worst_Solution_Value'].max() - scenario_diversity_data['Worst_Solution_Value'].min())/(NumberScenarios-1)#delta ideal no eixo dos y

    #Calcular a posição dos pontos
    scenario_diversity_data['Best_position_solution_value'] = np.arange(scenario_diversity_data['Best_Solution_Value'].min(),scenario_diversity_data['Best_Solution_Value'].max()+delta_x-1,delta_x)
    scenario_diversity_data['Worst_position_solution_value'] = np.arange(scenario_diversity_data['Worst_Solution_Value'].min(),scenario_diversity_data['Worst_Solution_Value'].max()+delta_y-1,delta_y)

    #Calcular o vetor de semelhança com base nos pontos que estão dentro dos quadrados
    scenario_diversity_data['Vetor_pontos'] = np.zeros(NumberScenarios)
    for i in range(1,NumberScenarios):
        for j in range(0,NumberScenarios):
            if (scenario_diversity_data['Best_Solution_Value'][j] < scenario_diversity_data['Best_position_solution_value'][i]) & (scenario_diversity_data['Best_Solution_Value'][j] >= scenario_diversity_data['Best_position_solution_value'][i-1]):
                if (scenario_diversity_data['Worst_Solution_Value'][j] < scenario_diversity_data['Worst_position_solution_value'][i]) & (scenario_diversity_data['Worst_Solution_Value'][j] >= scenario_diversity_data['Worst_position_solution_value'][i-1]):
                    scenario_diversity_data['Vetor_pontos'][i] += 1

    #Calcular a distância de Manhattan entre o vetor de pontos ideal e o vetor de pontos obtido
    distance = manhattan(np.ones(NumberScenarios),scenario_diversity_data['Vetor_pontos'])

    return distance, scenario_diversity_data

#Funcao para medir os extremos dos cenários
def meaure_extremes(PATHInstancia, PATH_scenario_diversity_filename, FamilyType, generation):

    #Ler os dados
    scenario_diversity_data = pd.read_csv(PATHInstancia + PATH_scenario_diversity_filename, sep='\t')

    #Filtrar o tipo de instancia
    scenario_diversity_data = scenario_diversity_data[scenario_diversity_data['Instance'].str.contains(FamilyType)]

    #Converter as colunas numéricas para o formato correto
    scenario_diversity_data['Generation'] = pd.to_numeric(scenario_diversity_data['Generation'], errors='coerce')
    scenario_diversity_data['Best_Solution_Value'] = pd.to_numeric(scenario_diversity_data['Best_Solution_Value'], errors='coerce')
    scenario_diversity_data['Worst_Solution_Value'] = pd.to_numeric(scenario_diversity_data['Worst_Solution_Value'], errors='coerce')

    #Dados da primeira geração
    scenario_diversity_data = scenario_diversity_data.loc[(scenario_diversity_data.Generation == generation)].reset_index(drop=True)
    NumberScenarios = scenario_diversity_data.shape[0]

    #Calcular dispersão ideal para a primeira geração
    delta_x = (scenario_diversity_data['Best_Solution_Value'].max() - scenario_diversity_data['Best_Solution_Value'].min())#delta ideal no eixo dos x
    delta_y = (scenario_diversity_data['Worst_Solution_Value'].max() - scenario_diversity_data['Worst_Solution_Value'].min())#delta ideal no eixo dos y

    #Calcular a distância euclideana entre dois pontos extremos
    distance = np.sqrt(delta_x**2+delta_y**2)

    return distance

#Função para criar um gif do diversity plot ao longo das gerações
def build_gif_diversity_plot(PATHInstancia, PATH_scenario_diversity_filename, FamilyType, instance_name, generation):

    #Ler os dados
    scenario_diversity_data = pd.read_csv(PATHInstancia + PATH_scenario_diversity_filename, sep='\t')

    #Filtrar o tipo de instancia
    scenario_diversity_data = scenario_diversity_data[scenario_diversity_data['Instance'].str.contains(FamilyType)]

    #Converter as colunas numéricas para o formato correto
    scenario_diversity_data['Generation'] = pd.to_numeric(scenario_diversity_data['Generation'], errors='coerce')
    scenario_diversity_data['Best_Solution_Value'] = pd.to_numeric(scenario_diversity_data['Best_Solution_Value'], errors='coerce')
    scenario_diversity_data['Worst_Solution_Value'] = pd.to_numeric(scenario_diversity_data['Worst_Solution_Value'], errors='coerce')

    #Plot dos dados para uma geração em específico
    Max_y_value, Max_x_value = scenario_diversity_data['Worst_Solution_Value'].max(), scenario_diversity_data['Best_Solution_Value'].max()
    plot_data = scenario_diversity_data.loc[(scenario_diversity_data.Generation == generation)]

    #Calcular as métricas chave da diversidade dos cenários
    number_of_conquered_squares, distance_square_root, number_of_conquered_circles, = calculate_dispersion(PATHInstancia, PATH_scenario_diversity_filename, 'Clustered', generation)
    title = f'Genation={generation} | Squares={number_of_conquered_squares} | std={distance_square_root}  | Circles={number_of_conquered_circles}'

    #Construição da notação do gráfico
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(plot_data['Best_Solution_Value'], plot_data['Worst_Solution_Value'], 'o', color='green')
    ax.grid()
    ax.set(xlabel='Best Solution', ylabel='Worst Solution',
           title=f'{FamilyType}_{title}')

    # IMPORTANT ANIMATION CODE HERE
    # Used to keep the limits constant
    ax.set_ylim(0, Max_y_value)
    ax.set_xlim(0, Max_x_value)

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image
