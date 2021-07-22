#####################################################################
###Script to evaluate the BRKGA scenarios population quality      ###
#####################################################################

#Ver https://math.stackexchange.com/questions/728040/placing-n-points-in-a-mxm-square-grid
#Ver https://stackoverflow.com/questions/11015252/countig-points-in-boxes-of-a-grid

#Bibliotecas a ser importadas
from BRKGA_utils import *

#Testes de funções
File_list_diversity =  [f for f in File_list if 'scenario_diversity_sol' in f]
number_of_conquered_squares,distance_square_root, number_of_conquered_circles = calculate_dispersion(PATHInstancia, File_list_diversity[3], 'Clustered', 1, True)
number_of_conquered_squares,distance_square_root, number_of_conquered_circles = calculate_dispersion(PATHInstancia, File_list_diversity[3], 'Clustered', 500, True)
ScenarioDiversityPlot(PATHInstancia, 'instance_name', File_list_diversity[0], 'Clustered')
# number_of_conquered_squares,distance_square_root, number_of_conquered_circles = calculate_dispersion(PATHInstancia, 'R0H0E0_N100TW5_HighUncHighRiskLowImp_scenario_diversity_sol_A5_Pe25_Pm10_rh80_scen_Pe25_Pm30_rh80.csv', 'Clustered', 999, True)


#Analise dispersão dos cenários (este código depois é para retirar)
# Distance_results = pd.DataFrame({'Instance': File_list_diversity,
#                                  'Initial_squares': np.zeros(len(File_list_diversity)),
#                                  'Final_squares': np.zeros(len(File_list_diversity)),
#                                  'Initial_std': np.zeros(len(File_list_diversity)),
#                                  'Final_std': np.zeros(len(File_list_diversity)),
#                                  'Initial_circles': np.zeros(len(File_list_diversity)),
#                                  'Final_circles': np.zeros(len(File_list_diversity)),
#                                  'Initial_extremes': np.zeros(len(File_list_diversity)),
#                                  'Final_extremes': np.zeros(len(File_list_diversity))})
#
# #Calcular a dispersao e extremos dos cenários
# for diversity in File_list_diversity:
#     try:
#         Distance_results.loc[(Distance_results.Instance == diversity),'Initial_squares'], \
#         Distance_results.loc[(Distance_results.Instance == diversity),'Initial_std'], \
#         Distance_results.loc[(Distance_results.Instance == diversity),'Initial_circles'] = calculate_dispersion(PATHInstancia, diversity, 'Clustered', 1)
#         Distance_results.loc[(Distance_results.Instance == diversity),'Final_squares'], \
#         Distance_results.loc[(Distance_results.Instance == diversity),'Final_std'], \
#         Distance_results.loc[(Distance_results.Instance == diversity),'Final_circles'] = calculate_dispersion(PATHInstancia, diversity, 'Clustered', 999)
#         Distance_results.loc[(Distance_results.Instance == diversity),'Initial_extremes'] = meaure_extremes(PATHInstancia, diversity, 'Clustered', 1)
#         Distance_results.loc[(Distance_results.Instance == diversity),'Final_extremes'] = meaure_extremes(PATHInstancia, diversity, 'Clustered', 999)
#     except:
#         print(f"A problem ocurred with instance {diversity}")
#
# #Calcular indicadores de improvement
# Distance_results['Improvement_squares'] = (Distance_results['Final_squares']-Distance_results['Initial_squares'])/Distance_results['Initial_squares']
# Distance_results['Improvement_std'] = (Distance_results['Initial_std']-Distance_results['Final_std'])/Distance_results['Initial_std']
# Distance_results['Improvement_circles'] = (Distance_results['Initial_circles']-Distance_results['Final_circles'])/Distance_results['Initial_circles']
# Distance_results['Improvement_extremes'] = (Distance_results['Final_extremes']-Distance_results['Initial_extremes'])/Distance_results['Initial_extremes']
#
# #Colocar a combinação de parâmetros (soluções e cenários)
# Distance_results['Instance'] = [file.replace(".csv","") for file in File_list_diversity]
# Distance_results['SolutionParameterization'] = [dist.split('_sol_')[1].split('_scen_')[0] for dist in Distance_results['Instance']]
# Distance_results['ScenarioParameterization'] = [dist.split('_sol_')[1].split('_scen_')[1] for dist in Distance_results['Instance']]
#
# #Export results (o path tem que se alterar)
# Distance_results.to_csv(path_or_buf=f'./New_Dispersion_results.csv',index=False)

#Calcular a dispersão iterativa para uma instância em particular
File_list_diversity_iteration = [f for f in File_list if 'scenario_diversity_iteration' in f]

#Calcular a dispersao dos cenários
export_results = True
for file in File_list_diversity_iteration:
    Generations = 999
    Distance_results = pd.DataFrame({
        'Generation': np.zeros(Generations),
        'Squares': np.zeros(Generations),
        'Std': np.zeros(Generations),
        'Circles': np.zeros(Generations),
        'Extremes': np.zeros(Generations)
    })
    for gen in range(1,Generations):
        try:
            Distance_results['Squares'][gen],Distance_results['Std'][gen],Distance_results['Circles'][gen] = calculate_dispersion(PATHInstancia, file, 'Clustered', gen)
            Distance_results['Extremes'][gen] = meaure_extremes(PATHInstancia, file, 'Clustered', gen)
            Distance_results['Generation'][gen] = gen
            export_results = True
        except:
            print(f"Last generation for {File_list_diversity_iteration[0]} is {gen-1}")
            export_results = False
            break

    #Exportar os resultados
    if export_results == True:
        Filename = file.replace('.csv','')
        Distance_results = Distance_results[Distance_results['Generation']!=0]
        Distance_results.to_csv(path_or_buf=f'./Results_example/{Filename}_Dispersion_iteration_results.csv',index=False)


# #Gif plot (o path tem que se alterar)
# imageio.mimsave('./Results_example/test_3.gif', [build_gif_diversity_plot(PATHInstancia, File_list_diversity_iteration[0], 'Random',gen) for gen in range(1,500)], fps=2)


