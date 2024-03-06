import mlrose_hiive as mlrose
import numpy as np
import plotly.graph_objects as go
import time
import networkx as nx
import matplotlib.pyplot as plt


'''
1) Define a fitness function object.
2) Define an optimization problem object.
3) Select and run a randomized optimization algorithm


Required Plots:

1) Fitness / Iteration
2) Fitness / Problem Size
3) Function Evaluation
4) Wall Clock Time
'''


# # MaxKcolor problem
# # defining fitness

graph = nx.erdos_renyi_graph(n=10, p=0.3, seed=42)

# Plot the graph
nx.draw(graph, with_labels=True, node_color='lightblue')
plt.title('Random Graph')
plt.show()

plt.savefig('max_k_color_problem.png')


# Convert the graph to an adjacency matrix
adjacency_matrix = nx.to_numpy_array(graph)

fitness = mlrose.MaxKColor(edges=graph.edges)

# optimization for num_iteration:

length = 100

iterations = list(range(length))

fitness_sa = []
decay_25_ft = []
decay_50_ft = []
decay_75_ft = []
decay_95_ft = []

decay = [0.25,0.50,0.75,0.95]
num_restart =[0,2,4,6]

num_rs0_ft = []
num_rs2_ft = []
num_rs4_ft = []
num_rs6_ft = []


pop_size = [10, 50, 75, 100]
ps_1_ft = []
ps_2_ft = []
ps_3_ft = []
ps_4_ft = []

pop_size_mimic = [10, 50, 75, 200]
ps_1_ft_m = []
ps_2_ft_m = []
ps_3_ft_m = []
ps_4_ft_m = []


# for i in iterations:
# # Solve problem using simulated annealing #################
    
for i in iterations:
    for de in decay:
        problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
        schedule = mlrose.GeomDecay(init_temp=5, decay=de, min_temp=0.001)

        best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = schedule,
                                                                max_attempts = 100, max_iters = i,
                                                                random_state = 1)

        if de == 0.25:
            decay_25_ft.append(best_fitness)
            
        elif de == 0.50:
            decay_50_ft.append(best_fitness)
            

        elif de == 0.75:
            decay_75_ft.append(best_fitness)
            
        elif de == 0.95:
            decay_95_ft.append(best_fitness)    
            
    print(i)

    # RHC#######################

    for rs in num_restart:
        #problem = mlrose.DiscreteOpt(length = 15, fitness_fn = fitness, maximize = True, max_val = 15)
        problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
        best_state_rhc, best_fitness_rhc, fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=i, restarts=rs, random_state=1)

        if rs == 0:
            num_rs0_ft.append(best_fitness_rhc)
            
        elif rs == 2:
            num_rs2_ft.append(best_fitness_rhc)
            
        elif rs == 4:
            num_rs4_ft.append(best_fitness_rhc)
            
        elif rs == 6:
            num_rs6_ft.append(best_fitness_rhc)

    # Genetics Algorithm##############################

    for p_size in pop_size:
        #problem = mlrose.DiscreteOpt(length=15, fitness_fn=fitness, maximize=True, max_val=15)
        problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3) 
        best_state_genalgo, best_fitness_genalgo, fitness_curve = mlrose.genetic_alg(problem, pop_size = p_size, mutation_prob=0.35, max_attempts=100, max_iters=i, random_state=1)
        
        if p_size == 10:
            ps_1_ft.append(best_fitness_genalgo)
            
        elif p_size == 50:
            ps_2_ft.append(best_fitness_genalgo)
            
        elif p_size == 75:
            ps_3_ft.append(best_fitness_genalgo)
            
        elif p_size == 100:
            ps_4_ft.append(best_fitness_genalgo)


    # MIMIC algorithm ###########################################
    for p_size in pop_size_mimic:
        #problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)
        problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
        problem.set_mimic_fast_mode(True)
        best_state, best_fitness_mimic,fitness_curve = mlrose.mimic(problem, pop_size=p_size, keep_pct=0.2, max_attempts=1000, max_iters=i ,random_state=1)
        
        if p_size == 10:
            ps_1_ft_m.append(best_fitness_mimic)
            
        elif p_size == 50:
            ps_2_ft_m.append(best_fitness_mimic)
            
        elif p_size == 75:
            ps_3_ft_m.append(best_fitness_mimic)
            
        elif p_size == 200:
            ps_4_ft_m.append(best_fitness_mimic)


    # fitness_sa.append(100-best_fitness)
    print(i)

# optimization Simulated annealing
fig = go.Figure()
fig.add_trace(go.Scatter(x=iterations, y=decay_25_ft,
                    mode='lines',
                    name='0.25 Decay'))

fig.add_trace(go.Scatter(x=iterations, y=decay_50_ft,
                    mode='lines',
                    name='0.50 Decay'))

fig.add_trace(go.Scatter(x=iterations, y=decay_75_ft,
                    mode='lines',
                    name='0.75 Decay'))

fig.add_trace(go.Scatter(x=iterations, y=decay_95_ft,
                    mode='lines',
                    name='0.95 Decay'))


# Update layout to add titles
fig.update_layout(title='Max K color- SA Decay Optimization',  # Title of the plot
                  xaxis_title='Iterations',       # Title of the x-axis
                  yaxis_title='Fitness')         # Title of the y-axis

fig.show()

#fig.write_image("/home/srasool/Documents/Machine_learning/Assignment_2/images/SA_Decay_opti.png", width=800, height=600, scale=2)


# optimization RHC

fig = go.Figure()
fig.add_trace(go.Scatter(x=iterations, y=num_rs0_ft,
                    mode='lines',
                    name='0 restart'))

fig.add_trace(go.Scatter(x=iterations, y=num_rs2_ft,
                    mode='lines',
                    name='2 restart'))

fig.add_trace(go.Scatter(x=iterations, y=num_rs4_ft,
                    mode='lines',
                    name='4 restart'))

fig.add_trace(go.Scatter(x=iterations, y=num_rs6_ft,
                    mode='lines',
                    name='6 restart'))


# Update layout to add titles
fig.update_layout(title='Max k Color- RHC Restart Optimization',  # Title of the plot
                  xaxis_title='Iterations',       # Title of the x-axis
                  yaxis_title='Fitness')         # Title of the y-axis

fig.show()

#fig.write_image("/home/srasool/Documents/Machine_learning/Assignment_2/images/RHC_Restart_opti.png", width=800, height=600, scale=2)

# optimization gen_algo

fig = go.Figure()
fig.add_trace(go.Scatter(x=iterations, y=ps_1_ft,
                    mode='lines',
                    name='10 pop_size'))

fig.add_trace(go.Scatter(x=iterations, y=ps_2_ft,
                    mode='lines',
                    name='50 pop_size'))

fig.add_trace(go.Scatter(x=iterations, y=ps_3_ft,
                    mode='lines',
                    name='75 pop_size'))

fig.add_trace(go.Scatter(x=iterations, y=ps_4_ft,
                    mode='lines',
                    name='100 pop_size'))


# Update layout to add titles
fig.update_layout(title='Max K color - Genetic algorithm Pop Size Optimization',  # Title of the plot
                  xaxis_title='Iterations',       # Title of the x-axis
                  yaxis_title='Fitness')         # Title of the y-axis

fig.show()

#fig.write_image("/home/srasool/Documents/Machine_learning/Assignment_2/images/GA_algo_pop_size_opti.png", width=800, height=600, scale=2)

# optimization minic

fig = go.Figure()
fig.add_trace(go.Scatter(x=iterations, y=ps_1_ft_m,
                    mode='lines',
                    name='10 pop_size'))

fig.add_trace(go.Scatter(x=iterations, y=ps_2_ft_m,
                    mode='lines',
                    name='50 pop_size'))

fig.add_trace(go.Scatter(x=iterations, y=ps_3_ft_m,
                    mode='lines',
                    name='75 pop_size'))

fig.add_trace(go.Scatter(x=iterations, y=ps_4_ft_m,
                    mode='lines',
                    name='100 pop_size'))


# Update layout to add titles
fig.update_layout(title='Max K Color - Mimic algorithm Pop Size Optimization',  # Title of the plot
                  xaxis_title='Iterations',       # Title of the x-axis
                  yaxis_title='Fitness')         # Title of the y-axis

fig.show()

#fig.write_image("/home/srasool/Documents/Machine_learning/Assignment_2/images/Mimic_algo_pop_size_opti.png", width=800, height=600, scale=2)


print('done')



# Define different problem sizes
problem_sizes = [10, 15, 20]


ps_10_ft_sa = np.array([])
ps_15_ft_sa = np.array([])
ps_20_ft_sa = np.array([])


ps_10_ft_gen = np.array([])
ps_15_ft_gen = np.array([])
ps_20_ft_gen = np.array([])


ps_10_ft_rhc = np.array([])
ps_15_ft_rhc = np.array([])
ps_20_ft_rhc = np.array([])


ps_10_ft_mimic = np.array([])
ps_15_ft_mimic = np.array([])
ps_20_ft_mimic = np.array([])


for ps in problem_sizes:

     graph = nx.erdos_renyi_graph(n=ps, p=0.3, seed=42)

# Convert the graph to an adjacency matrix
     adjacency_matrix = nx.to_numpy_array(graph)

     fitness = mlrose.MaxKColor(edges=graph.edges)


    #num_iterations
     length = 100
     iterations = list(range(length))

    # solve problem using simulated annealing
     schedule = mlrose.GeomDecay(init_temp=5, decay=0.75, min_temp=0.001)

     for i in iterations:

          #SA
          problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
          best_state_sa, best_fitness_sa = mlrose.simulated_annealing(problem, schedule = schedule,
                                                                      max_attempts = 100, max_iters = i,
                                                                      random_state = 1)
          
          if ps == 10:
               ps_10_ft_sa = np.append(ps_10_ft_sa, best_fitness_sa)
               
          elif ps == 15:
               ps_15_ft_sa = np.append(ps_15_ft_sa, best_fitness_sa)
               
          elif ps == 20:
               ps_20_ft_sa = np.append(ps_20_ft_sa, best_fitness_sa)


          #Gen
          problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
          best_state_gen, best_fitness_gen, fitness_curve = mlrose.genetic_alg(problem, pop_size = 500, mutation_prob=0.35, max_attempts=100, max_iters=i, random_state=1)
          
          if ps == 10:
               ps_10_ft_gen = np.append(ps_10_ft_gen, best_fitness_gen)
               
          elif ps == 15:
               ps_15_ft_gen = np.append(ps_15_ft_gen, best_fitness_gen)
               
          elif ps == 20:
               ps_20_ft_gen = np.append(ps_20_ft_gen, best_fitness_gen)


          #RHC
          problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
          best_state_rhc, best_fitness_rhc, fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=i, restarts=6, random_state=1)
          
          if ps == 10:
               ps_10_ft_rhc = np.append(ps_10_ft_rhc, best_fitness_rhc)
               
          elif ps == 15:
               ps_15_ft_rhc = np.append(ps_15_ft_rhc, best_fitness_rhc)
               
          elif ps == 20:
               ps_20_ft_rhc = np.append(ps_20_ft_rhc, best_fitness_rhc)


          #Mimic
          problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
          problem.set_mimic_fast_mode(True)
          best_state, best_fitness_mimic,fitness_curve = mlrose.mimic(problem, pop_size=500, keep_pct=0.2, max_attempts=100, max_iters=i ,random_state=1)

          if ps == 10:
               ps_10_ft_mimic = np.append(ps_10_ft_mimic, best_fitness_mimic)
               
          elif ps == 15:
               ps_15_ft_mimic = np.append(ps_15_ft_mimic, best_fitness_mimic)
               
          elif ps == 20:
               ps_20_ft_mimic = np.append(ps_20_ft_mimic, best_fitness_mimic)


          print(i)

     print(ps)

avg_ft_sa = (ps_10_ft_sa +  ps_15_ft_sa + ps_20_ft_sa)/3

avg_ft_gen = (ps_10_ft_gen +  ps_15_ft_gen + ps_20_ft_gen)/3

avg_ft_rhc = (ps_10_ft_rhc +  ps_15_ft_rhc + ps_20_ft_rhc)/3

avg_ft_mimic = (ps_10_ft_mimic +  ps_15_ft_mimic + ps_20_ft_mimic)/3

# optimization Simulated annealing
fig = go.Figure()
fig.add_trace(go.Scatter(x=iterations, y=avg_ft_sa,
                    mode='lines',
                    name='Sa - Avg Fitness'))

fig.add_trace(go.Scatter(x=iterations, y=avg_ft_gen,
                    mode='lines',
                    name='Gen - Avg Fitness'))

fig.add_trace(go.Scatter(x=iterations, y=avg_ft_rhc,
                    mode='lines',
                    name='Rhc - Avg Fitness'))

fig.add_trace(go.Scatter(x=iterations, y=avg_ft_mimic,
                    mode='lines',
                    name='mimic - Avg Fitness'))


# Update layout to add titles
fig.update_layout(title='Max k Color - RO With Varying Problem Size (10, 15, 20)',  # Title of the plot
                  xaxis_title='Iterations',       # Title of the x-axis
                  yaxis_title='Avg Fitness')         # Title of the y-axis

fig.show()

#fig.write_image("/home/srasool/Documents/Machine_learning/Assignment_2/images/Ro-Problem-size.png", width=800, height=600, scale=2)

print('done')

# Function Evaluation
# decay 0.95
ps = 15

graph = nx.erdos_renyi_graph(n=ps, p=0.3, seed=42)

# Convert the graph to an adjacency matrix
adjacency_matrix = nx.to_numpy_array(graph)

fitness = mlrose.MaxKColor(edges=graph.edges)

# optimization for num_iteration:

length = 100

iterations = list(range(length))


for i in iterations:
# Solve problem using simulated annealing
    schedule = mlrose.GeomDecay(init_temp=5, decay=0.75, min_temp=0.001)
    problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
    best_state, best_fitness, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule,
                                                                max_attempts = 100, max_iters = i,
                                                                fevals= True,curve=True ,random_state = 1)
    
    # RHC
    problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=i, restarts=6, curve=True, random_state=1)

    # Gen
    problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
    best_state_gen, best_fitness_gen, fitness_curve_gen = mlrose.genetic_alg(problem, pop_size = 500, mutation_prob=0.35, max_attempts=100,curve=True, max_iters=i, random_state=1)
        
    # Mimic
    problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
    problem.set_mimic_fast_mode(True)
    best_state, best_fitness_mimic,fitness_curve_mimic = mlrose.mimic(problem, pop_size=500, keep_pct=0.2, max_attempts=100,curve=True, max_iters=i ,random_state=1)

    print(i)
FeVal_SA = fitness_curve_sa[-1][1]

FeVal_RHC = fitness_curve_rhc[-1][1]

FeVal_GEN = fitness_curve_gen[-1][1]

FeVal_MIMIC = fitness_curve_mimic[-1][1]


FeVals = [FeVal_SA, FeVal_RHC, FeVal_GEN, FeVal_MIMIC]
labels = ['SA', 'RHC', 'Gen', 'Mimic']

fig = go.Figure(data=[go.Bar(x=labels, y=FeVals)])
fig.update_layout(title='Function Evaluations by Algorithm', xaxis_title='Algorithm', yaxis_title='Function Evaluations')
fig.show()
#fig.write_image("/home/srasool/Documents/Machine_learning/Assignment_2/images/Max_k_color_ro_fevals.png", width=800, height=600, scale=2)


# Best Algorithm  / clock wall time /SA


random_seeds_ls = [10, 150000, 2000, 3, 200]

#fitness = mlrose.Queens()


#num_iterations
length = 100
iterations = list(range(length))

# solve problem using simulated annealing
schedule = mlrose.GeomDecay(init_temp=5, decay=0.75, min_temp=0.001)

rs_1_sa = np.array([])
rs_2_sa = np.array([])
rs_3_sa = np.array([])
rs_4_sa = np.array([])
rs_5_sa = np.array([])


# Start the timer
start_time = time.time()
for rs in random_seeds_ls:
     ps = 15
     graph = nx.erdos_renyi_graph(n=ps, p=0.3, seed=rs)
     # Convert the graph to an adjacency matrix
     adjacency_matrix = nx.to_numpy_array(graph)
     fitness = mlrose.MaxKColor(edges=graph.edges)
     problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)


     for i in iterations:
               
               #Sa
               best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = schedule,
                                                                           max_attempts = 100, max_iters = i,
                                                                           random_state = rs)

               if rs == 10:
                    rs_1_sa = np.append(rs_1_sa, best_fitness)

               elif rs == 150000:
                    rs_2_sa = np.append(rs_2_sa, best_fitness)

               elif rs == 2000:
                    rs_3_sa = np.append(rs_3_sa, best_fitness)

               elif rs == 3:
                    rs_4_sa = np.append(rs_4_sa, best_fitness)

               elif rs == 200:
                    rs_5_sa = np.append(rs_5_sa, best_fitness)

     print(rs)

fitness_avg_sa = (rs_1_sa +rs_2_sa + rs_3_sa + rs_4_sa + rs_5_sa)/len(random_seeds_ls)
fitness_std_sa = np.std(fitness_avg_sa) * 0.1

# Calculate the elapsed time
ro_sa_wall_time_sa = time.time() - start_time


# Best Algorithm  / clock wall time /rhc
random_seeds_ls = [10, 150000, 2000, 3, 200]

#num_iterations
length = 100
iterations = list(range(length))

# solve problem using simulated annealing
schedule = mlrose.GeomDecay(init_temp=5, decay=0.95, min_temp=0.001)


rs_1_rhc = np.array([])
rs_2_rhc = np.array([])
rs_3_rhc = np.array([])
rs_4_rhc = np.array([])
rs_5_rhc = np.array([])

# Start the timer
start_time = time.time()
for rs in random_seeds_ls:
     ps = 15
     graph = nx.erdos_renyi_graph(n=ps, p=0.3, seed=rs)
     # Convert the graph to an adjacency matrix
     adjacency_matrix = nx.to_numpy_array(graph)
     fitness = mlrose.MaxKColor(edges=graph.edges)
     problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)    
     for i in iterations:
               
               #rhc    
               best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=i, restarts=6, curve=True, random_state=rs)

               if rs == 10:
                    rs_1_rhc = np.append(rs_1_rhc, best_fitness_rhc)

               elif rs == 150000:
                    rs_2_rhc = np.append(rs_2_rhc, best_fitness_rhc)

               elif rs == 2000:
                    rs_3_rhc = np.append(rs_3_rhc, best_fitness_rhc)

               elif rs == 3:
                    rs_4_rhc = np.append(rs_4_rhc, best_fitness_rhc)

               elif rs == 200:
                    rs_5_rhc = np.append(rs_5_rhc, best_fitness_rhc)

     print(rs)

fitness_avg_rhc = (rs_1_rhc +rs_2_rhc + rs_3_rhc + rs_4_rhc + rs_5_rhc)/len(random_seeds_ls)
fitness_std_rhc = np.std(fitness_avg_rhc) * 0.1

# Calculate the elapsed time
ro_sa_wall_time_rhc = time.time() - start_time



# Best Algorithm  / clock wall time /Gen


random_seeds_ls = [10, 150000, 2000, 3, 200]

#num_iterations
length = 100
iterations = list(range(length))



rs_1_gen = np.array([])
rs_2_gen = np.array([])
rs_3_gen = np.array([])
rs_4_gen = np.array([])
rs_5_gen = np.array([])

# Start the timer
start_time = time.time()
for rs in random_seeds_ls:
     ps = 15
     graph = nx.erdos_renyi_graph(n=ps, p=0.3, seed=rs)
     # Convert the graph to an adjacency matrix
     adjacency_matrix = nx.to_numpy_array(graph)
     fitness = mlrose.MaxKColor(edges=graph.edges)
     problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)    

     for i in iterations:
               
               #gen    
               best_state_gen, best_fitness_gen, fitness_curve_gen = mlrose.genetic_alg(problem, pop_size = 500, mutation_prob=0.35, max_attempts=100, curve=True, max_iters=i, random_state=rs)

               if rs == 10:
                    rs_1_gen = np.append(rs_1_gen, best_fitness_gen)

               elif rs == 150000:
                    rs_2_gen = np.append(rs_2_gen, best_fitness_gen)

               elif rs == 2000:
                    rs_3_gen = np.append(rs_3_gen, best_fitness_gen)

               elif rs == 3:
                    rs_4_gen = np.append(rs_4_gen, best_fitness_gen)

               elif rs == 200:
                    rs_5_gen = np.append(rs_5_gen, best_fitness_gen)

     print(rs)

fitness_avg_gen = (rs_1_gen +rs_2_gen + rs_3_gen + rs_4_gen + rs_5_gen)/len(random_seeds_ls)
fitness_std_gen = np.std(fitness_avg_gen) * 0.1

# Calculate the elapsed time
ro_sa_wall_time_gen = time.time() - start_time


# Best Algorithm  / clock wall time /mimic

random_seeds_ls = [10, 150000, 2000, 3, 200]


#num_iterations
length = 100
iterations = list(range(length))
problem.set_mimic_fast_mode(True)

rs_1_mimic = np.array([])
rs_2_mimic = np.array([])
rs_3_mimic = np.array([])
rs_4_mimic = np.array([])
rs_5_mimic = np.array([])


# Start the timer
start_time = time.time()
for rs in random_seeds_ls:
     ps = 15
     graph = nx.erdos_renyi_graph(n=ps, p=0.3, seed=rs)
     # Convert the graph to an adjacency matrix
     adjacency_matrix = nx.to_numpy_array(graph)
     fitness = mlrose.MaxKColor(edges=graph.edges)
     problem = mlrose.DiscreteOpt(length=len(adjacency_matrix), fitness_fn=fitness, maximize=True, max_val=3)
     for i in iterations:
               
               #mimic
               best_state, best_fitness_mimic,fitness_curve_mimic = mlrose.mimic(problem, pop_size=500, keep_pct=0.2, max_attempts=100,curve=True, max_iters=i ,random_state=rs)

               if rs == 10:
                    rs_1_mimic = np.append(rs_1_mimic, best_fitness_mimic)

               elif rs == 150000:
                    rs_2_mimic = np.append(rs_2_mimic, best_fitness_mimic)

               elif rs == 2000:
                    rs_3_mimic = np.append(rs_3_mimic, best_fitness_mimic)

               elif rs == 3:
                    rs_4_mimic = np.append(rs_4_mimic, best_fitness_mimic)

               elif rs == 200:
                    rs_5_mimic = np.append(rs_5_mimic, best_fitness_mimic)

     print(rs)

fitness_avg_mimic = (rs_1_mimic +rs_2_mimic + rs_3_mimic + rs_4_mimic + rs_5_mimic)/len(random_seeds_ls)
fitness_std_mimic = np.std(fitness_avg_mimic) * 0.1

# Calculate the elapsed time
ro_sa_wall_time_mimic = time.time() - start_time



fig = go.Figure()
fig.add_trace(go.Scatter(x=iterations, y=fitness_avg_sa,
                    mode='lines',
                    name='Sa - Avg Fitness'))

fig.add_trace(go.Scatter(x=np.concatenate([iterations, iterations[::-1]]),
                         y=np.concatenate([fitness_avg_sa - fitness_std_sa, (fitness_avg_sa + fitness_std_sa)[::-1]]),
                         fill='tozeroy', fillcolor='rgba(0,100,80,0.2)', mode='lines', line=dict(color='rgba(255,255,255,0)'),
                         name='0.1 Standard Deviation-sa'))

fig.add_trace(go.Scatter(x=iterations, y=fitness_avg_rhc,
                    mode='lines',
                    name='Rhc - Avg Fitness'))

fig.add_trace(go.Scatter(x=np.concatenate([iterations, iterations[::-1]]),
                         y=np.concatenate([fitness_avg_rhc - fitness_std_rhc, (fitness_avg_rhc + fitness_std_rhc)[::-1]]),
                         fill='tozeroy', fillcolor='rgba(0,100,80,0.2)', mode='lines', line=dict(color='rgba(255,255,255,0)'),
                         name='0.1 Standard Deviation-rhc'))


fig.add_trace(go.Scatter(x=iterations, y=fitness_avg_gen,
                    mode='lines',
                    name='Gen - Avg Fitness'))

fig.add_trace(go.Scatter(x=np.concatenate([iterations, iterations[::-1]]),
                         y=np.concatenate([fitness_avg_gen - fitness_std_gen, (fitness_avg_gen + fitness_std_gen)[::-1]]),
                         fill='tozeroy', fillcolor='rgba(0,100,80,0.2)', mode='lines', line=dict(color='rgba(255,255,255,0)'),
                         name='0.1 Standard Deviation-gen'))

fig.add_trace(go.Scatter(x=iterations, y=fitness_avg_mimic,
                    mode='lines',
                    name='Mimic - Avg Fitness'))

fig.add_trace(go.Scatter(x=np.concatenate([iterations, iterations[::-1]]),
                         y=np.concatenate([fitness_avg_mimic - fitness_std_mimic, (fitness_avg_mimic + fitness_std_mimic)[::-1]]),
                         fill='tozeroy', fillcolor='rgba(0,100,80,0.2)', mode='lines', line=dict(color='rgba(255,255,255,0)'),
                         name='0.1 Standard Deviation-mimic'))

# Update layout to add titles
fig.update_layout(title='K Max Color - Randomized Optimization Alogrithm Comparisions',  # Title of the plot
                  xaxis_title='Iterations',       # Title of the x-axis
                  yaxis_title='Fitness')         # Title of the y-axis

fig.show()
#fig.write_image("/home/srasool/Documents/Machine_learning/Assignment_2/images/best_algo_RS_Max_K_color.png", width=800, height=600, scale=2)

#wall time plot

FeVals = [ro_sa_wall_time_sa, ro_sa_wall_time_rhc, ro_sa_wall_time_gen, ro_sa_wall_time_mimic]
labels = ['SA','RHC', 'Genetics','mimic']

fig = go.Figure(data=[go.Bar(x=labels, y=FeVals)])
fig.update_layout(title='Wall Clock time by Algorithms - Per 100 Interations', xaxis_title='Algorithm', yaxis_title='Seconds')
fig.show()

#fig.write_image("/home/srasool/Documents/Machine_learning/Assignment_2/images/Ro_Max_k_Color_walltime.png", width=800, height=600, scale=2)

print('completed')








