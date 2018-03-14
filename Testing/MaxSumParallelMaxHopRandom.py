# coding=utf-8

'''
Created on 16 Dic 2017

@author: Andrea Montanari

This class is a testing about Experiments.
The Graphs colored have 50 variables and 10 values in domain. 
The results are saved as chart (.png) and as log file (.txt)
'''

import sys, os
from collections import defaultdict
import random
import matplotlib.pyplot as pl
import argparse

from copy import deepcopy

import time
import Queue

sys.path.append(os.path.abspath('../maxsum/'))
sys.path.append(os.path.abspath('../solver/'))
sys.path.append(os.path.abspath('../system/'))
sys.path.append(os.path.abspath('../Graph/'))
sys.path.append(os.path.abspath('../misc/'))
sys.path.append(os.path.abspath('../function/'))

from Agent import Agent
from NodeVariable import NodeVariable
from NodeFunction import NodeFunction
from TabularFunction import TabularFunction
from NodeArgument import NodeArgument
from COP_Instance import COP_Instance
from MaxSum import MaxSum


def main():    
    '''
        invoke the parser that takes parameters from the command line
    '''
    args = getParser()
    '''
        How many iterations?
    '''
    nIterations = args.iterations
    '''
        How many instances? 
    '''   
    nIstances = args.instances
    '''
        number of variables in Dcop
    '''
    nVariables = args.variables
    '''
        max/min
    '''
    op = args.op
    '''
        location of MaxSum report
    '''
    reportMaxSum = args.reportMaxSum 
     
    '''
        location of FactorGraph report
    '''
    infoGraphPathFile = args.reportFactorGraph 
         
    '''
        location of Charts
    '''
    chartsFile = args.reportCharts    
     
    '''
        Constraint Optimization Problem
    '''
    cop = None 
     
    '''
        average of 50 instances of problem
    '''
    average = list()
     
    for i in range(nVariables + 1):
        average.append(0) 
         
    state = 0
     
    variables = list()
    functions = list()
    agents = list()
     
    agent = None
     
    finalAssignaments = list()
     
    oldAssignaments = list()
     
    core = None
    
    originalCop = None
    
    actualValue = 0

    '''
        list of all values of RUN for each iteration
    '''
    averageValues = list()
    
    iterations = list()
    
    iterationsInstances = list()
    
    averageValuesInstances= list()

    varSet = (0.1 * nVariables)
    
    string = "Max Sum\t Average Conflicts\n"
 
    for run in range(nIstances): 
    
            averageValues = list()
            
            iterations = list()
            '''
                fileName of log (Iteration Conflicts AverageDifferenceLink)
                save on file Iteration Conflicts AverageDifferenceLink
            '''           
            finalAssignaments = list()
             
            oldAssignaments = list()
             
            for i in range(nVariables):
                finalAssignaments.append(0)
                oldAssignaments.append(0)
                         
            '''
                values of MaxSum for each iteration
            '''
            values = list()    
           
            '''
                create a new COP with a colored Graph and 
                save the factorgraph on file
            '''
            cop = create_DCop(infoGraphPathFile, nVariables, run) 
             
            i = 0
             
            going = False
             
            functions = cop.getNodeFunctions()
            variables = cop.getNodeVariables()
             
            '''
                repeat the algorithm for 50 times
                every exec set the value of variable
            '''
            while((len(variables) > 0) & (len(functions) > 0)):
                '''
                    if it isn't the first exec of MaxSum,
                    change the factor graph and repeat the algorithm
                '''
                if(going == True):                    
                    '''
                        the agent of the cop
                    '''
                    agent = (core.getCop().getAgents())[0]
                     
                    '''
                        the variables of the cop
                    '''
                    variables = core.getCop().getNodeVariables()
                     
                    '''
                        the functions of the cop
                    '''
                    functions = core.getCop().getNodeFunctions()
                    
                    graph = dict()
                    
                    hop = dict()
                    
                    for variable in variables:
                        graph[variable] = list()
                        '''
                            for each node function
                        '''
                        for f in variable.getNeighbour():
                            for v in f.getNeighbour():
                                if((v.getId()) != (variable.getId())):
                                    graph[variable].append(v)  
                                    hop[(variable,v)] = float('-inf')              
                    
                    for v in range(len(variables)):
                        for k in range(v+1,len(variables)): 
                             '''
                                if there is at least a neighbour
                             '''
                             if(len(graph[variable]) > 0):
                                 '''
                                    bfs visit with shortest path
                                 '''    
                                 shortestPath=list(bfs_shortest_path(graph, variables[v], variables[k]))
    
                                 '''
                                    if there is a path, update the number of hops
                                 '''        
                                 if(len(shortestPath) > 0):    
                                     hop[(variables[v],variables[k])] = len(shortestPath) - 1         
                                       
                    variablesToRemove = list()                   
                                        
                    '''
                        choose a random variable
                    ''' 
                    indexRemove = random.randint(0, len(variables) - 1)
                    
                    '''
                        add the random variable to remove
                    '''
                    variablesToRemove.append(variables[indexRemove])
                    
                    maxHop = float('-inf') 
                    
                    v = None

                    '''
                        find the variable with max distance from random variable
                    '''
                    for tuple in hop.keys():
                        if((tuple[0].getId()) == (variables[indexRemove].getId())):
                            if(hop[tuple] > maxHop):
                                maxHop = hop[tuple]
                                '''
                                    far variable
                                '''
                                v = tuple[1]
                    
                    '''
                        there is a variable with max hop
                    '''
                    if(v != None):
                        '''
                           add the variable more far
                        '''
                        variablesToRemove.append(v)
                        
                        while( (len(variablesToRemove) < varSet) & ((len(variables) - len(variablesToRemove)) > 0) ):
                             
                            inter = list()
                            '''
                                find the intersection
                            '''
                            for j in range(1,len(variablesToRemove)):
                                if(j == 1):
                                    inter = list(set(graph[variablesToRemove[j-1]]).intersection(set(graph[variablesToRemove[j]])) )
                                else:
                                    inter = list(set(inter).intersection(set(graph[variablesToRemove[j]])))
                            
                            liste = list()
                            
                            '''
                                intersection not empty
                            '''
                            if(len(inter) > 0):
                            
                                k = 0
                                
                                while(k < len(variablesToRemove)):
                                    
                                    list1 = list()
                                    
                                    for n in inter:
                                        list1.append(hop[(variablesToRemove[k],n)])
                                    
                                    liste.append(list1)
                                    
                                    k = k + 1

                                '''
                                    sum the distances between variables
                                '''   
                                content = [sum(x) for x in zip(*liste)]
                                
                                maxHopIndex = content.index(max(content))

                                '''
                                    add the variable more far
                                '''
                                variablesToRemove.append(inter[maxHopIndex])
                            else:
                                '''
                                    remaining variables to choose
                                '''
                                diff = list(set(variables) - set(variablesToRemove))
                                
                                while( (len(variablesToRemove) < varSet) & (len(diff) > 0) ):
                                     '''
                                         choose random variables
                                     '''
                                     indexRemove = random.randint(0, len(diff) - 1)
                                    
                                     '''
                                         add the random variable
                                     '''
                                     variablesToRemove.append(diff[indexRemove])
                                     
                                     diff.remove(diff[indexRemove])
                                     
                    else:                        
                        '''
                            remaining variables to choose
                        '''
                        diff = list(set(variables) - set(variablesToRemove))
                        
                        while( ((len(variablesToRemove) < varSet) & (len(diff) > 0) ) ):
                                 '''
                                    choose random variables
                                 '''
                                 indexRemove = random.randint(0, len(diff) - 1)
                                    
                                 '''
                                    add the random variable
                                 '''
                                 variablesToRemove.append(diff[indexRemove])
                                     
                                 diff.remove(diff[indexRemove])
                                              
                    
                    for j in range(len(variablesToRemove)):     
                        '''
                            state of variable i
                        '''
                        state = variablesToRemove[j].getStateArgument()
                         
                        print('state of variable:', variablesToRemove[j].toString(), state.getValue())
                        
                        index = getIndex(originalCop,variablesToRemove[j])
                        '''
                            save the value of the variable
                        '''
                        finalAssignaments[index] = state.getValue()
                         
                        oldAssignaments[index] = finalAssignaments[index]
                    
                    
                    for var in variablesToRemove:
                        '''
                            remove the variable on the agent
                        '''
                        agent.removeNodeVariable(var)
                     
                    for var in variablesToRemove:
                        '''
                            neighbours of variable  
                        '''
                        neighbours = var.getNeighbour()
                         
                        for function in neighbours:
                            functions.remove(function)
                            agent.removeNodeFunction(function)
                                                                                 
                        '''
                            remove the variable from the list
                        '''
                        variables.remove(var)
                             
                        for variable in variables:
                             
                            n1 = set(neighbours)
                            n2 = set(variable.getNeighbour())
                             
                            if((len(set(n1).intersection(n2))) > 0 ):
                                
                                '''
                                    intersection of neighbours
                                '''
                                inter = (set(n1).intersection(n2))
                                
                                while(len(inter) > 0):
                                
                                    func = inter.pop()
                                    
                                    '''
                                        list of the arguments of the function
                                    '''
                                    argumentsOfFunction = list()
                                     
                                    funId = 0
                    
                                    '''
                                        there is at least a function
                                    '''
                                    if((len(functions)) > 0):
                                         funId = ((functions[len(functions) - 1]).getId()) + 1   
                                    else:
                                         funId = 0
                                    
                                    '''
                                        creates an unary function linked to the variable
                                    '''
                                    nodefunction = NodeFunction(funId)
                                     
                                    functionevaluator = TabularFunction()  
                                    nodefunction.setFunction(functionevaluator) 
                                     
                                    '''
                                        add this nodefunction to the actual nodevariable's neighbour
                                    '''
                                    variable.addNeighbour(nodefunction)         
                                    '''
                                        add this variable as nodefunction's neighbour
                                    '''
                                    nodefunction.addNeighbour(variable)
                                     
                                    '''
                                        add the variable as an argument of the function
                                    '''    
                                    argumentsOfFunction.append(variable)
                                    
                                    costTable = (func.getFunction()).getCostTable()
                                    
                                    cost = 0
                                    
                                    '''
                                        state of variableToRemove
                                    '''
                                    state = var.getStateArgument()
                                    
                                    index = (func.getFunction()).getParameterPosition(var)
                                        
                                    '''
                                        create the unary function
                                    '''      
                                    for j in range(0, variable.size()):
                                        
                                        t = (j)
                                        
                                        if(index == 0):     
                                            cost = costTable[(state.getValue(),j)]
                                        else:
                                            cost = costTable[(j,(state.getValue()))]
                                        '''
                                            add to the cost function: [parameters -> cost]
                                        '''
                                        nodefunction.getFunction().addParametersCost(t, cost) 
                                        
                                    '''
                                        add the neighbors of the function node
                                    '''
                                    nodefunction.getFunction().setParameters(argumentsOfFunction)
                                     
                                    '''
                                        add the function node
                                    '''
                                    functions.append(nodefunction) 
                                     
                                    '''
                                        add the function node to the agent
                                    '''
                                    agent.addNodeFunction(nodefunction)
        
                                variable.removeNeighbours(neighbours)
                                 
  
                    if((len(variables) > 0) & (len(functions) > 0)):
                    
                        agents = list() 
                             
                        agents.append(agent)    
                                 
                        cop = COP_Instance(variables, functions, agents)
                             
                        i = i + varSet
 
                     
                if((len(variables) > 0) & (len(functions) > 0)):
                    '''
                        create new MaxSum instance (max/min)
                    '''           
                    core = MaxSum(cop, op) 
                    '''
                        update only at end?
                    '''
                    core.setUpdateOnlyAtEnd(False)    
         
                    core.setIterationsNumber(nIterations)
                             
                    start_time = time.time()          
                                                                     
                    '''
                        invoke the method that executes the MaxSum algorithm
                    '''
                    core.solve_complete()
                    
                    values = core.getValues()

                    elapse_time = time.time() - start_time
                    print('MaxSum elapse_time:', elapse_time)  
                     
                    going = True 
                     
                    if(i == 0):
                        '''
                             create a copy of cop
                        '''
                        originalCop = deepcopy(core.getCop())
                         
                        oldVariables = originalCop.getNodeVariables()
                         
                        for k in range(len(oldVariables)):
                            oldAssignaments[k] = oldVariables[k].getStateIndex()
                                         
                                
                    actualValue = calculateActualValue(originalCop,oldAssignaments) 
                    
                    averageValues.append(actualValue)
                    
                    iterations.append(len(values))
                     
            if((len(variables) > 0) & (len(functions) == 0)):
                '''
                    the variables of the cop
                '''
                variables = core.getCop().getNodeVariables()

                '''
                    remaining variables to set
                '''
                index = (nVariables - i) - 1
                
                k = 0   
                
                j = 1

                while(j < ((index / varSet) + 1)):   
                
                    while(k < (varSet * j)):
                        '''
                            state of variable i
                        '''
                        state = (variables[k]).getStateArgument() 
                        '''
                            save the value of the variable
                        '''
                        finalAssignaments[i] = state.getValue()
                         
                        oldAssignaments[i] = finalAssignaments[i]

                        i = i + k
                        
                        k = k + 1
                        
                    actualValue = calculateActualValue(originalCop,oldAssignaments)
                         
                    averageValues.append(actualValue)
                        
                    iterations.append(len(values))
                    
                    j = j + 1
 
 
            print('\nAssegnamenti MaxSum')
 
            for k in range(len(finalAssignaments)):
                print('x:', k, 'val:', finalAssignaments[k])
 
 
            for i in range(len(iterations)):
                if(i > 0):
                    iterations[i] = iterations[i] + iterations[i-1]
                    
            averageValuesInstances.append(averageValues)
            iterationsInstances.append(iterations)
             
            # draw the chart 
            # x axis: number of MaxSum exec
            # y axis: conflicts
            x = iterations
             
            '''
                x axis: number of MaxSum exec
            '''
            y = averageValues
            pl.title('Cost / Iterations chart')
            pl.xlabel('Iterations')
            pl.ylabel('Cost')
            pl.plot(x, y)
            #pl.show()
            pl.savefig(chartsFile + "MaxSumParallelMaxHopRandom/min/Charts/Chart_RUN_" + str(run) + ".png")
            pl.close()
      
    sumIterations = [sum(x) for x in zip(*iterationsInstances)] 
    sumValues = [sum(x) for x in zip(*averageValuesInstances)] 
    
    for i in range(len(sumIterations)):
        sumIterations[i] = sumIterations[i] / nIstances
        
    for i in range(len(sumValues)):
        sumValues[i] = sumValues[i] / nIstances
    
    # draw the chart 
    # x axis: number of MaxSum exec
    # y axis: conflicts
    x = sumIterations
             
    '''
        x axis: number of MaxSum exec
    '''
    y = sumValues
    pl.title('Cost / Iterations chart')
    pl.xlabel('Iterations')
    pl.ylabel('Cost')
    pl.plot(x, y)
    #pl.show()
    pl.savefig(chartsFile + "MaxSumParallelMaxHopRandom/min/Charts/AverageAllInstances.png")
    pl.close()    
    
    string = 'Iteration\tConflict\n'
    
    for i in range(len(sumIterations)):
        string = string + str(sumIterations[i]) + '\t\t' + str(sumValues[i]) + '\n'
            
    output_file = open(infoGraphPathFile + "MaxSumParallelMaxHopRandom/min/FactorGraph/reportIterations.txt", "w")
    output_file.write(string)
    output_file.write("\n")
    output_file.close()
    
# finds shortest path between 2 nodes of a graph using BFS
def bfs_shortest_path(graph, start, goal):
    # keep track of explored nodes
    explored = []
    # keep track of all the paths to be checked
    queue = [[start]]
 
    # return path if start is goal
    if start == goal:
        return "That was easy! Start = goal"
 
    # keeps looping until all possible paths have been checked
    while queue:
        # pop the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        if node not in explored:
            neighbours = graph[node]
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # return path if neighbour is goal
                if neighbour == goal:
                    return new_path
 
            # mark node as explored
            explored.append(node)
 
    # in case there's no path between the 2 nodes
    return []
    
def getIndex(originalCop, variable):
    variables = originalCop.getNodeVariables()
    
    for i in range(len(variables)):
        if(((variables[i]).getId()) == variable.getId()):
            return i
            
    return -1  
    
def calculateActualValue(originalCop, oldAssignaments):
     
    variables = originalCop.getNodeVariables()
     
    for i in range(len(variables)):
        (variables[i]).setStateIndex(oldAssignaments[i])
         
    return originalCop.actualValue()
                         
                                                        
def create_DCop(infoGraphPathFile, nVariables, run):
    '''
        infoGraphPathFile: location where saving the factor graph
        nVariables: how many variables are there in Dcop instance?
        run: number of Dcop instance 
    '''
    
    '''
        configurations of variables's values in a function (10 values in domain):
        0 0, 0 1, 0 2, 1 0 ....
    '''
    arguments = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
                 (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
                 (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
                 (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
                 (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
                 (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
                 (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
                 (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9),
                 (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9),
                 (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9),          
                ]
    
    '''
        list of variable in Dcop
    '''
    nodeVariables = list()
    '''
        list of function in Dcop
    '''
    nodeFunctions = list()
    '''
        list of agents in Dcop
        In this case there is only one agent
    '''
    agents = list()
    
    '''
        agent identifier
    '''    
    agent_id = 0
    '''
        variable identifier
    '''
    variable_id = 0
    '''
        function identifier
    '''
    function_id = 0
    '''
        variable identifier as parameter in the function
    '''
    variable_to_function = 0
    
    '''
        list of arguments in the function
    '''
    argumentsOfFunction = list()
    
    '''
       each variable has 10 values in its domain (0..9) 
    '''
    number_of_values = 10    
    
    '''
        only one agent controls all the variables and functions
    '''
    agent = Agent(agent_id)  
    
    nodeVariable = None
    nodefunction = None
    functionEvaluator = None     
    '''
        activation probability (random) to create an edge
    '''
    p = None
     
    '''
        create nVariables dcop variables
    '''
    for j in range(nVariables):
        '''
            create new NodeVariable with variable_id
        '''     
        nodeVariable = NodeVariable(variable_id)
        '''
            create the variable's domain with 10 values (0...9)
        '''
        nodeVariable.addIntegerValues(number_of_values)
        '''
            append this variable to Dcop's list of variable
        '''
        nodeVariables.append(nodeVariable)
            
        '''
           add the variable under its control
        '''
        agent.addNodeVariable(nodeVariable)
            
        variable_id = variable_id + 1  
  
    
    '''
        for each variable in Dcop's list
    '''
    for j in range(0, len(nodeVariables)):        
        '''
            for each variable close to variable j with bigger id 
        '''        
        for u in range(j+1, len(nodeVariables)):   
            '''
                generate a random value to enable the edge (if p <= 0.6)
            '''                
            p = random.random()    
            
            '''
                if activation prabability is less than 0.6
            '''    
            if (p < 0.6):    
                    '''
                        if you can create an edge between the two node
                    '''
                                           
                    '''
                       list of the arguments of the function 
                       each function is binary
                    '''          
                    argumentsOfFunction = list()
                                
                    nodefunction = NodeFunction(function_id)
                                
                    functionEvaluator = TabularFunction()  
                    nodefunction.setFunction(functionEvaluator)
                                
                    '''
                        add the function_id function to the neighbors 
                        of variable_to_function
                    '''    
                    for v in range(len(nodeVariables)):
                        if (((nodeVariables[v].getId()) == nodeVariables[j].getId())
                        
                            | ((nodeVariables[v].getId()) == nodeVariables[u].getId()) ):
                            '''
                                add this nodefunction to the actual nodevariable's neighbour
                            '''
                            nodeVariables[v].addNeighbour(nodefunction)
                                        
                            '''
                                add this variable as nodefunction's neighbour
                            '''
                            nodefunction.addNeighbour(nodeVariables[v])
                            '''
                                add the variable as an argument of the function
                            '''      
                            argumentsOfFunction.append(nodeVariables[v])
              
                    '''
                        add the function parameters
                    '''
                    nodefunction.getFunction().setParameters(argumentsOfFunction)         
                                
                    for tuple in arguments:
                        '''
                            generate a random uniform cost between 1 and 10
                        '''
                        cost = random.randint(1, 10)
                        '''
                            add to the cost function: [parameters -> cost]
                        '''        
                        nodefunction.getFunction().addParametersCost(tuple, cost)
                                
                    '''
                        add the function node
                    '''
                    nodeFunctions.append(nodefunction) 
                                
                    '''
                        add the function node to the agent
                    '''
                    agent.addNodeFunction(nodefunction)
                
                    '''
                        update the id of the next function node
                    '''
                    function_id = function_id + 1                              

    '''
        there is only one agent in this Dcop
    '''    
    agents.append(agent)
          
    string = ""         
    
    '''
        create the COP: list of variables, list of functions, agents
    '''                
    cop = COP_Instance(nodeVariables, nodeFunctions, agents)
    
    string = string + "How many agents?" + str(len(agents)) + "\n"
    
    '''
        create the factor graph report
    '''
    for agent in agents:
            string = string + "\nAgent Id: " + str(agent.getId()) + "\n\n"
            string = string + "How many NodeVariables?" + str(len(agent.getVariables())) + "\n"
            for variable in agent.getVariables():
                string = string + "Variable: " + str(variable.toString()) + "\n"
                
            string = string + "\n"
            
            for function in agent.getFunctions():
                string = string + "Function: " + str(function.toString()) + "\n"
                
            string = string + "\n"    
    
    for variable in nodeVariables:
            string = string + "Variable: " + str(variable.getId()) + "\n"
            for neighbour in variable.getNeighbour():
                string = string + "Neighbour: " + str(neighbour.toString()) + "\n"
            string = string + "\n"
    
    for function in nodeFunctions:
            string = string + "\nFunction: " + str(function.getId()) + "\n"
            string = string + "Parameters Number: " + str(function.getFunction().parametersNumber()) + "\n"
            for param in function.getFunction().getParameters():
                string = string + "parameter:" + str(param.toString()) + "\n"
                
            string = string + "\n\tCOST TABLE\n"
            
            string = string + str(function.getFunction().toString()) + "\n" 
    
    string = string + "\t\t\t\t\t\t\tFACTOR GRAPH\n\n" + str(cop.getFactorGraph().toString())
    
    '''info_graph_file = open(infoGraphPathFile + "MaxSumLowestGrade/FactorGraph/factor_graph_run_" + str(run) + ".txt", "w")
    info_graph_file.write(string)
    info_graph_file.write("\n")
    info_graph_file.close()'''   
    
    return cop  


def getParser():
    '''
        This is the Parser that takes the parameters of Command Line
    '''
    parser = argparse.ArgumentParser(description="MaxSum-Algorithm")
    
    parser.add_argument("-iterations", metavar='iterations', type=int,
                        help="number of iterations")
    
    parser.add_argument("-instances", metavar='instances', type=int,
                        help="number of instances in Dcop")
    
    parser.add_argument("-variables", metavar='variables', type=int,
                        help="number of variables in Dcop")
    
    parser.add_argument("-op", metavar='op',
                        help="operator (max/min)")
    
    parser.add_argument("-reportMaxSum", metavar='reportMaxSum',
                        help="FILE of reportMaxSum")
    
    parser.add_argument("-reportFactorGraph", metavar='reportFactorGraph',
                        help="FILE of reportFactorGraph")
    
    parser.add_argument("-reportCharts", metavar='reportCharts',
                        help="FILE of reportCharts")
    
    
    args = parser.parse_args()

    '''
        All parameters ARE REQUIRED!!
        if the parameters are correct
    '''
    if  ((args.iterations > 0 & (not(args.iterations == None))) & 
        (args.instances > 0 & (not(args.instances == None))) & 
        (args.variables > 0 & (not(args.variables == None))) & 
        (not(args.op == None) & ((args.op == 'max') | (args.op == 'min'))) & (not(args.reportMaxSum == None)) & 
        (not(args.reportFactorGraph == None)) & (not(args.reportCharts == None))):
        
        return args
    else:
        printUsage()
        sys.exit(2)
        
        

def printUsage():
    
    description = '\n----------------------------------- MAX SUM ALGORITHM ---------------------------------------\n\n'
    
    description = description + 'This program is a testing about NotColored Graphs, 3-colorability.\nThe colored Graphs'
    description = description + ' have 10 and many more variables.\nThe aim is to analyze the average of the rmessages'
    description = description + 'differences which tends to 0,\nand that the number of conflicts tends to 0.\n'
    description = description + 'The results are: report about the average of the rmessages differences,\n' 
    description = description + 'factor graph of Dcop, the charts about the average of the rmessages differences\n'
    description = description + 'and the conflicts during the iterations.\n'
    description = description + 'The results are saved as chart (.png) and as log file (.txt)\n'
    
    usage = 'All parameters ARE REQUIRED!!\n\n'
    
    usage = usage + 'Usage: python -iterations=Iter -instances=Inst -variables=V -op=O -reportMaxSum=reportM -reportFactorGraph=reportG -reportCharts=reportC [-h]\n\n'
    
    usage = usage + '-iterations Iter\tThe number of MaxSum iterations\n'
    usage = usage + '-instances Inst\t\tThe number of instances of Dcop to create\n'
    usage = usage + '-variables V\t\tThe number of variables in each instance\n'
    usage = usage + '-op O\t\t\tmax/min (maximization or minimization of conflicts)\n'
    usage = usage + '-reportMaxSum reportM\t\tFILE where writing the report of the MaxSum execution (FILE location with final /)\n'
    usage = usage + '-reportFactorGraph reportG\tFILE where writing the factorGraph and information about MaxSum execution (FILE location with final /)\n'
    usage = usage + '-reportCharts reportC\t\tFILE where saving the average of the rmessagesdifferences and the number of conflicts of the MaxSum execution\n\t\t\t\t(FILE location with final /)\n'
    usage = usage + '-h help\tInformation about parameters\n'
    
    print(description)
    
    print(usage)
    

    
if __name__ == '__main__':
    main()
