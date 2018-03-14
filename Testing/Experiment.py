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

import time

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
        create a directory for reports if it isn't exist
    '''
    '''directory = os.path.dirname(reportMaxSum + "Colored_Graph/TestingColoring/10Variabili/")
    if not os.path.exists(directory):
        os.mkdir(directory, 777)'''
    
    '''
        location of FactorGraph report
    '''
    infoGraphPathFile = args.reportFactorGraph
    '''
        create a directory for reports if it isn't exist
    '''
    '''directory = os.path.dirname(reportMaxSum + "Colored_Graph/FactorGraph/10Variabili/")
    if not os.path.exists(directory):
        os.mkdir(directory, 777)'''

        
    '''
        location of Charts
    '''
    chartsFile = args.reportCharts    
    '''
        create a directory for charts if it isn't exist
    '''
    ''''directory = os.path.dirname(reportMaxSum + "Colored_Graph/Charts/10Variabili/")
    if not os.path.exists(directory):
        os.mkdir(directory,777)''' 
    
    '''
        Constraint Optimization Problem
    '''
    cop = None  
    
    '''
        average of 50 instances of problem
    '''
    average = list()
    
    for i in range(nIterations):
        average.append(0)
    
    for run in range(nIstances): 
            '''
                fileName of log (Iteration Conflicts AverageDifferenceLink)
                save on file Iteration Conflicts AverageDifferenceLink
            '''                        
            fileOutputReport = reportMaxSum + "Experiment/min/Reports/Experiment_Report_RUN_" + str(run) + ".txt"
            '''
                values of MaxSum for each iteration
            '''
            values = list()    
          
            '''
                create a new COP with a colored Graph and 
                save the factorgraph on file
            '''
            cop = create_DCop(infoGraphPathFile, nVariables, run)  
                     
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
            
            elapse_time = time.time() - start_time
            print('MaxSum elapse_time:', elapse_time)  
            
            '''
                values of MaxSum for each iteration in this instance
            '''
            values = core.getValues()
            
            final = "\tITERATION\tCOST\n"              
            
            for i in range(len(values)):
                final = final + "\t" + str(i) + "\t\t" + str(values[i]) + "\n"
                average[i] = average[i] + values[i]
                
            '''
                save on file the log file
            '''
            core.stringToFile(final, fileOutputReport)
                
            
            # draw the chart 
            # x axis: number of iterations
            # y axis: max-cost found
            x = list()
            
            '''
                x axis: iterations
            '''
            for i in range(nIterations):
                x.append(i)
        
            #pl.xticks([20 * k for k in range(0, 16)])
            
            y = values
            pl.title('Cost / Iteration chart')
            pl.xlabel('Iterations')
            pl.ylabel('Cost')
            pl.plot(x, y)
            #pl.show()
            pl.savefig(chartsFile + "Experiment/min/Charts/Chart_RUN_" + str(run) + ".png")
            pl.close()
     
     
    for i in range(nIterations):
        average[i] = average[i] / nIstances        
    
    # draw the chart 
    # x axis: number of iterations
    # y axis: average of costs
    x = list()
            
    '''
        x axis: iterations
    '''
    for i in range(nIterations):
        x.append(i)
        
    #pl.xticks([20 * k for k in range(0, 16)])
            
    y = average
    pl.title('Average Cost / Iteration chart')
    pl.xlabel('Iterations')
    pl.ylabel('Cost')
    pl.plot(x, average)
    
    #pl.show()
    pl.savefig(chartsFile + "Experiment/min/Charts/AverageAllCharts.png")
    pl.close()
    
    string = 'Iteration\tConflict\n'
    
    for k in range(nIterations):
        string = string + str(k) + '\t\t' + str(average[k]) + '\n'
            
    output_file = open(infoGraphPathFile + "Experiment/min/Reports/reportAverageAllInstances.txt", "w")
    output_file.write(string)
    output_file.write("\n")
    output_file.close()
            
                                                        
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
    
    '''info_graph_file = open(infoGraphPathFile + "Experiment/FactorGraph/factor_graph_run_" + str(run) + ".txt", "w")
    info_graph_file.write(string)
    info_graph_file.write("\n")
    info_graph_file.close() '''   
    
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
