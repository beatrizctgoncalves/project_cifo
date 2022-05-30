import pygame
import random
from random import choice, sample, uniform
import os
import math
import numpy as np
import time
import sys
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PIL import Image
from operator import attrgetter
from matplotlib import pyplot as plt

pygame.init()  # Initialize pygame
# Some variables initializations
img = 0  
size = width, height = 1600, 900  # Size to use when creating pygame window

n = 0

# Colors
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
black = (0, 0, 0)
gray = pygame.Color('gray12')
Color_line = (255, 0, 0)

checkpointFitness = 50
fitness_values=[]
generation_values=[]
generation = 1
mutationRate = 90
FPS = 30
selectedCars = []
selected = 0
lines = True  # If true then lines of player are shown
player = True  # If true then player is shown
display_info = True  # If true then display info is shown
frames = 0
maxspeed = 10
number_track = 1

white_small_car = pygame.image.load('Images/Sprites/white_small.png')
white_big_car = pygame.image.load('Images/Sprites/white_big.png')
green_small_car = pygame.image.load('Images/Sprites/green_small.png')
green_big_car = pygame.image.load('Images/Sprites/green_big.png')

bg = pygame.image.load('bg7.png')
bg4 = pygame.image.load('bg4.png')


def calculateDistance(x1, y1, x2, y2):  # Used to calculate distance between points
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


# Used to rotate points #rotate(origin, point, math.radians(10))
def rotation(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def move(point, angle, unit):  # Translate a point in a given direction
    x = point[0]
    y = point[1]
    rad = math.radians(-angle % 360)

    x += unit*math.sin(rad)
    y += unit*math.cos(rad)

    return x, y


def sigmoid(z):  # Sigmoid function, used as the neurons activation function
    return 1.0/(1.0+np.exp(-z))


########################################################### SELECTION ###########################################################
#################################################################################################################################
def tournament(population, size=10):

    for i in range(2):
        tournament = [choice(population) for i in range(size)]
        select=max(tournament, key=attrgetter("fitness"))
        selectedCars.append(select)
        population.remove(select)


def fitnessProportionateSelection(population):

    for i in range(2):
        # Sum total fitness
        total_fitness = sum([n.fitness for n in population])
        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness)
        position = 0
        # Find individual in the position of the spin
        for individual in population:
            position += individual.fitness
            if position > spin:
                selectedCars.append(individual)
                population.remove(individual)
                break

########################################################### CROSSOVERS ###########################################################
##################################################################################################################################
#Single Point Crossover
def singlePointCrossoverWeights(parent1, parent2, child1, child2):
    sizenn = len(child1.sizes)
    #pt = random.randint(1, parent1.sizes-2)

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]
                
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]
                
    #Copy parent biases into child biases
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child1.biases[i][j] = parent1.biases[i][j]
                
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child2.biases[i][j] = parent2.biases[i][j]
     
    genome1 = [] #This will be a list containing all weights of child1
    genome2 = [] #This will be a list containing all weights of child2
        
    for i in range(sizenn-1): #i=0,1
        for j in range(child1.sizes[i]*child1.sizes[i+1]):
            genome1.append(child1.weights[i].item(j))
            
    for i in range(sizenn-1): #i=0,1
        for j in range(child2.sizes[i]*child2.sizes[i+1]):
            genome2.append(child2.weights[i].item(j))

    pt = random.randint(1, len(parent1.sizes)-2)
    genome1_new=[]
    genome2_new=[]
    genome1_new = genome1[:pt] + genome2[pt:]
    genome2_new = genome2[:pt] + genome1[pt:]

    #Go back from genome list to weights numpy array on child object
    count = 0
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genome1_new[count]
                count += 1
          
    count = 0
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
            for k in range(child2.sizes[i]):
                child2.weights[i][j][k] = genome2_new[count]
                count += 1                        
    return


def singlePointCrossoverBiases(parent1, parent2, child1, child2):
    sizenn = len(parent1.sizes)

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]
                
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]
                
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child1.biases[i][j] = parent1.biases[i][j]
                
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child2.biases[i][j] = parent2.biases[i][j]
    
    genome1 = [] 
    genome2 = [] 
        
    for i in range(sizenn-1): 
        for j in range(child1.sizes[i+1]):
            genome1.append(child1.biases[i].item(j))
            
    for i in range(sizenn-1):
        for j in range(child2.sizes[i]*child2.sizes[i+1]):
            genome2.append(child2.weights[i].item(j))

    pt = random.randint(1, len(parent1.sizes)-2)
    genome1_new=[]
    genome2_new=[]
    genome1_new = genome1[:pt] + genome2[pt:]
    genome2_new = genome2[:pt] + genome1[pt:]
    
    count = 0    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
                child1.biases[i][j] = genome1_new[count]
                count += 1
                
    count = 0    
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child2.biases[i][j] = genome2_new[count]
                count += 1
    return 

# Uniform_CrossOver
def uniformCrossOverWeights(parent1, parent2, child1, child2): #Given two parent car objects, it modifies the children car objects weights
    sizenn = len(child1.sizes)
    
    #Copy parent weights into child weights
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]
                
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]
                
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child1.biases[i][j] = parent1.biases[i][j]
                
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child2.biases[i][j] = parent2.biases[i][j]

    genome1 = []
    genome2 = []
        
    for i in range(sizenn-1): 
        for j in range(child1.sizes[i]*child1.sizes[i+1]):
            genome1.append(child1.weights[i].item(j))
            
    for i in range(sizenn-1): 
        for j in range(child2.sizes[i]*child2.sizes[i+1]):
            genome2.append(child2.weights[i].item(j))
         
    alter = True    
    for i in range(len(genome1)):
        if alter == True:
            aux = genome1[i]
            genome1[i] = genome2[i]
            genome2[i] = aux
            alter = False
        else:
            alter = True

    count = 0
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genome1[count]
                count += 1
          
    count = 0
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
            for k in range(child2.sizes[i]):
                child2.weights[i][j][k] = genome2[count]
                count += 1              
    return 

def uniformCrossOverBiases(parent1, parent2, child1, child2): 
    sizenn = len(parent1.sizes)
    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]
                
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]
                
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child1.biases[i][j] = parent1.biases[i][j]
                
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child2.biases[i][j] = parent2.biases[i][j]
                
    genome1 = []
    genome2 = []
    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            genome1.append(child1.biases[i].item(j))
            
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
            genome2.append(child2.biases[i].item(j))
     
    alter = True    
    for i in range(len(genome1)):
        if alter == True:
            aux = genome1[i]
            genome1[i] = genome2[i]
            genome2[i] = aux
            alter = False
        else:
            alter = True
    
    count = 0    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
                child1.biases[i][j] = genome1[count]
                count += 1
                
    count = 0    
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child2.biases[i][j] = genome2[count]
                count += 1
    return 

#Arithmetic CrossOver
def arithmeticCrossOverWeights(parent1, parent2, child1, child2):
    sizenn = len(parent1.sizes)
    alpha = uniform(0, 1)
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k] * \
                    alpha + (1 - alpha) * parent2.weights[i][j][k]
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
            for k in range(child2.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k] * \
                    alpha + (1 - alpha) * parent1.weights[i][j][k]
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
            child1.biases[i][j] = parent1.biases[i][j]

    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
            child2.biases[i][j] = parent2.biases[i][j]

    genome1 = []  
    genome2 = []  

    for i in range(sizenn-1):  
        for j in range(child1.sizes[i]): 
            genome1.append(child1.weights[i].item(j))

    for i in range(sizenn-1):  
        for j in range(child2.sizes[i]): 
            genome2.append(child2.weights[i].item(j))

    return


def arithmeticCrossOverBiases(parent1, parent2, child1, child2):
    sizenn = len(parent1.sizes)
    alpha = uniform(0, 1)
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]

    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
            child1.biases[i][j] = parent1.biases[i][j] * alpha + (1 - alpha) * parent2.biases[i][j]

    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
            child2.biases[i][j] = parent2.biases[i][j] * alpha + (1 - alpha) * parent1.biases[i][j]

    genome1 = []
    genome2 = []

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            genome1.append(child1.biases[i].item(j))

    for i in range(sizenn-1): 
        for j in range(child2.sizes[i+1]): 
            genome2.append(child2.biases[i].item(j))

    return


########################################################### MUTATIONS ###########################################################
#################################################################################################################################
##Swap_mutation
def swapMutationWeights(parent1, child1):
    sizenn = len(child1.sizes)

    # Copy parent weights into child weights
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    # Copy parent biases into child biases
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            child1.biases[i][j] = parent1.biases[i][j]

    genomeWeights = []  # This will be a list containing all weights, easier to modify this way

    for i in range(sizenn-1):  # i=0,1
        for j in range(child1.sizes[i]*child1.sizes[i+1]):
            genomeWeights.append(child1.weights[i].item(j))

    mut_points = sample(range(len(genomeWeights)), 2)
    genomeWeights[mut_points[0]], genomeWeights[mut_points[1]] = genomeWeights[mut_points[1]], genomeWeights[mut_points[0]]

    count = 0
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genomeWeights[count]
                count += 1
    return

def swapMutationBiases(parent1, child1):
    sizenn = len(child1.sizes)

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            child1.biases[i][j] = parent1.biases[i][j]

    genomeBiases = []

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            genomeBiases.append(child1.biases[i].item(j))

   # Modify a random gene by a random amount
    mut_points = sample(range(len(genomeBiases)), 2)
    # Swap them
    genomeBiases[mut_points[0]], genomeBiases[mut_points[1]] = genomeBiases[mut_points[1]], genomeBiases[mut_points[0]]

    count = 0
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            child1.biases[i][j] = genomeBiases[count]
            count += 1
    return

#Inversion_mutation
def inversionMutationWeights(parent1, child1):
    sizenn = len(child1.sizes)

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            child1.biases[i][j] = parent1.biases[i][j]
    
    genomeWeights = [] 

    for i in range(sizenn-1):  
        for j in range(child1.sizes[i]*child1.sizes[i+1]):
            genomeWeights.append(child1.weights[i].item(j))

    mut_points = sample(range(len(genomeWeights)), 2)
    mut_points.sort()
    genomeWeights[mut_points[0]:mut_points[1]] = genomeWeights[mut_points[0]:mut_points[1]][::-1]

    count = 0
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genomeWeights[count]
                count += 1

    return

def inversionMutationBiases(parent1, child1):
    sizenn = len(child1.sizes)

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            child1.biases[i][j] = parent1.biases[i][j]

    genomeBiases = []

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            genomeBiases.append(child1.biases[i].item(j))

    mut_points = sample(range(len(genomeBiases)), 2)
    mut_points.sort()
    genomeBiases[mut_points[0]:mut_points[1]] = genomeBiases[mut_points[0]:mut_points[1]][::-1]

    count = 0
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            child1.biases[i][j] = genomeBiases[count]
            count += 1
    return

##Mutate One Weight
def mutateOneWeightGene(parent1, child1):
    sizenn = len(child1.sizes)
    
    #Copy parent weights into child weights
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]          
                
    #Copy parent biases into child biases
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
                child1.biases[i][j] = parent1.biases[i][j]

    genomeWeights = [] #This will be a list containing all weights, easier to modify this way
    
    for i in range(sizenn-1): #i=0,1
        for j in range(child1.sizes[i]*child1.sizes[i+1]):
            genomeWeights.append(child1.weights[i].item(j))
    
    #Modify a random gene by a random amount
    r1 = random.randint(0, len(genomeWeights)-1)
    genomeWeights[r1] = genomeWeights[r1]*random.uniform(0.8, 1.2)
    
    count = 0
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genomeWeights[count]
                count += 1
    return


def mutateOneBiasesGene(parent1, child1):
    sizenn = len(child1.sizes)
    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]          
                
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
                child1.biases[i][j] = parent1.biases[i][j]

    genomeBiases = []
    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            genomeBiases.append(child1.biases[i].item(j))
            
    r1 = random.randint(0, len(genomeBiases)-1)
    genomeBiases[r1] = genomeBiases[r1]*random.uniform(0.8, 1.2)
    
    count = 0    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
                child1.biases[i][j] = genomeBiases[count]
                count += 1
    return

# # # # Class of our Cars # # # # 

class Car:
    def __init__(self, sizes):
        self.fitness = 0
        self.num_layers = len(sizes)  # Number of nn layers
        self.sizes = sizes  # List with number of neurons per layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # Biases
        self.weights = [np.random.randn(y, x) for x, y in zip(
            sizes[:-1], sizes[1:])]  # Weights
        # c1, c2, c3, c4, c5 are five 2D points where the car could collided, updated in every frame
        self.c1 = 0, 0
        self.c2 = 0, 0
        self.c3 = 0, 0
        self.c4 = 0, 0
        self.c5 = 0, 0
        # d1, d2, d3, d4, d5 are distances from the car to those points, updated every frame too and used as the input for the NN
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        self.d4 = 0
        self.d5 = 0
        self.yaReste = False
        # The input and output of the NN must be in a numpy array format
        self.inp = np.array(
            [[self.d1], [self.d2], [self.d3], [self.d4], [self.d5]])
        self.outp = np.array([[0], [0], [0], [0]])
        # Boolean used for toggling distance lines
        self.showlines = False
        # Initial location of the car
        self.x = 120
        self.y = 480
        self.center = self.x, self.y
        # Height and width of the car
        self.height = 35  # 45
        self.width = 17  # 25
        # These are the four corners of the car, using polygon instead of rectangle object, when rotating or moving the car, we rotate or move these
        self.d = self.x-(self.width/2), self.y-(self.height/2)
        self.c = self.x + self.width-(self.width/2), self.y-(self.height/2)
        # El rectangulo está centrado en (x,y)
        self.b = self.x + self.width - \
            (self.width/2), self.y + self.height-(self.height/2)
        self.a = self.x-(self.width/2), self.y + self.height - \
            (self.height/2)
        self.velocity = 0
        self.acceleration = 0
        self.angle = 180
        # Boolean which goes true when car collides
        self.collided = False
        # Car color and image
        self.color = white
        self.car_image = white_small_car

    def set_accel(self, accel):
        self.acceleration = accel

    def rotate(self, rot):
        self.angle += rot
        if self.angle > 360:
            self.angle = 0
        if self.angle < 0:
            self.angle = 360 + self.angle

    def update(self): 
        self.fitness += self.velocity
        if self.checkpoint():
            self.fitness += checkpointFitness
        if self.acceleration != 0:
            self.velocity += self.acceleration
            if self.velocity > maxspeed:
                self.velocity = maxspeed
            elif self.velocity < 0:
                self.velocity = 0
        else:
            self.velocity *= 0.92

        self.x, self.y = move((self.x, self.y), self.angle, self.velocity)
        self.center = self.x, self.y

        self.d = self.x-(self.width/2), self.y-(self.height/2)
        self.c = self.x + self.width-(self.width/2), self.y-(self.height/2)
        self.b = self.x + self.width - \
            (self.width/2), self.y + self.height-(self.height/2)
        self.a = self.x-(self.width/2), self.y + self.height - \
            (self.height/2) 

        self.a = rotation((self.x, self.y), self.a, math.radians(self.angle))
        self.b = rotation((self.x, self.y), self.b, math.radians(self.angle))
        self.c = rotation((self.x, self.y), self.c, math.radians(self.angle))
        self.d = rotation((self.x, self.y), self.d, math.radians(self.angle))

        self.c1 = move((self.x, self.y), self.angle, 10)
        while bg4.get_at((int(self.c1[0]), int(self.c1[1]))).a != 0:
            self.c1 = move((self.c1[0], self.c1[1]), self.angle, 10)
        while bg4.get_at((int(self.c1[0]), int(self.c1[1]))).a == 0:
            self.c1 = move((self.c1[0], self.c1[1]), self.angle, -1)

        self.c2 = move((self.x, self.y), self.angle+45, 10)
        while bg4.get_at((int(self.c2[0]), int(self.c2[1]))).a != 0:
            self.c2 = move((self.c2[0], self.c2[1]), self.angle+45, 10)
        while bg4.get_at((int(self.c2[0]), int(self.c2[1]))).a == 0:
            self.c2 = move((self.c2[0], self.c2[1]), self.angle+45, -1)

        self.c3 = move((self.x, self.y), self.angle-45, 10)
        while bg4.get_at((int(self.c3[0]), int(self.c3[1]))).a != 0:
            self.c3 = move((self.c3[0], self.c3[1]), self.angle-45, 10)
        while bg4.get_at((int(self.c3[0]), int(self.c3[1]))).a == 0:
            self.c3 = move((self.c3[0], self.c3[1]), self.angle-45, -1)

        self.c4 = move((self.x, self.y), self.angle+90, 10)
        while bg4.get_at((int(self.c4[0]), int(self.c4[1]))).a != 0:
            self.c4 = move((self.c4[0], self.c4[1]), self.angle+90, 10)
        while bg4.get_at((int(self.c4[0]), int(self.c4[1]))).a == 0:
            self.c4 = move((self.c4[0], self.c4[1]), self.angle+90, -1)

        self.c5 = move((self.x, self.y), self.angle-90, 10)
        while bg4.get_at((int(self.c5[0]), int(self.c5[1]))).a != 0:
            self.c5 = move((self.c5[0], self.c5[1]), self.angle-90, 10)
        while bg4.get_at((int(self.c5[0]), int(self.c5[1]))).a == 0:
            self.c5 = move((self.c5[0], self.c5[1]), self.angle-90, -1)

        self.d1 = int(calculateDistance(
            self.center[0], self.center[1], self.c1[0], self.c1[1]))
        self.d2 = int(calculateDistance(
            self.center[0], self.center[1], self.c2[0], self.c2[1]))
        self.d3 = int(calculateDistance(
            self.center[0], self.center[1], self.c3[0], self.c3[1]))
        self.d4 = int(calculateDistance(
            self.center[0], self.center[1], self.c4[0], self.c4[1]))
        self.d5 = int(calculateDistance(
            self.center[0], self.center[1], self.c5[0], self.c5[1]))

    def drawCheckpoints(self):
        pygame.draw.line(gameDisplay,(0,250, 0, 255),(95.93604263824119, 86.12440191387562), (190.7819029737751, 190.7819029737751), 5)
        #pygame.draw.line(gameDisplay,(0,250, 0),(263.8241172551633, 260.5535703470414), (404.4576343044032, 262.73393495245597), 2)
        pygame.draw.line(gameDisplay,(0,250, 0, 255),(335.77614923384414, 563.624250499667), (464.41766095330394, 525.4678699049119), 5)
        #pygame.draw.line(gameDisplay,(0,250, 0),(573.4358912240325, 422.99073345042706), (681.3639391920539, 457.8765671370602), 2)
        pygame.draw.line(gameDisplay,(0,250, 0, 255),(853.6127430198051, 184.24080915753137), (929.9255042093151, 301.98049784991827), 5)
        #pygame.draw.line(gameDisplay,(0,250, 0),(1255.8900127187937, 39.246562897462304), (1256.980195021501, 170.06843922233665), 2)
        pygame.draw.line(gameDisplay,(0,250, 0, 255),(1408.5155350978139, 279.08666949306524), (1541.5177760281026, 280.17685179577256), 5)
        #pygame.draw.line(gameDisplay,(0,250, 0),(1219.9139967294532, 453.5158379262311), (1264.611471140452, 575.6162558294471), 2)
        pygame.draw.line(gameDisplay,(0,250, 0, 255),(1096.72339652353, 721.7006843922235), (1097.8135788262373, 849.252013808976), 5)
        #pygame.draw.line(gameDisplay,(0,250, 0),(700.987220640785, 748.9552419599056), (702.0774029434923, 823.0876385440012), 2)
        pygame.draw.line(gameDisplay,(0,250, 0, 255),(341.22706074738056, 719.520319786809), (341.22706074738056, 849.252013808976), 5)
        #pygame.draw.line(gameDisplay,(0,250, 0),(188.6015383683605, 609.411907213373), (55.599297438071595, 609.411907213373), 2)


    def draw(self, display):
        rotated_image = pygame.transform.rotate(
            self.car_image, -self.angle-180)
        rect_rotated_image = rotated_image.get_rect()
        rect_rotated_image.center = self.x, self.y
        gameDisplay.blit(rotated_image, rect_rotated_image)

        center = self.x, self.y
        if self.showlines:
            pygame.draw.line(gameDisplay, Color_line,
                             (self.x, self.y), self.c1, 2)
            pygame.draw.line(gameDisplay, Color_line,
                             (self.x, self.y), self.c2, 2)
            pygame.draw.line(gameDisplay, Color_line,
                             (self.x, self.y), self.c3, 2)
            pygame.draw.line(gameDisplay, Color_line,
                             (self.x, self.y), self.c4, 2)
            pygame.draw.line(gameDisplay, Color_line,
                             (self.x, self.y), self.c5, 2)

    def showLines(self):
        self.showlines = not self.showlines

    def feedforward(self):
        # Return the output of the network
        self.inp = np.array([[self.d1], [self.d2], [self.d3], [
                            self.d4], [self.d5], [self.velocity]])
        for b, w in zip(self.biases, self.weights):
            self.inp = sigmoid(np.dot(w, self.inp)+b)
        self.outp = self.inp
        return self.outp

    def collision(self):
        if (bg4.get_at((int(self.a[0]), int(self.a[1]))).a == 0) or (bg4.get_at((int(self.b[0]), int(self.b[1]))).a == 0) or (bg4.get_at((int(self.c[0]), int(self.c[1]))).a == 0) or (bg4.get_at((int(self.d[0]), int(self.d[1]))).a == 0):
            return True
        else:
            return False

    def checkpoint(self):
        if (gameDisplay.get_at((int(self.a[0]), int(self.a[1]))) == (0,250,0,255)) or (gameDisplay.get_at((int(self.b[0]), int(self.b[1]))) == (0,250,0,255)) or (gameDisplay.get_at((int(self.c[0]), int(self.c[1]))) == (0,250,0,255)) or (gameDisplay.get_at((int(self.d[0]), int(self.d[1]))) == (0,250,0,255)):
            #print("Checkpoint!!")
            return True
        else:
            return False

    def resetPosition(self):
        self.x = 120
        self.y = 480
        self.angle = 180
        return

    def takeAction(self):
        if self.outp.item(0) > 0.5:  # Accelerate
            self.set_accel(0.2)
        else:
            self.set_accel(0)
        if self.outp.item(1) > 0.5:  # Brake
            self.set_accel(-0.2)
        if self.outp.item(2) > 0.5:  # Turn right
            self.rotate(-5)
        if self.outp.item(3) > 0.5:  # Turn left
            self.rotate(5)
        return


nnCars = []  # List of neural network cars
num_of_nnCars = 50  # Number of neural network cars
alive = num_of_nnCars  # Number of not collided (alive) cars
collidedCars = []  # List containing collided cars

# These is just the text being displayed on pygame window
infoX = 1365
infoY = 600
font = pygame.font.Font('freesansbold.ttf', 18)
text1 = font.render('0..9 - Change Mutation', True, white)
text2 = font.render('LMB - Select/Unselect', True, white)
text3 = font.render('RMB - Delete', True, white)
text4 = font.render('L - Show/Hide Lines', True, white)
text5 = font.render('R - Reset', True, white)
text6 = font.render('B - Breed', True, white)
text7 = font.render('C - Clean', True, white)
#text8 = font.render('N - Next Track', True, white)
text9 = font.render('A - Toggle Player', True, white)
text10 = font.render('D - Toggle Info', True, white)
#text11 = font.render('M - Breed and Next Track', True, white)
text1Rect = text1.get_rect().move(infoX, infoY)
text2Rect = text2.get_rect().move(infoX, infoY+text1Rect.height)
text3Rect = text3.get_rect().move(infoX, infoY+2*text1Rect.height)
text4Rect = text4.get_rect().move(infoX, infoY+3*text1Rect.height)
text5Rect = text5.get_rect().move(infoX, infoY+4*text1Rect.height)
text6Rect = text6.get_rect().move(infoX, infoY+5*text1Rect.height)
text7Rect = text7.get_rect().move(infoX, infoY+6*text1Rect.height)
#text8Rect = text8.get_rect().move(infoX, infoY+7*text1Rect.height)
text9Rect = text9.get_rect().move(infoX, infoY+8*text1Rect.height)
text10Rect = text10.get_rect().move(infoX, infoY+9*text1Rect.height)
#text11Rect = text11.get_rect().move(infoX, infoY+10*text1Rect.height)


def displayTexts():
    infotextX = 20
    infotextY = 600
    infotext1 = font.render('Gen ' + str(generation), True, white)
    infotext2 = font.render('Cars: ' + str(num_of_nnCars), True, white)
    infotext3 = font.render('Alive: ' + str(alive), True, white)
    infotext4 = font.render('Selected: ' + str(selected), True, white)
    if lines == True:
        infotext5 = font.render('Lines ON', True, white)
    else:
        infotext5 = font.render('Lines OFF', True, white)
    if player == True:
        infotext6 = font.render('Player ON', True, white)
    else:
        infotext6 = font.render('Player OFF', True, white)
    #infotext7 = font.render('Mutation: '+ str(2*mutationRate), True, white)
    #infotext8 = font.render('Frames: ' + str(frames), True, white)
    infotext9 = font.render('FPS: 30', True, white)
    infotext1Rect = infotext1.get_rect().move(infotextX, infotextY)
    infotext2Rect = infotext2.get_rect().move(
        infotextX, infotextY+infotext1Rect.height)
    infotext3Rect = infotext3.get_rect().move(
        infotextX, infotextY+2*infotext1Rect.height)
    infotext4Rect = infotext4.get_rect().move(
        infotextX, infotextY+3*infotext1Rect.height)
    infotext5Rect = infotext5.get_rect().move(
        infotextX, infotextY+4*infotext1Rect.height)
    infotext6Rect = infotext6.get_rect().move(
        infotextX, infotextY+5*infotext1Rect.height)
    infotext9Rect = infotext9.get_rect().move(
        infotextX, infotextY+6*infotext1Rect.height)

    gameDisplay.blit(text1, text1Rect)
    gameDisplay.blit(text2, text2Rect)
    gameDisplay.blit(text3, text3Rect)
    gameDisplay.blit(text4, text4Rect)
    gameDisplay.blit(text5, text5Rect)
    gameDisplay.blit(text6, text6Rect)
    gameDisplay.blit(text7, text7Rect)
    gameDisplay.blit(text9, text9Rect)
    gameDisplay.blit(text10, text10Rect)

    gameDisplay.blit(infotext1, infotext1Rect)
    gameDisplay.blit(infotext2, infotext2Rect)
    gameDisplay.blit(infotext3, infotext3Rect)
    gameDisplay.blit(infotext4, infotext4Rect)
    gameDisplay.blit(infotext5, infotext5Rect)
    gameDisplay.blit(infotext6, infotext6Rect)
    gameDisplay.blit(infotext9, infotext9Rect)
    return


gameDisplay = pygame.display.set_mode(size)  # creates screen
clock = pygame.time.Clock()

inputLayer = 6
hiddenLayer = 6
outputLayer = 4
car = Car([inputLayer, hiddenLayer, outputLayer])
auxcar = Car([inputLayer, hiddenLayer, outputLayer])

for i in range(num_of_nnCars):
    nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))


def redrawGameWindow():  # Called on very frame

    global alive
    global frames
    global img

    frames += 1

    gameD = gameDisplay.blit(bg, (0, 0))
    
    # NN cars
    for nncar in nnCars:
        if not nncar.collided:
            nncar.update()  # Update: Every car center coord, corners, directions, collision points and collision distances
        if nncar.collision():  # Check which car collided
            nncar.collided = True  # If collided then change collided attribute to true
            if nncar.yaReste == False:
                alive -= 1
                nncar.yaReste = True
        else:  # If not collided then feedforward the input and take an action
            nncar.feedforward()
            nncar.takeAction()
        nncar.drawCheckpoints()
        nncar.draw(gameDisplay)

    # Same but for player
    if player:
        car.update()
        if car.collision():
            car.resetPosition()
            car.update()
        car.drawCheckpoints()
        car.draw(gameDisplay)

    if display_info:
        displayTexts()
    pygame.display.update()


while True:
    #now1 = time.time()

    for event in pygame.event.get():  # Check for events
        if event.type == pygame.QUIT:
            plt.plot(generation_values, fitness_values, linewidth=2.5, color='blue')
            plt.xticks(np.arange(1, len(generation_values)+1))
            plt.title('Fitness vs Generation', fontsize=15)
            plt.xlabel("Generation", fontsize=10)
            plt.ylabel("Fitness", fontsize=10)
            plt.show()
            fitness_values=[]
            generation_values=[]
            pygame.quit()  #quits
            quit()
        if event.type == pygame.KEYDOWN:  # If user uses the keyboard
            if event.key == ord("l"):  # If that key is l
                car.showLines()
                lines = not lines
            if event.key == ord("c"):  # If that key is c
                for nncar in nnCars:
                    if nncar.collided == True:
                        nnCars.remove(nncar)
                        if nncar.yaReste == False:
                            alive -= 1
            if event.key == ord("a"):  # If that key is a
                player = not player
            if event.key == ord("d"):  # If that key is d
                display_info = not display_info

            # # #  Condition that inicializes a new generation # # # 
            if event.key == ord("b"):
                ### Apply Selection ###
                #tournament(nnCars)
                fitnessProportionateSelection(nnCars)
                if (len(selectedCars) == 2):
                    fitness_values.append(max(selectedCars[0].fitness, selectedCars[1].fitness))
                    generation_values.append(generation)
                    print("Fitness_Values=", fitness_values)
                    print("Generation_Values=", generation_values)
                    for nncar in nnCars:
                        nncar.fitness = 0

                    alive = num_of_nnCars
                    generation += 1
                    selected = 0
                    nnCars.clear()

                    for i in range(num_of_nnCars):
                        nnCars.append(
                            Car([inputLayer, hiddenLayer, outputLayer]))

                    ### Apply CrossOver ###
                    for i in range(0, num_of_nnCars-2, 2):
                        uniformCrossOverWeights(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i+1])
                        uniformCrossOverBiases(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i+1])

                        #arithmeticCrossOverBiases(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i+1])
                        #arithmeticCrossOverWeights(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i+1])

                        #singlePointCrossoverWeights(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i+1])
                        #singlePointCrossoverBiases(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i+1])

                    nnCars[num_of_nnCars-2] = selectedCars[0]
                    nnCars[num_of_nnCars-1] = selectedCars[1]

                    nnCars[num_of_nnCars-2].car_image = green_small_car
                    nnCars[num_of_nnCars-1].car_image = green_small_car

                    nnCars[num_of_nnCars-2].resetPosition()
                    nnCars[num_of_nnCars-1].resetPosition()

                    nnCars[num_of_nnCars-2].collided = False
                    nnCars[num_of_nnCars-1].collided = False
                    
                    #print("First: ", selectedCars[0].weights)
                    mutate=True
                    ### Apply Mutation ###
                    for i in range(num_of_nnCars-2):
                        #if mutate ==True:
                                #print("Hello: ", nnCars[i].weights)
                        for j in range(mutationRate):
                            mutateOneWeightGene(nnCars[i], auxcar)
                            mutateOneWeightGene(auxcar, nnCars[i])
                            mutateOneBiasesGene(nnCars[i], auxcar)
                            mutateOneBiasesGene(auxcar, nnCars[i])

                            #swapMutationWeights(nnCars[i], auxcar)
                            #swapMutationWeights(auxcar, nnCars[i])
                            #swapMutationBiases(nnCars[i], auxcar)
                            #swapMutationBiases(auxcar, nnCars[i])
        
                            #inversionMutationWeights(nnCars[i], auxcar)
                            #inversionMutationWeights(auxcar, nnCars[i])
                            #inversionMutationBiases(nnCars[i], auxcar)
                            #inversionMutationBiases(auxcar, nnCars[i])

                        #if mutate ==True:
                            #print("Hi: ", nnCars[i].weights)
                            #mutate=False
                    if number_track != 1:
                        for nncar in nnCars:
                            nncar.x = 140
                            nncar.y = 610

                    selectedCars.clear()

            if event.key == ord("r"):
                generation = 1
                alive = num_of_nnCars
                nnCars.clear()
                selectedCars.clear()
                for i in range(num_of_nnCars):
                    nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))
                for nncar in nnCars:
                    if number_track == 1:
                        nncar.x = 120
                        nncar.y = 480
                    elif number_track == 2:
                        nncar.x = 100
                        nncar.y = 300
            if event.key == ord("0"):
                mutationRate = 0
            if event.key == ord("1"):
                mutationRate = 10
            if event.key == ord("2"):
                mutationRate = 20
            if event.key == ord("3"):
                mutationRate = 30
            if event.key == ord("4"):
                mutationRate = 40
            if event.key == ord("5"):
                mutationRate = 50
            if event.key == ord("6"):
                mutationRate = 60
            if event.key == ord("7"):
                mutationRate = 70
            if event.key == ord("8"):
                mutationRate = 80
            if event.key == ord("9"):
                mutationRate = 90

        if event.type == pygame.MOUSEBUTTONDOWN:
            # This returns a tuple:
            #(leftclick, middleclick, rightclick)
            # Each one is a boolean integer representing button up/down.
            mouses = pygame.mouse.get_pressed()
            if mouses[0]:
                pos = pygame.mouse.get_pos()
                point = Point(pos[0], pos[1])
                for nncar in nnCars:
                    polygon = Polygon([nncar.a, nncar.b, nncar.c, nncar.d])
                    if (polygon.contains(point)):
                        if nncar in selectedCars:
                            selectedCars.remove(nncar)
                            selected -= 1
                            if nncar.car_image == white_big_car:
                                nncar.car_image = white_small_car
                            if nncar.car_image == green_big_car:
                                nncar.car_image = green_small_car
                            if nncar.collided:
                                nncar.velocity = 0
                                nncar.acceleration = 0
                            nncar.update()
                        else:
                            if len(selectedCars) < 2:
                                selectedCars.append(nncar)
                                selected += 1
                                if nncar.car_image == white_small_car:
                                    nncar.car_image = white_big_car
                                if nncar.car_image == green_small_car:
                                    nncar.car_image = green_big_car
                                if nncar.collided:
                                    nncar.velocity = 0
                                    nncar.acceleration = 0
                                nncar.update()
                        break

            if mouses[2]:
                pos = pygame.mouse.get_pos()
                point = Point(pos[0], pos[1])
                for nncar in nnCars:
                    polygon = Polygon([nncar.a, nncar.b, nncar.c, nncar.d])
                    if (polygon.contains(point)):
                        if nncar not in selectedCars:
                            nnCars.remove(nncar)
                            alive -= 1
                        break

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car.rotate(-5)
    if keys[pygame.K_RIGHT]:
        car.rotate(5)
    if keys[pygame.K_UP]:
        car.set_accel(0.2)
    else:
        car.set_accel(0)
    if keys[pygame.K_DOWN]:
        car.set_accel(-0.2)

    redrawGameWindow()

    clock.tick(FPS)
