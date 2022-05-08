import pygame
import os
import math
import sys
import random
import neat

screen_width = 1400
screen_height = 800
generation = 0

##########################################################

class Car:
    def __init__(self):
        self.surface = pygame.image.load("car_diesel_1.png")
        self.rotate_surface = self.surface
        self.pos = [800, 730] #700, 650
        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 25, self.pos[1] + 25] #self.pos[0] + 50, self.pos[1] + 50
        pygame.draw.circle(self.surface, (0, 0, 255), (25,25), 3)
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.goal = False
        self.distance = 0
        self.time_spent = 0

    def draw(self, screen, map):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)
        self.drawCheckpoint(map)
        pygame.draw.line(screen, (0, 0, 255), (750, 687), (750, 775), 5)

    def draw_radar(self, screen):
        for r in self.radars:
            pos, dist = r
            pygame.draw.line(screen, (255, 0, 0), self.center, pos, 1)
            #pygame.draw.circle(screen, (255,0, 0), pos, 5)

    def drawCheckpoint(self, map):
        
        pygame.draw.circle(map, (0, 250, 0, 255), (1200, 230), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (970, 400), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (640,300), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (1240,730), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (930,100), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (600,400), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (165,400), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (1165,450), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (765,500), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (105,720), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (475,725), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (350,425), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (55,540), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (470,535), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (1325,530), 15)
        pygame.draw.circle(map, (0, 250, 0, 255), (280,240), 15)

    def checkpoint(self, map):
        for p in self.four_points:
            if map.get_at((int(p[0]), int(p[1]))) == (0, 250, 0, 255):
                print("checkpoint_____________")

    def check_collision(self, map):
        self.is_alive = True
        for p in self.four_points:
            if map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree, map):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not map.get_at((x, y)) == (255, 255, 255, 255) and len < 300:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, map):
        #check speed
        self.speed = 10

        #check position
        self.rotate_surface = self.rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
      

        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        

        # caculate 4 collision points
        self.center = [int(self.pos[0])+25, int(self.pos[1])+25] #[int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        #   print(self.center)
        len = 20
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, map)
        self.checkpoint(map)

    def get_data(self):
        radars = self.radars
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 10)

        return ret

    def get_alive(self):
        return self.is_alive

    def get_reward(self):
        return self.distance / 50.0

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image



##########################################################


def run_car(genomes, config):

    # Init NEAT
    nets = []
    cars = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        # Init my cars
        cars.append(Car())


    # Init my game
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()
    generation_font = pygame.font.SysFont("Arial", 20)
    font = pygame.font.SysFont("Arial", 15)
    timefont = pygame.font.SysFont("Arial", 15)
    map = pygame.image.load('track_6.png')


    ####### Main loop #######
    global generation
    generation += 1
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)


        pygame.time.delay(40)
        # Input my data and get result from network
        for index, car in enumerate(cars):
            output = nets[index].activate(car.get_data())
            pygame.time.delay(2)
            i = output.index(max(output))
            if i == 0:
                car.angle += 15
            else:
                car.angle -= 15

        # Update car and fitness
        remain_cars = 0
        for i, car in enumerate(cars):
            if car.get_alive():
                remain_cars += 1
                car.update(map)
                genomes[i][1].fitness += car.get_reward()

        # check
        if remain_cars == 0:
            break

        # Drawing
        screen.blit(map, (0, 0))
        for car in cars:
            if car.get_alive():
                car.draw(screen, map)

        text = generation_font.render("Generation: " + str(generation), True, (1, 1, 1))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/15, 50)
        screen.blit(text, text_rect)

        text = font.render("Cars on track: " + str(remain_cars), True, (1, 1, 1))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/15, 100)
        screen.blit(text, text_rect)

        text = timefont.render("Time Elapsed: " + str((pygame.time.get_ticks() - start_time)/1000), True, (1, 1, 1))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/15, 120)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(0)

##########################################################

# Run Main Function
if __name__ == "__main__":
    # Set configuration file
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    p.run(run_car, 1000)