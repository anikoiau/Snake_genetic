import pygame
from pygame import Vector2 
import random
from nn import initialize, predict
import numpy as np


GEN = 100

block = 10

HEIGHT = 350
WIDTH = 350

params = None


W1 = np.genfromtxt('W1_multilayer.csv', delimiter = ',')
b1 = np.genfromtxt('b1_multilayer.csv', delimiter = ',')
W2 = np.genfromtxt('W2_multilayer.csv', delimiter = ',')
b2 = np.genfromtxt('b2_multilayer.csv', delimiter = ',')

b2 = np.reshape(b2, ((3, 1)))
b1 = np.reshape(b1, ((16, 1)))

params = {'W1' : W1, 'b1' : b1, 'W2' : W2, 'b2' : b2}


walls = [[0, 0, WIDTH - 1, 0], [0, 0, 0, HEIGHT - 1], [WIDTH - 1, 0, WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1, WIDTH - 1, HEIGHT - 1]]

SCREEN = (WIDTH, HEIGHT)


food = [None, None, None]



class Snake():
    def __init__(self, pos, brain = None):
        self.pos = Vector2(pos)
        self.born = Vector2(pos)
        self.v = Vector2(1, 0)
        self.w = block
        self.h = block
        self.length = 4
        self.guide = Vector2(self.v)
        self.body = [self.pos, self.pos - self.v * block, self.pos - self.v * block * 2, self.pos - self.v * block * 3]
        self.score = 0
        self.life = 0
        self.time_limit = 150
        self.alive = True
        self.dirs = [Vector2(0, -1), Vector2(1, -1), Vector2(1, 0), Vector2(1, 1), Vector2(1, 0), Vector2(-1, 1), Vector2(-1, 0), Vector2(-1, -1)]
        
        if brain == None:
            self.brain = initialize([26, 16, 3])
        else:
            self.brain = brain
            
        
    def update(self): 
        
        self.life += 1
        self.time_limit -= 1
                
        if(self.length == len(self.body)):
            for i in range(self.length - 1):
                self.body[i] = self.body[i + 1]
                
            self.body[-1] = Vector2(self.pos)
                        
        else:
            while len(self.body) != self.length:
                self.body.append(Vector2(self.pos))
        
        self.pos = self.pos + (self.v * block)
        

    def draw(self, screen):        
        for i in range(self.length):
            pygame.draw.rect(screen, (0, 255, 0), (self.body[i].x - block // 2, self.body[i].y - block // 2, self.w, self.h), 1)
        
    def move(self, d):
        self.v = Vector2(d)
        
    def eat(self, food):

        if food.distance_to(self.pos) <= block + 3:
            self.length += 1
            self.time_limit += 50
            self.score += 150
            
            if(self.time_limit > 500):
                self.time_limit = 500
                
            return True
        else: 
            return False
            
    def collision(self):
        for b in self.body:
            if b.distance_to(self.pos) <= 2 and b != self.pos:
                # del self.body[:]
                # self.body = [self.pos]
                # self.length = 1
                return True
            
        return False
                
                
    def intersect(self, ray, line):
        x1 = line[0]
        y1 = line[1]
        # end point
        x2 = line[2]
        y2 = line[3]
        
        #position of the ray
        x3 = ray.x
        y3 = ray.y
        x4 = ray.x + ray.x
        y4 = ray.y + ray.y
    
        #denominator
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        #numerator
        num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        if den == 0:
            return None
        
        #formulas
        t = num / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    
        if t > 0 and t < 1 and u > 0:
            #Px, Py
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            pot = Vector2(x, y)
            return pot
        
    
    def look(self, direction, food):
        inp = [0, 0, 0]
        
        lookat = Vector2(self.pos)
        
        foodfound = False
        bodyfound = False
        
        dist = 1
        
        lookat += direction * block
        
        while not(lookat.x < 0 or lookat.x > WIDTH or lookat.y < 0 or lookat.y > HEIGHT):
            
            if (not foodfound) and lookat.distance_to(food) <= 10:
                inp[0] = 1
                foodfound = True
                
            if (not bodyfound) and lookat in self.body and lookat != self.pos:
                inp[1] = 1 / dist
                bodyfound = True
                
            lookat += direction * block
            dist += 1
            
            
        inp[2] = 1 / dist
        
        return inp[0], inp[1], inp[2]
    
    
                
    def processDistances(self, foods, walls, screen):
                     
        inp = [0 for i in range(26)]
        
        # front = self.v * block
        # left = Vector2(self.v.rotate(-90)) * block
        # right = Vector2(self.v.rotate(90)) * block
        # fl = Vector2(self.v.rotate(-45)) * block
        # fr = Vector2(self.v.rotate(45)) * block
        
        # f = self.pos + front
        # l = self.pos + left
        # r = self.pos + right
        
        
        m = 100000000
        f = None
        for food in foods:
            if self.pos.distance_to(food) < m:
                m = self.pos.distance_to(food)
                f = food
                
        
        inp[0], inp[1], inp[2] = self.look(Vector2(0, -1), f)
        
        inp[3], inp[4], inp[5] = self.look(Vector2(1, -1), f)
        
        inp[6], inp[7], inp[8] = self.look(Vector2(1, 0), f)
        
        inp[9], inp[10], inp[11] = self.look(Vector2(1, 1), f)
        
        inp[12], inp[13], inp[14] = self.look(Vector2(0, 1), f)
        
        inp[15], inp[16], inp[17] = self.look(Vector2(-1, 1), f)
        
        inp[18], inp[19], inp[20] = self.look(Vector2(-1, 0), f)
        
        inp[21], inp[22], inp[23] = self.look(Vector2(-1, -1), f)
        
        foodv = f - self.pos
    
        if foodv.magnitude() != 0:
            angle = np.arccos(foodv.dot(self.v) / (foodv.magnitude() * self.v.magnitude())) * 180 / np.pi
            inp[24] = np.cos(angle * np.pi / 180)
        
        else:
            inp[24] = 1
            
        if self.time_limit == 0:
            inp[25] = 1
        
        else:
            inp[25] = 1 / self.time_limit
        
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################



        # m = 100000000
        # f = None
        # for food in foods:
        #     if self.pos.distance_to(food) < m:
        #         m = self.pos.distance_to(food)
        #         f = food
         
        # for i in range(1, 4):
        #     f = self.pos + front * i
        #     if f.x < 0 or f.x > WIDTH or f.y < 0 or f.y > HEIGHT:
        #         inputs[0] = 1
        #         break
        
        # for i in range(1, 4):
        #     l = self.pos + left * i
        #     if l.x < 0 or l.x > WIDTH or l.y < 0 or l.y > HEIGHT:
        #         inputs[1] = 1
        #         break
         
        # for i in range(1, 4):
        #     r = self.pos + right * i
        #     if r.x < 0 or r.x > WIDTH or r.y < 0 or r.y > HEIGHT:
        #         inputs[2] = 1
        #         break
            
            

        # #BODY IS FINE ########################################################
        # for i in range(1, 5):
        #     temp = Vector2(self.pos + front * i)
            
        #     if temp in self.body: 
        #         inputs[3] = 1
                
        #         break
                
                
        # for i in range(1, 8):
        #     temp = Vector2(f + left * i)
            
        #     if temp in self.body:
        #         inputs[4] = 1
                
        #         break
                  
                
        # for i in range(1, 5):
        #     temp = Vector2(f + right * i)
            
        #     if temp in self.body:
        #         inputs[5] = 1
           
        #         break
            
        # for i in range(1, 8):
        #     temp = self.pos + fl * i
            
        #     if temp in self.body:
        #         inputs[6] = 1
         
        #         break
            
        # for i in range(1, 8):
        #     temp = self.pos + fr * i
            
        #     if temp in self.body:
        #         inputs[7] = 1
            
        #         break
     
        
        # #FOOD IS FINE ########################################################
        # if f != None:
        #     foodv = f - self.pos
        
        #     if foodv.magnitude() != 0:
        #         angle = np.arccos(foodv.dot(self.v) / (foodv.magnitude() * self.v.magnitude())) * 180 / np.pi
        #         inputs[12] = np.cos(angle * np.pi / 180)
                
        #     else:
        #         angle = 0
            
        #     if angle == 0:
        #         inputs[8] = 1
        #         # print('food straight')
        #     else:
        #         if self.v == Vector2(0, -1):
        #             if food.x < self.pos.x:
        #                 inputs[9] = 1
        #                 # print('left')
        #             else:
        #                 inputs[10] = 1
        #                 # print('right')
                    
        #         elif self.v == Vector2(1, 0):
        #             if food.y < self.pos.y:
        #                 inputs[9] = 1 
        #                 # print('left')
        #             else:
        #                 inputs[10] = 1
        #                 # print('right')
                    
        #         elif self.v == Vector2(0, 1):
        #             if food.x > self.pos.x:
        #                 inputs[9] = 1
        #                 # print('left')
        #             else:
        #                 inputs[10] = 1
        #                 # print('right')
                    
        #         elif self.v == Vector2(-1, 0):
        #              if food.y > self.pos.y:
        #                 inputs[9] = 1
        #                 # print('left')
        #              else:
        #                 inputs[10] = 1
        #                 # print('right')
                        
                    
        # # if self.time_limit <= 200:
        # #     inputs[9] = 1
            
        # if self.time_limit != 0:
        #     inputs[11] = 1 / self.time_limit
        # else:
        #     inputs[11] = 1
            
        # # inputs[12] = self.length / 100
        
            
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################


        inputs = np.array(inp)

             
        m = len(inputs)
        
        # print(m)
        
        inputs = inputs.reshape((m, 1))
               
        return inputs
       


screen = pygame.display.set_mode(SCREEN)


snakes = [Snake(Vector2(block * 10, block * 10), brain = params) for i in range(GEN)]

prevscores = [snake.score for snake in snakes]
def foodloc():
    food = []
    for i in range(20):
        x = random.randint(1, WIDTH // block - 1)
        y = random.randint(1, HEIGHT // block - 1)
        
        food.append(Vector2(x, y) * block)

    return food


food = foodloc()

prev = Vector2(1, 0)

saved = []
    
#######################################################################################################################################################


def mutate(parameters, rate):

    
    # if np.random.random() < rate:         
    for (key, value) in parameters.items():
        m, n = value.shape
        
        for i in range(m):
            for j in range(n):
                if np.random.random() < rate:
                    r = (np.random.random() * 2 - 1) * rate * .1
                    
                    value[i][j] += r
                    
                    if value[i][j] < -1:
                        value[i][j] = -1
                        
                    if value[i][j] > 1:
                        value[i][j] = 1
                    
                
        parameters[key] = value
                    
            
    return parameters



def chooseParent(array, total):
    
    np.random.seed()
    
    r = np.random.randint(0, total)
    
    i = 0
    s = 0
    
    while s < r:
        s += array[i]
        i += 1
        
   
    return i - 1


    
def callNextGeneration(saved, GEN, rate):
    
    
    total_score = 0
    max_score = 0
    fitness_prob = []
    
    scores = []
    
    for snake in saved:
        
        
        total_score += snake.score
        
        scores.append(snake.score)

    
    
    lists = []
    
    
    for k in range(0, GEN):
        
        np.random.seed()
        
        par1_index = chooseParent(scores, total_score)
        par2_index = chooseParent(scores, total_score)
        
            
        while par1_index == par2_index:
            par2_index = chooseParent(scores, total_score)
   
        
        parent1 = saved[par1_index]
        
        parent2 = saved[par2_index]
 
        
        brain1 = parent1.brain
        brain2 = parent2.brain
        
        
        child_dict = {}
        
        for (key, value) in brain1.items():
            m, n = brain1[key].shape
            child = np.zeros_like(brain1[key])
            
            for i in range(m):
                for j in range(n):
                    
                    if np.random.random() < .5:
                        child[i][j] = brain1[key][i][j]
                        # print('1')
                    else:
                        child[i][j] = brain2[key][i][j]
                        # print('2')
            
            
                        
            child_dict[key] = child
       
        child_dict = mutate(child_dict, rate)
    
      
        
        lists.append(Snake(brain = child_dict, pos = Vector2(block * 16, block * 16)))   
    
    return lists
    


################################################################################################################################################################


def drawLine(snake, screen):
    f = snake.pos + snake.v * 50
    pygame.draw.line(screen, (255, 0, 0), snake.pos, f, 2)
    
    
play = True

generation = 0


best = snakes[0]

data = np.array(np.zeros((1, 35)))

val = 10

pygame.init()

rate = .3

while play == True:
    
    
    screen.fill((50, 50, 50))
    
    pygame.time.Clock().tick(val)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            play = False
            pygame.quit()
            
        # if event.type == pygame.MOUSEBUTTONDOWN:
        #     snakes[0].length += 1
   

   
    
        
        
    for f in food:   
        pygame.draw.rect(screen, (255, 0, 0), (int(f.x) - block // 2, int(f.y) - block // 2, int(block), int(block)), 1)
    
    for n, snake in enumerate(snakes):
        
        keys = pygame.key.get_pressed()
        

        if snake.alive == False:
            continue
    
        # pygame.draw.rect(screen, (255, 0, 0), (int(food[n].x) - block // 2, int(food[n].y) - block // 2, int(block), int(block)), 1)        
        
        # if keys[pygame.K_SPACE]:
        #     drawLine(snake, screen)
   
        if keys[pygame.K_q]:
            play = False
            
        if keys[pygame.K_x]:
            print('mutate_rate : ', rate)
            
        inp = snake.processDistances(food, walls, screen)
        outputs = predict(inp, snake.brain)
        
        
        # a = inp.T
        # a = a[0]
        
        # print(a[0], a[3], a[6], a[9], a[12], a[15], a[18], a[21])
        
        # if outputs == 0:
        #     snake.move(Vector2(0, -1))
            
        # elif outputs == 1:
        #     snake.move(Vector2(1, 0))  
        
        # elif outputs == 2:
        #     snake.move(Vector2(0, 1))
            
        # elif outputs == 3:
        #     snake.move(Vector2(-1, 0))
            
        
        if outputs == 0:
            snake.v.rotate_ip(-90)
            
        elif outputs == 1:
            snake.v.rotate_ip(90)
       
            
        
        # if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        #     snake.move(Vector2(-1, 0))
            
        # if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        #     snake.move(Vector2(1, 0))
            
        # if keys[pygame.K_w] or keys[pygame.K_UP]:
        #     snake.move(Vector2(0, -1))
            
        # if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        #     snake.move(Vector2(0, 1))  
        
       
        
        temp = snake.pos + snake.v * block
        
        
        if temp.x < 0 or temp.x > WIDTH or temp.y < 0 or temp.y > HEIGHT:
            if len(snake.body) < 10:
                snake.score = snake.life * 2 + (2 ** len(snake.body))
                
            else:
                snake.score = snake.life * 2 + (2 ** 10) * (len(snake.body) - 9)
                
            saved.append(snake)
            snake.alive = False
            
            
            # print('wall crash')
            continue
        
        if snake.time_limit <= 0:
            if len(snake.body) < 10:
                snake.score = snake.life * 2 + (2 ** len(snake.body))
                
            else:
                snake.score = snake.life * 2 + (2 ** 10) * (len(snake.body) - 9)
                
            saved.append(snake)
            snake.alive = False
            

            # print('timelimit expired')
            continue
        
        if temp in snake.body:
            # print('collided with body', len(snake.body))  
            if len(snake.body) < 10:
                snake.score = snake.life * 2 + (2 ** len(snake.body))
                
            else:
                snake.score = snake.life * 2 + (2 ** 10) * (len(snake.body) - 9)
                
            pygame.draw.rect(screen, (255, 100, 255), (temp.x, temp.y, block, block))
            saved.append(snake)
            snake.alive = False
            

            continue
        
      
            
        
        mfood = 1000000000
        ind = None
        for i, f in enumerate(food):
            if snake.pos.distance_to(f) < mfood:
                mfood = snake.pos.distance_to(f)
                ind = i
                
        
        if keys[pygame.K_SPACE]:
            pygame.draw.line(screen, (255, 255, 255), snake.pos, food[ind])
            
            
        if snake.pos.distance_to(food[ind]) <= block:
            snake.length += 1
            snake.time_limit += 75
            
            if(snake.time_limit > 500):
                snake.time_limit = 500
                
            x = random.randint(1, WIDTH // block - 1)
            y = random.randint(1, HEIGHT // block - 1)
            
            food[ind] = (Vector2(x, y) * block)
            
            
        

        snake.update()
        snake.draw(screen)
    
    #.time_limit)
    
    if keys[pygame.K_r]:
        rate += .001
        if rate >= 1:
            rate = 1
            
    if keys[pygame.K_i]:
        rate -= .001
        if rate <= 0:
            rate = 0
        
    if keys[pygame.K_j]:
        val -= 1
            
    if keys[pygame.K_k]:
        val += 1
       
    flag = 1
    for snake in snakes:
        if snake.alive == True:
            flag = 0
            
    if flag == 1:
        # print('all dead')
        prevscores = [snake.score for snake in snakes]
        max_indx = np.argmax(np.array(prevscores))
        min_indx = np.argmin(np.array(prevscores))
        
        if best.score < snakes[max_indx].score:
            best = snakes[max_indx]
    
        else:
            saved[max_indx] = best
        
        print('generation : ', generation, ' global best score : ', best.score, ' current best : ', snakes[max_indx].score)

        del snakes[:]
        snakes = callNextGeneration(saved, GEN, rate)   
        generation += 1
        del saved[:]
        continue
    
    
    pygame.display.update()
    
    
    


# al = []

# for s in snakes:
#     if s.alive == True:
#         al.append(s)

# best = al[0]

# best_brain = best.brain

# best_brain_W1 = best_brain['W1']
# best_brain_b1 = best_brain['b1']
# best_brain_W2 = best_brain['W2']
# best_brain_b2 = best_brain['b2']
# # best_brain_W3 = best_brain['W3']
# # best_brain_b3 = best_brain['b3']


# np.savetxt('W1_multilayer.csv', best_brain_W1, delimiter = ',')
# np.savetxt('b1_multilayer.csv', best_brain_b1, delimiter = ',')

# np.savetxt('W2_multilayer.csv', best_brain_W2, delimiter = ',')
# np.savetxt('b2_multilayer.csv', best_brain_b2, delimiter = ',')

# np.savetxt('W3_multilayer.csv', best_brain_W3, delimiter = ',')
# np.savetxt('b3_multilayer.csv', best_brain_b3, delimiter = ',')


v1 = Vector2(0, -1)
v2 = Vector2(0, -1)

np.arccos(v1.dot(v2) / (v1.magnitude() * v2.magnitude())) * 180 / np.pi
    