# import the pygame module
import pygame

import tensorflow as tf
import numpy as np
from scipy import misc
import random
import math

eigenvalues = np.load("eigenvalues.npy")
eigenvectors = np.load("eigenvectors.npy")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_COUNT = 13014
DENSE_SIZE = 300
learning_rate = 0.0002  # Used to be 0.001
settings = np.zeros((DENSE_SIZE))
denseData = np.load("denseArray.npy")

f = open('allNames.txt', 'r+')
allNames = f.read()
f.close()
allPeople = allNames.split('\n')
nearestPerson = 0

famous_people = []
for i in range(len(allPeople)-1):
    parts = allPeople[i].split(",")
    rank = int(parts[1])
    if(rank < 7):
        famous_people.append(i)
# ~2000 famous people

meanData = denseData.mean(axis=0)

inputs_ = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='targets')




### Encoder
conv0 = tf.layers.conv2d(inputs=inputs_, filters=120, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 64x64x25
maxpool0 = tf.layers.max_pooling2d(conv0, pool_size=(2,2), strides=(2,2), padding='same')
# Now 32x32x25
conv1 = tf.layers.conv2d(inputs=maxpool0, filters=160, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 32x32x40
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
# Now 16x16x40
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=200, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 16x16x60
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
# Now 8x8x60
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=240, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 8x8x80
maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
# Now 4x4x80

maxpool3_flat = tf.reshape(maxpool3, [-1,4*4*240])

W_fc1 = weight_variable([4*4*240, 300])
b_fc1 = bias_variable([300])
tesy = tf.matmul(maxpool3_flat, W_fc1)
encoded = tf.nn.relu(tf.matmul(maxpool3_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([300, 4*4*240])
b_fc2 = bias_variable([4*4*240])
predecoded_flat = tf.nn.relu(tf.matmul(encoded, W_fc2) + b_fc2)

predecoded = tf.reshape(predecoded_flat, [-1,4,4,240])

### Decoder
upsample1 = tf.image.resize_images(predecoded, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 8x8x80
conv4 = tf.layers.conv2d(inputs=upsample1, filters=200, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 8x8x60
upsample2 = tf.image.resize_images(conv4, size=(16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 16x16x60
conv5 = tf.layers.conv2d(inputs=upsample2, filters=160, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 16x16x40
upsample3 = tf.image.resize_images(conv5, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 32x32x40
conv6 = tf.layers.conv2d(inputs=upsample3, filters=120, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 32x32x25
upsample4 = tf.image.resize_images(conv6, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 64x64x25
conv7 = tf.layers.conv2d(inputs=upsample4, filters=15, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 64x64x10


logits = tf.layers.conv2d(inputs=conv7, filters=3, kernel_size=(3,3), padding='same', activation=None)
#Now 64x64x1

# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)

# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess,  "model6528.ckpt")

def calculateImage():
    realSettings = meanData.copy()
    for i in range(DENSE_SIZE):
        s = eigenvalues[i]
        realSettings += settings[i]*s*eigenvectors[i]

    minDistance = 99999999999
    recordHolder = -1
    for pre_i in range(len(famous_people)):
        i = famous_people[pre_i]
        distance = np.linalg.norm(realSettings-denseData[i])
        if distance < minDistance:
            minDistance = distance
            recordHolder = i

    realSettings = realSettings.reshape((1,DENSE_SIZE))
    reconstructedImage = sess.run([decoded], feed_dict={encoded: realSettings})
    ri_np = np.array(reconstructedImage).reshape((64,64,3))
    ri_np = np.swapaxes(ri_np,0,1)
    return ri_np*255, recordHolder

X = 900  # screen width
Y = 600  # screen height

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 50)
BLUE = (50, 50, 255)
GREY = (200, 200, 200)
ORANGE = (200, 100, 50)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
TRANS = (1, 1, 1)
calculatedImage, nearestPerson = calculateImage()
# import pygame.locals for easier access to key coordinates
from pygame.locals import *

# Define our player object and call super to give it all the properties and methods of pygame.sprite.Sprite
# The surface we draw on the screen is now a property of 'player'
'''class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface((75, 25))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()'''

class Slider():
    def __init__(self, i, name, val, maxi, mini, xpos, ypos):
        self.val = val
        self.maxi = maxi
        self.mini = mini
        self.xpos = xpos
        self.ypos = ypos
        self.surf = pygame.surface.Surface((150,48))
        self.hit = False
        self.i = i

        self.font = pygame.font.SysFont("Helvetica", 16)
        self.txt_surf = self.font.render(name, 1, WHITE)
        self.txt_rect = self.txt_surf.get_rect(center = (75,13))

        s = 70
        if i%2+(i//2)%2 == 1:
            s = 100
        self.surf.fill((s,s,s))
        pygame.draw.rect(self.surf, (220,220,220), [10,28,130,5], 0)
        for g in range(7):
            pygame.draw.rect(self.surf, (s+50,s+50,s+50), [9+21.6666*g,40,2,5], 0)


        self.surf.blit(self.txt_surf, self.txt_rect)

        self.button_surf = pygame.surface.Surface((10,20))
        self.button_surf.fill(TRANS)
        self.button_surf.set_colorkey(TRANS)
        pygame.draw.rect(self.button_surf, WHITE, [0,0,10,20])

    def draw(self):
        surf = self.surf.copy()

        pos = (10+int((self.val-self.mini)/(self.maxi-self.mini)*130), 33)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)

        screen.blit(surf, (self.xpos, self.ypos))

    def move(self):
        self.val = (pygame.mouse.get_pos()[0] - self.xpos - 10) / 130 * (self.maxi - self.mini) + self.mini
        if self.val < self.mini:
            self.val = self.mini
        if self.val > self.maxi:
            self.val = self.maxi
        settings[self.i] = self.val
# initialize pygame
pygame.init()
slides = []
for i in range(20):
    eigen = "%.2f" % eigenvalues[i]
    slides.append(Slider(i,"PCA #"+str(i+1)+" ("+eigen+")",0,3,-3,(i%2)*150,(i//2)*48+60))

# create the screen object
# here we pass it a size of 800x600
screen = pygame.display.set_mode((800, 600))

# Variable to keep our main loop running
running = True

# Our main loop!
while running:
    # for loop through the event queue
    for event in pygame.event.get():
        # Check for KEYDOWN event; KEYDOWN is a constant defined in pygame.locals, which we imported earlier
        if event.type == KEYDOWN:
            # If the Esc key has been pressed set running to false to exit the main loop
            if event.key == K_ESCAPE:
                running = False
        # Check for QUIT event; if QUIT, set running to false
        elif event.type == QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            pos1 = (pos[0]-10,pos[1])
            pos2 = (pos[0]+10,pos[1])
            for s in slides:
                if s.button_rect.collidepoint(pos1) or s.button_rect.collidepoint(pos2) or s.button_rect.collidepoint(pos):
                    s.hit = True
            mouse_loc = pygame.mouse.get_pos()
            if mouse_loc[0] < 300 and mouse_loc[1] < 60:
                for i in range(DENSE_SIZE):
                    settings[i] = min(max(np.random.normal(0,1,1),-3),3)
                    if i < 20:
                        slides[i].val = settings[i]
                calculatedImage, nearestPerson = calculateImage()
            elif mouse_loc[0] < 300 and mouse_loc[1] >= 540:
                for i in range(DENSE_SIZE):
                    settings[i] = 0
                    if i < 20:
                        slides[i].val = settings[i]
                calculatedImage, nearestPerson = calculateImage()
        elif event.type == pygame.MOUSEBUTTONUP:
            for s in slides:
                s.hit = False

    for s in slides:
        if s.hit:
            s.move()
            calculatedImage, nearestPerson = calculateImage()
    screen.fill(BLACK)
    for s in slides:
        s.draw()
    randomizeButton = pygame.surface.Surface((300,60))
    pygame.draw.rect(randomizeButton, (230,30,30), [5,5,290,50], 0)
    font = pygame.font.SysFont("Helvetica", 44)
    rb_text = font.render("RANDOMIZE", 1, WHITE)
    rb_text_rect = rb_text.get_rect(center = (150,30))
    randomizeButton.blit(rb_text, rb_text_rect)
    screen.blit(randomizeButton, (0,0))

    normalizeButton = pygame.surface.Surface((300,60))
    pygame.draw.rect(normalizeButton, (30,230,30), [5,5,290,50], 0)
    nb_text = font.render("GO TO MEAN", 1, WHITE)
    nb_text_rect = nb_text.get_rect(center = (150,30))
    normalizeButton.blit(nb_text, nb_text_rect)
    screen.blit(normalizeButton, (0,540))

    image_surface = pygame.surfarray.make_surface(calculatedImage)
    bigger = pygame.transform.scale(image_surface,(500,500))
    screen.blit(bigger,(300,0))

    near = pygame.surface.Surface((500,100))
    font = pygame.font.SysFont("Helvetica", 24)
    ne_text = font.render("Hey! You look like "+allPeople[nearestPerson].split(",")[0]+".", 1, WHITE)
    ne_text_rect = ne_text.get_rect(center = (250,50))
    near.blit(ne_text, ne_text_rect)
    screen.blit(near,(300,500))
    #num += 2
    #wave(num)
    # Draw the player to the screen
    #screen.blit(player.surf, (400, 300))
    # Update the display
    pygame.display.flip()
