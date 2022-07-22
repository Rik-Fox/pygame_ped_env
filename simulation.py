import random
import time
import threading
import pygame
import sys


speeds = {
    "car": 2.25,
    "bus": 1.8,
    "truck": 1.8,
    "bike": 2.5,
}  # average speeds of vehicles

# Coordinates of vehicles' start
x = {
    "right": [0, 0, 0],
    "down": [755, 727, 697],
    "left": [1400, 1400, 1400],
    "up": [602, 627, 657],
}
y = {
    "right": [348, 370, 398],
    "down": [0, 0, 0],
    "left": [498, 466, 436],
    "up": [800, 800, 800],
}

vehicleTypes = {0: "car", 1: "bus", 2: "truck", 3: "bike"}
directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}

# Gap between vehicles
# movingGap = 15  # moving gap
# spacings = {0: -movingGap, 1: -movingGap, 2: movingGap, 3: movingGap}

pygame.init()
simulation = pygame.sprite.Group()


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.crossed = 0
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        self.rect = self.image.get_rect()
        self.rect.x = x[direction][lane]
        self.rect.y = y[direction][lane]

        simulation.add(self)

    # def get_rect(self):
    #     return self.image.get_rect()

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        if self.direction == "right":
            self.rect.x += self.speed  # move the vehicle
        elif self.direction == "down":
            self.rect.y += self.speed
        elif self.direction == "left":
            self.rect.x -= self.speed
        elif self.direction == "up":
            self.rect.y -= self.speed


class RL_Vehicle(Vehicle):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        super().__init__(lane, vehicleClass, direction_number, direction)
        # self.crossed = 1


# Generating vehicles in the simulation
def generateVehicles():
    while True:
        vehicle_type = random.randint(0, 3)
        # lane_number = random.randint(1, 2)

        lane_number = 1
        temp = random.randint(0, 99)
        direction_number = 0
        # dist = [25, 50, 75, 100]
        # if temp < dist[0]:
        #     direction_number = 0
        # elif temp < dist[1]:
        #     direction_number = 1
        # elif temp < dist[2]:
        #     direction_number = 2
        # elif temp < dist[3]:
        #     direction_number = 3
        dist = 50
        if temp < 50:
            direction_number = 0
        else:
            direction_number = 2
        Vehicle(
            lane_number,
            vehicleTypes[vehicle_type],
            direction_number,
            directionNumbers[direction_number],
        )
        time.sleep(2)


def step_vehicles(screen):
    for vehicle in simulation:
        old_x = vehicle.rect.x
        old_y = vehicle.rect.y
        vehicle.move()
        # remove vehicles after they drive offscreen
        if not (
            -vehicle.rect.width
            <= vehicle.rect.x
            <= screen.get_width() + vehicle.rect.width
        ):
            simulation.remove(vehicle)
        if not (
            -vehicle.rect.height
            <= vehicle.rect.y
            <= screen.get_height() + vehicle.rect.height
        ):
            simulation.remove(vehicle)
        # check if vehicles collided
        collided = pygame.sprite.spritecollide(vehicle, simulation, False)
        if len(collided) > 1:  # if they are move them back

            vehicle.rect.x = old_x
            vehicle.rect.y = old_y
            # or destroy them if collision at spawn
            if (
                old_x == 0.0
                or old_y == 0.0
                or old_x == screen.get_width()
                or old_y == screen.get_height()
            ):
                pygame.sprite.spritecollide(vehicle, simulation, True)

        screen.blit(vehicle.image, [vehicle.rect.x, vehicle.rect.y])


class Main:

    # Setting background image i.e. image of intersection
    background = pygame.image.load("images/intersection.png")

    screen = pygame.display.set_mode((1400, 800))
    pygame.display.set_caption("SIMULATION")

    # Loading signal images and font
    redSignal = pygame.image.load("images/signals/red.png")
    yellowSignal = pygame.image.load("images/signals/yellow.png")
    greenSignal = pygame.image.load("images/signals/green.png")
    font = pygame.font.Font(None, 30)

    # add RL agent
    # simulation.add(RL_Vehicle(1, "bike", 3, "up"))

    # background thread to generate traffic
    thread2 = threading.Thread(
        name="generateVehicles", target=generateVehicles, args=()
    )
    thread2.daemon = True
    thread2.start()

    # init vehicle sprites
    for vehicle in simulation:
        screen.blit(vehicle.image, [vehicle.rect.x, vehicle.rect.y])

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (0, 0))  # display background in simulation

        step_vehicles(screen)
        # display the vehicles

        pygame.display.update()


if __name__ == "__main__":
    Main()
