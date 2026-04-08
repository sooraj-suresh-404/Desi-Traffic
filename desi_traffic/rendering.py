import pygame
import numpy as np

class DesiTrafficRenderer:
    def __init__(self, display_size=600):
        pygame.init()
        self.size = display_size
        self.screen = pygame.display.set_mode((self.size, self.size))
        pygame.display.set_caption("DesiTraffic Visualizer")
        self.font = pygame.font.SysFont("Arial", 18)
        self.clock = pygame.time.Clock()

    def render(self, state_dict):
        # Fill background
        self.screen.fill((200, 200, 200)) # Grey roads
        
        # Draw roads
        road_width = 100
        center = self.size // 2
        # Vertical Road
        pygame.draw.rect(self.screen, (100, 100, 100), (center - road_width//2, 0, road_width, self.size))
        # Horizontal Road
        pygame.draw.rect(self.screen, (100, 100, 100), (0, center - road_width//2, self.size, road_width))

        # Extract values
        ql = state_dict['queue_lengths']
        tw = state_dict['two_wheeler_density']
        amb = state_dict['ambulance_approaching']
        phase = state_dict['current_green_phase']

        # UI Text
        text_y = 10
        info_lines = [
            f"Phase: {phase}  (0:N/S, 1:E/W, 4:Red)",
            f"Queues -> N:{ql[0]} S:{ql[1]} E:{ql[2]} W:{ql[3]}",
            f"2-Wheel Dense -> N:{tw[0]}% S:{tw[1]}% E:{tw[2]}% W:{tw[3]}%",
            f"Ambulance -> N:{amb[0]} S:{amb[1]} E:{amb[2]} W:{amb[3]}"
        ]
        for line in info_lines:
            surface = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(surface, (10, text_y))
            text_y += 20

        # Draw generic queue representations (cars)
        self._draw_queues(ql, center, road_width)
        
        pygame.display.flip()
        self.clock.tick(10) # 10 FPS
        
        # Handle events so window doesn't freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def _draw_queues(self, ql, center, road_width):
        car_size = 10
        # North queue
        for i in range(min(ql[0], 20)):
            pygame.draw.rect(self.screen, (200, 0, 0), (center - road_width//4 - 5, center - road_width//2 - 20 - (i*15), car_size, car_size))
        # South queue
        for i in range(min(ql[1], 20)):
            pygame.draw.rect(self.screen, (0, 0, 200), (center + road_width//4 - 5, center + road_width//2 + 10 + (i*15), car_size, car_size))
        # East queue
        for i in range(min(ql[2], 20)):
            pygame.draw.rect(self.screen, (0, 200, 0), (center + road_width//2 + 10 + (i*15), center - road_width//4 - 5, car_size, car_size))
        # West queue
        for i in range(min(ql[3], 20)):
            pygame.draw.rect(self.screen, (200, 200, 0), (center - road_width//2 - 20 - (i*15), center + road_width//4 - 5, car_size, car_size))

    def close(self):
        pygame.quit()
