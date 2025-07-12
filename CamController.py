from bereshit import Vector3
class CamController:
    def __init__(self, player):
        self.player = player

    def main(self):
        # Get the player's current position
        player_pos = self.player.position

        # Calculate new camera position: 10 cm behind player
        new_cam_pos = player_pos + Vector3(0,1.0,- 0.2)

        # Update this camera's transform
        self.parent.position = new_cam_pos
