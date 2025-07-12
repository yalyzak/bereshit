tension_force = (
                        self.rigidbody.force.reduce_vector_along_direction(
                            self.joint.other.position.direction_vector(self.position)) *-1
                        if joint is not None
                        else Vector3(0, 0, 0)
                    )


# if self.get_component("joint") != None:
        #     if self.joint.look_position:
        #         # 4.4) Integrate position
        #         self.joint.other.position += self.rigidbody.velocity * dt \
        #                          + 0.5 * self.rigidbody.acceleration * dt * dt
        #
        #         # self.set_position(self.position)
        #     if self.joint.look_rotation:
        #         self.joint.other.add_rotation(ang_disp)
