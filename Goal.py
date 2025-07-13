class Goal:


    def OnTriggerEnter(self, other):

        if other.parent.name == "body":
            print("🎉 You reached the goal!")

            other.parent.reset_to_default()

            # You can put other logic here, e.g.:
            # - Advance level
            # - Show UI
            # - Play sound
