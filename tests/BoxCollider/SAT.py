from bereshit import Object, Vector3, BoxCollider
from bereshit.tests.GraphTimeComplexity import benchmark_function

obj1 = Object(position=Vector3(0,0,0)).add_component(BoxCollider())

obj2 = Object(
    position=Vector3(0.1,-0.3,0.2),
    rotation=Vector3(10,20,34)
).add_component(BoxCollider())

col1 = obj1.Collider
col2 = obj2.Collider

for i in range(100000):
    BoxCollider.check_collision(col1, col2)

# benchmark_function(
#     [10, 100, 1000, 5000, 10000, 100000],
#     BoxCollider.check_collision,
#     col1,
#     col2
# )