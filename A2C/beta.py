import copy


class Dog:
    def __init__(self,age):
        self.age = age

    def change_age(self,age):
        self.age = age
    

l1=[[Dog(1)],[Dog(2)],[Dog(3)]]
l2=copy.deepcopy(l1)

l1[0][0].change_age(5)
print(l2[0][0].age)