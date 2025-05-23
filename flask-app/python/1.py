unit=int(input("whats the units?"))
if(unit<=100):
   bill=unit*3
elif(unit<200):
   bill=(unit-100)*4+100*3
else:
   bill=(unit-200)*5+100*4+100*3
print(bill)
