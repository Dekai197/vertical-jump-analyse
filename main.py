g =9.8
t=float(input("hang_time:"))
h=(1/8)*g*t**2
hc=h*100
print(hc)
standing_reach=float(input("sr:"))
how_far_away_from_dunking=320-(standing_reach+hc)
if how_far_away_from_dunking<=0:
    print("congratulations")
else:
    print("keep_working")
if hc>=80:
    print("excellent")
else:
    print("train_more")