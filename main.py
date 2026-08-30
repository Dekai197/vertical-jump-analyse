while True:
    g =9.8
    try:
        t=float(input("hang_time:"))
    except ValueError:
        print("wrong value, try again")
        continue

    if t>1.2:
        print('impossible')
        continue

    h=(1/8)*g*t**2
    hc=h*100
    print(hc)
    try:
        standing_reach=float(input("sr:"))
    except ValueError:
        print("wrong value, try again")
        continue
    if standing_reach<400:
        print("that_cannot_be_true")
        continue
    how_far_away_from_dunking=320-(standing_reach+hc)
    if how_far_away_from_dunking<=0:
        print("congratulations")
    else:
        print("keep_working")
    if hc>=80:
        print("excellent")
    else:
        print("train_more")