import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

FILE = "jump_data.csv"

# 如果文件不存在，就创建文件
if not os.path.exists(FILE):
    with open(FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "date",
            "hang_time",
            "jump_height",
            "standing_reach",
            "max_reach",
            "distance_to_dunk"
        ])


while True:

    # 输入滞空时间
    try:
        t = float(input("hang_time: "))
    except ValueError:
        print("wrong value, try again")
        continue

    if t <= 0 or t > 1.2:
        print("impossible")
        continue

    # 计算弹跳高度
    g = 9.8
    h = (1 / 8) * g * t ** 2
    hc = h * 100

    print(f"jump height: {hc:.2f} cm")

    # 输入站立摸高
    try:
        standing_reach = float(input("standing_reach: "))
    except ValueError:
        print("wrong value, try again")
        continue

    if standing_reach <= 0:
        print("that_cannot_be_true")
        continue

    # 计算最高摸高
    max_reach = standing_reach + hc

    # 计算距离扣篮还有多少
    how_far_away_from_dunking = 320 - max_reach

    if how_far_away_from_dunking <= 0:
        print("congratulations")
    else:
        print(
            f"keep_working, "
            f"{how_far_away_from_dunking:.2f} cm to dunk"
        )

    # 判断弹跳
    if hc >= 80:
        print("excellent")
    else:
        print("train_more")

    # 获取当前日期
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 保存数据
    with open(FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            date,
            t,
            round(hc, 2),
            standing_reach,
            round(max_reach, 2),
            round(how_far_away_from_dunking, 2)
        ])

    print("data saved!")

    # 是否查看曲线
    show_graph = input(
        "look the progress curve (y/n): "
    )

    if show_graph.lower() == "y":

        dates = []
        jump_heights = []

        # 读取历史数据
        with open(FILE, "r", encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:
                dates.append(row["date"])
                jump_heights.append(
                    float(row["jump_height"])
                )

        # 画图
        plt.figure(figsize=(10, 5))

        plt.plot(
            dates,
            jump_heights,
            marker="o",
            linewidth=2
        )

        plt.xlabel("Date")
        plt.ylabel("Jump Height (cm)")
        plt.title("Vertical Jump Progress")

        plt.grid(True)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()