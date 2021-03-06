#!/bin/bash -x

# エラー時はエラー終了する
set -e
# 未定義変数参照時はエラーにする
set -u

#echo "start prepare.sh"

# run roscore
gnome-terminal --geometry=50x1 -- roscore
sleep  1

# init judge server, timer window, etc
gnome-terminal --geometry=50x2 --title="windowManager" -- python ../ai_race/sim_environment/scripts/window_management.py
sleep 1

gnome-terminal --geometry=50x1 --title="judgeServer" -- python3 ../judge/judgeServer.py
sleep  1

gnome-terminal --geometry=50x2 --title="keyboard" -- python ../ai_race/utility/scripts/keyboard_con_pygame2.py
sleep 1

gnome-terminal --geometry=50x1 --title="timer" -- python3 ../judge/timer.py
sleep 1

# [future work] if necessary, register some data to server here.

# init simulator, course and vehicle
roslaunch sim_environment wheel_robot_with_surveillance.launch
