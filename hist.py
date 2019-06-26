# 如果在工作站跑需要加下面兩行
# import matplotlib
# matplotlib.use("Agg")
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os

FILE_NUM = 4
FILE_PATH = []
PROD_FIG = True
FIG_TYPE = "bar"
FIG_NAME = []
on_chip_buffer_size = 1283
expand = []
rd_bar_x = []
rd_bar_y = []
x = []
y = []
max_distance = 0

def get_file_name(num):
	for i in range(num):
		FILE_PATH.append(sys.argv[i+1])
		FIG_NAME.append(sys.argv[i+1])

def get_many_file_name():
	for item in os.listdir("./"):
		if "reuse_dist" in item:
			FILE_PATH.append(item)
	return len(FILE_PATH)

def clean_parameters():
	expand.clear()
	rd_bar_x.clear()
	rd_bar_y.clear()
	x.clear()
	y.clear()

def read_mem_access_rd(path, fig_bool, fig_type):
	current_access = 0
	hit_access = 0
	stop_flag = 0

	file_R = open(path, "r")
	while 1:
		line = file_R.readline()
		data = line.split(" ")
		if data[0] == "-1":
			max_distance = distance
			infinite = int(data[1])
			break

		# 取出 distance 和 count
		for i in range(len(data)):
			if i == 0:
				distance = int(data[i])
			if (data[i] != '') and (data[i] != "\n") and (i != 0):
				count = int(data[i])
				break

		# 看進度用
		current_access += count
		# print("distance =", distance, "\tcount =", count)
		if stop_flag == 0:
			if distance >= on_chip_buffer_size:
				hit_access = current_access
				hit_access -= count
				stop_flag = 1

		# 依照要做成什麼圖(hist or bar or plot)來決定怎麼存資料
		if fig_bool:
			# 圖的內容 (用plt.hist)
			if fig_type == "hist":
				if count == 0:
					log_count = 0
				else:
					log_count = int(round(math.log(count, 10)))
				for i in range(log_count):
					expand.append(distance)
			# 圖的內容 (用plt.bar)
			if fig_type == "bar":
				if count == 0:
					log_count = 0
				else:
					log_count = int(round(math.log(count, 10)))
				rd_bar_x.append(distance)
				rd_bar_y.append(log_count)
			# 圖的內容 (用plt.plot)
			if fig_type == "plot":
				x.append(distance)
				y.append(count)
	file_R.close()

	if hit_access == 0:
		hit_access = current_access

	print("reading", path, "finished !!!")
	# print("max_distance =", max_distance)
	print("Total memory access :", (current_access+infinite), "\t(", current_access, " +", infinite, ")")
	print("hit =", hit_access, " (", (hit_access*100/(current_access+infinite)), ")\tmiss =", (current_access+infinite-hit_access))

def get_rd_hist(fig_name):
	plt.figure()
	plt.xlabel("reuse distance")
	plt.ylabel("number")
	# plt.xlim((100, 500))
	if FIG_TYPE == "hist":
		plt.hist(np.array(expand), label = fig_name, color = '#FF5733')
	elif FIG_TYPE == "bar":
		plt.bar(rd_bar_x, rd_bar_y, label = fig_name, color = 'darkgreen')
	elif FIG_TYPE == "plot":
		plt.plot(x, y, label = fig_name, color = 'darkgreen')
	plt.legend()
	plt.savefig(fig_name + ".jpg")

def run():
	get_file_name(FILE_NUM)
	# FILE_NUM = get_many_file_name()
	for i in range(FILE_NUM):
		read_mem_access_rd(FILE_PATH[i], PROD_FIG, FIG_TYPE)
		if PROD_FIG:
			get_rd_hist(FIG_NAME[i])
			clean_parameters()
		print("//========================================//")

	if PROD_FIG:
		plt.show()

	print("total combinations =", FILE_NUM)

if __name__ == '__main__':
	run()