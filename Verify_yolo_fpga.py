#如果在工作站跑需要加下面兩行
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import math

conv_config = [210, 32, 3, 208, 64]
tile = [0, 0, 0, 0]   # I：8bits、W：4bits、O：16bits

rd = []
num = []
cache = (4.9*1024*1024/16)/2   # 4.9 Mbits

x = []
y = []

def yolo_conv():
	t_ihsize = tile[0]-1+conv_config[2];
	t_iwsize = tile[1]-1+conv_config[2];

	rd.append(t_ihsize*t_iwsize*conv_config[1] + conv_config[2]*conv_config[2]*conv_config[1]*tile[3] + tile[0]*tile[1]*tile[3])
	rd.append(t_ihsize*t_iwsize*conv_config[1] + conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4] + tile[0]*tile[1]*conv_config[4])
	rd.append(t_ihsize*t_iwsize*tile[2] + conv_config[2]*conv_config[2]*tile[2]*tile[3] + tile[0]*tile[1]*tile[3])

	num.append((conv_config[4]/tile[3])-1)
	num.append((conv_config[3]/tile[0])*(conv_config[3]/tile[1])-1)
	num.append((conv_config[1]/tile[2])-1)
	num[0] = num[0]*conv_config[0]*conv_config[0]*conv_config[1];
	num[1] = num[1]*conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4];
	num[2] = num[2]*conv_config[3]*conv_config[3]*conv_config[4];

	# for i in range(3):
	# 	print("rd = ", rd[i], "\tnum = ", num[i])

def compute_hit_miss():
	hit_sum = 0
	miss_sum = 0

	for i in range(3):
		if rd[i] <= cache :
			hit_sum = hit_sum + num[i];
		else:
			miss_sum = miss_sum + num[i];

	tile_I = (tile[0]-1+conv_config[2]) * (tile[1]-1+conv_config[2]) * tile[2];
	tile_W = conv_config[2] * conv_config[2] * tile[2] * tile[3];
	tile_O = tile[0] * tile[1] * tile[3];
	tile_size = tile_I + tile_O;

	i_block_num = ((((tile[0]+2)*(tile[1]+2)*8)/(36*1024))+1)*tile[2]
	o_block_num = (((tile[0]*tile[1])*16)/(36*1024)+1)*tile[3]
	num_block = (int)((i_block_num + o_block_num)*2)

	if (tile[2] == 8) and (tile[3] == 32):
		print("<tr, tc, tn, tm> = <%d, %d, %d, %d>" %(tile[0], tile[1], tile[2], tile[3]));
		print("tile_size = ", tile_size, "num_block_ram = ", num_block)
		print("============> hit_sum = ", hit_sum, "\tmiss_sum = ", miss_sum, "\n");
		print("\n")
		x.append(tile_size)
		y.append(miss_sum)

def run():
	print("on-chip buffer can hold ", cache, "data\n\n")

	count = 1
	for tr in range(1, conv_config[3]+1):
		if conv_config[3] % tr == 0:
			for tc in range(1, conv_config[3]+1):
				if conv_config[3] % tc == 0:
					for tn in range(1, conv_config[1]+1):
						if conv_config[1] % tn == 0:
							for tm in range(1, conv_config[4]+1):
								if conv_config[4] % tm == 0:

									tile_I = (tr-1+conv_config[2]) * (tc-1+conv_config[2]) * tn
									tile_W = conv_config[2] * conv_config[2] * tn * tm
									tile_O = tr * tc * tm

									if (tile_I + tile_O) <= cache:
										rd.clear()
										num.clear()
										tile[0] = tr
										tile[1] = tc
										tile[2] = tn
										tile[3] = tm
										yolo_conv()
										compute_hit_miss()
										count = count + 1

def rd_hist():
	total_data_size = conv_config[0]*conv_config[0]*conv_config[1] + conv_config[2]*conv_config[2]*conv_config[1]*conv_config[4] + conv_config[3]*conv_config[3]*conv_config[4]
	print(total_data_size)
	for i in range(len(y)):
		y[i] = y[i] + total_data_size
		y[i] = math.log(y[i], 2)

	plt.figure()
	plt.title("CONV2")
	plt.xlabel("tiled data size")
	plt.ylabel("miss count")
	plt.scatter(x, y, label="output reuse", color="darkgreen")
	plt.legend()
	# plt.show()
	plt.savefig("CONV2.jpg")

if __name__ == '__main__':
	run()
	rd_hist()