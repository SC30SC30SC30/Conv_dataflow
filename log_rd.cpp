#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <stdint.h>

using namespace std;

vector<char> data_type;
vector<uint64_t> data_addr;
vector<int> reuse_distance;

void trace_file_add_rd(char* input_file_name, char* output_file_name)
{
	FILE *fp;
	fp = fopen(input_file_name, "r");
	if(!fp)
	{
		printf("open file error !!!\n");
		exit(1);
	}

	char c;
	uint64_t m_address;
	while(fscanf(fp,"%c %lx\n", &c, &m_address) != EOF )
	{
		// printf("%c %lx\n", c, m_address);
		data_type.push_back(c);
		data_addr.push_back(m_address);
	}
	fclose(fp);
	printf("Read file finish !!!\n");

	vector<uint64_t> only_data_addr;
	bool no_rd = true;
	for(int i = 0; i < data_addr.size(); i++)
	{
		uint64_t target = data_addr[i];
		for(int j = i-1; j >= 0; j--)
		{
			if(target == data_addr[j])
			{
				no_rd = false;
				break;
			}
			else
			{
				bool is_have = false; 
				for(int k = 0; k < only_data_addr.size(); k++)
				{
					if(data_addr[j] == only_data_addr[k])
					{
						is_have = true;
						break;
					}
				}
				if(!is_have)
				{
					only_data_addr.push_back(data_addr[j]);
				}
			}
		}
		if(no_rd)
			reuse_distance.push_back(-1);
		else
			reuse_distance.push_back(only_data_addr.size());

		only_data_addr.clear();
		no_rd = true;
	}

	fp = fopen(output_file_name, "w");
	for(int i = 0; i < data_addr.size(); i++)
	{
		fprintf(fp, "%c %lx %d\n", data_type[i], data_addr[i], reuse_distance[i]);
	}
	fclose(fp);
	printf("Write file finish !!!\n");
}

void clear_global_variable()
{
	data_type.clear();
	data_addr.clear();
	reuse_distance.clear();
}

int main(int argc, char* argv[])
{
	trace_file_add_rd(argv[1], argv[2]);

	return 0;
}