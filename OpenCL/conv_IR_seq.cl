__kernel void convolution(__global float* input, __global float* weight, __global float* output, int tr, int tc, int tn, int tm)
{
	for(int oh = 0; (oh+tr-1) < 224; oh += tr)
	{
		for(int ow = 0; (ow+tc-1) < 224; ow += tc)
		{
			for(int ic = 0; ic < 64; ic += tn)
			{
				for(int oc = 0; oc < 64; oc += tm)
				{
					global float* i_base_addr = (input + ic*51076 + oh*226 + ow);
					global float* w_base_addr = (weight + oc*27 + ic*9);
					global float* o_base_addr = (output + oc*50176 + oh*224 + ow);

					for(int t_oc = 0; t_oc < tm; t_oc++)
					{
						for(int t_ic = 0; t_ic < tn; t_ic++)
						{
							for(int t_oh = 0; t_oh < tr; t_oh++)
							{
								for(int t_ow = 0; t_ow < tc; t_ow++)
								{
									global float* i_addr = (i_base_addr + t_ic*51076 + t_oh*226 + t_ow);
									global float* w_addr = (w_base_addr + t_oc*27 + t_ic*9);
									global float* o_addr = (o_base_addr + t_oc*50176 + t_oh*224 + t_ow);
									for(int kh = 0; kh < 3; kh++)
									{
										for(int kw = 0; kw < 3; kw++)
										{
											*(o_addr) += (*(i_addr + kh*226 + kw) * *(w_addr + kh*3 + kw));
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}